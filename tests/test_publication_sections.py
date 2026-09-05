from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd

from src.models.linkage_log import LinkageLog
from src.models.schemas import PrimaryResultStatus, PublicationRole, TrialIdentityStatus
from src.pipeline import publication_family as pf
from src.pipeline.article_classifier import ArticleVerdict, classify_article
from src.pipeline.config import CT_EXCLUDE_POPULATION_CLASSES
from src.pipeline.module1_linker import (
    _classify_population,
    _extract_ctgov_publications,
    _extract_published_endpoint,
    link_to_pubmed,
)
from src.pipeline.publication_family import (
    _int_or_none,
    _pubtype_screen,
    _trial_match,
    build_publication_family,
    select_primary_result,
)
from src.pipeline.pubmed_client import PubMedClient, PubMedRecord


class FakeFamilyClient:
    def __init__(
        self,
        records: dict[str, PubMedRecord],
        exact: list[str] | None = None,
        identity: list[str] | None = None,
        citations: list[str] | None = None,
    ) -> None:
        self.records = records
        self.exact = exact or []
        self.identity = identity or []
        self.citations = citations or []

    def search_by_trial_id(self, nct_id: str) -> list[str]:
        return self.exact

    def search(self, term: str, max_results: int = 50) -> list[str]:
        return self.identity

    def citation_neighbors(self, pmids: list[str], max_results: int = 100) -> list[str]:
        return self.citations

    def fetch_records_batch(self, pmids: list[str]) -> dict[str, PubMedRecord]:
        return {pmid: self.records[pmid] for pmid in pmids if pmid in self.records}


def _fake_llm(verdicts: dict[str, dict], primary: str | None, confidence: str = "high"):
    """Return a stand-in for _llm_link_trial that echoes canned per-candidate verdicts."""

    def _call(row, shortlist):
        return {
            "candidates": [
                {
                    "pmid": c.pmid,
                    "same_trial": verdicts.get(c.pmid, {}).get("same_trial", True),
                    "reports_randomized_arms": True,
                    "reports_outcome_data": verdicts.get(c.pmid, {}).get("outcome_data", True),
                    "role": verdicts.get(c.pmid, {}).get("role", "other"),
                    "analysis_stage": verdicts.get(c.pmid, {}).get("stage", "UNSPECIFIED"),
                    "population_scope": verdicts.get(c.pmid, {}).get("scope", "unknown"),
                    "reason": "test verdict",
                }
                for c in shortlist
            ],
            "primary_results_pmid": primary,
            "confidence": confidence,
            "reason": "test",
        }

    return _call


def test_pubmed_client_parses_sections_identifiers_and_publication_date() -> None:
    xml = """
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <MedlineJournalInfo><MedlineTA>J Clin Trial</MedlineTA></MedlineJournalInfo>
      <Article>
        <ArticleTitle>Primary trial results NCT00000001</ArticleTitle>
        <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        <ArticleDate><Year>2024</Year><Month>05</Month><Day>03</Day></ArticleDate>
        <Abstract>
          <AbstractText Label="Results">The primary endpoint was met.</AbstractText>
          <AbstractText Label="Conclusions">Treatment improved outcomes.</AbstractText>
        </Abstract>
        <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
      </Article>
    </MedlineCitation><PubmedData><ArticleIdList>
      <ArticleId IdType="doi">10.1000/Test</ArticleId>
    </ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>
    """

    record = PubMedClient()._parse_article(ET.fromstring(xml), "123")

    assert record.results_text == "The primary endpoint was met."
    assert record.conclusion_text == "Treatment improved outcomes."
    assert record.pub_types == {"Randomized Controlled Trial"}
    assert record.doi == "10.1000/test"
    assert record.nct_ids == {"NCT00000001"}
    assert record.pub_date == "2024-05-03"


def test_extract_published_endpoint_prefers_results_section() -> None:
    record = PubMedRecord(
        pmid="123",
        title="Primary trial results",
        abstract_text=(
            "BACKGROUND: Background text.\n"
            "RESULTS: The primary endpoint was progression-free survival at 12 months.\n"
            "CONCLUSIONS: The regimen was active."
        ),
        abstract_sections={
            "BACKGROUND": "Background text.",
            "RESULTS": "The primary endpoint was progression-free survival at 12 months.",
            "CONCLUSIONS": "The regimen was active.",
        },
        results_text="The primary endpoint was progression-free survival at 12 months.",
        conclusion_text="The regimen was active.",
    )

    assert (
        _extract_published_endpoint(record)
        == "primary endpoint was progression-free survival at 12 months"
    )


def test_extract_ctgov_publications_keeps_all_pmids_and_best_reference_type() -> None:
    references = [
        {"pmid": "300", "type": "BACKGROUND"},
        {"pmid": "200", "type": "DERIVED"},
        {"pmid": "100", "type": "RESULT"},
        {"pmid": "300", "type": "RESULT"},
        {"citation": "No PMID"},
    ]

    assert _extract_ctgov_publications(references) == [
        ("100", "RESULT"),
        ("300", "RESULT"),
        ("200", "DERIVED"),
    ]


def _trial_row() -> pd.Series:
    return pd.Series(
        {
            "nct_id": "NCT00000002",
            "acronym": "FAMILY",
            "official_title": "A Phase 3 Trial of Drug A in Breast Cancer",
            "conditions": "Breast Cancer",
            "intervention_names": "Drug A | Placebo",
            "arm_names": "Drug A arm | Placebo arm",
            "investigators": "Jane Jones",
            "lead_sponsor": "Example Pharma",
            "phase": "PHASE3",
            "enrollment": "120",
            "start_date": "2016-01-01",
            "primary_completion_date": "2020-01-01",
            "ctgov_publication_pmids": "100 | 200",
            "ctgov_publication_types": "RESULT | DERIVED",
            "primary_outcomes": "Progression-free survival",
        }
    )


def _rec(pmid: str, title: str, abstract: str, **kw) -> PubMedRecord:
    kw.setdefault("nct_ids", {"NCT00000002"})
    return PubMedRecord(pmid=pmid, title=title, abstract_text=abstract, **kw)


def test_pubtype_screen_excludes_protocols_and_syntheses() -> None:
    assert _pubtype_screen({"Clinical Trial Protocol"}) == "excluded"
    assert _pubtype_screen({"Meta-Analysis"}) == "excluded"
    assert _pubtype_screen({"Editorial"}) == "excluded"
    assert _pubtype_screen({"Randomized Controlled Trial"}) == "results_candidate"
    assert _pubtype_screen({"Clinical Trial, Phase III", "Multicenter Study"}) == "results_candidate"
    assert _pubtype_screen(set()) == "neutral"


def test_llm_links_the_primary_paper_and_labels_the_others(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "100": _rec("100", "FAMILY study protocol", "Trial design.", pub_types={"Clinical Trial Protocol"}),
        "200": _rec("200", "FAMILY trial primary analysis", "Primary results. NCT00000002.",
                    pub_types={"Randomized Controlled Trial"}, pub_year="2021"),
        "300": _rec("300", "FAMILY trial final OS analysis", "Final analysis. NCT00000002.", pub_year="2024"),
        "400": _rec("400", "FAMILY trial subgroup analysis", "Subgroup. NCT00000002."),
    }
    monkeypatch.setattr(
        pf,
        "_llm_link_trial",
        _fake_llm(
            {
                "200": {"role": "primary_results", "stage": "PRIMARY", "scope": "complete_randomized"},
                "300": {"role": "final_results", "stage": "FINAL"},
                "400": {"role": "subgroup_posthoc"},
            },
            primary="200",
        ),
    )
    client = FakeFamilyClient(records, exact=["200", "300", "400"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    roles = {e.pmid: e.publication_role for e in family}
    selection = select_primary_result(family)

    assert roles["200"] == PublicationRole.PRIMARY_RESULTS
    assert roles["300"] == PublicationRole.FINAL_RESULTS
    assert roles["400"] == PublicationRole.SUBGROUP_POSTHOC
    assert roles["100"] == PublicationRole.PROTOCOL_SAP  # pubtype-excluded, never sent to LLM
    assert selection.status == PrimaryResultStatus.SELECTED
    assert selection.selected_pmid == "200"
    assert selection.needs_review is False


def test_protocol_pubtype_never_becomes_the_primary_pick(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "100": _rec("100", "FAMILY protocol", "Design and rationale. NCT00000002.",
                    pub_types={"Clinical Trial Protocol"}),
    }
    # Even if the model tried to name the protocol, it was never in the shortlist.
    monkeypatch.setattr(pf, "_llm_link_trial", _fake_llm({}, primary="100"))
    client = FakeFamilyClient(records, exact=["100"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    assert select_primary_result(family).status == PrimaryResultStatus.NOT_FOUND
    assert family[0].pubtype_screen == "excluded"


def test_single_arm_trial_links_when_llm_names_no_pick_but_one_results_paper(monkeypatch) -> None:
    """Single-arm phase 2 trials: the model may not call anything a 'primary analysis'."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "700": _rec(
            "700",
            "Phase II trial of Drug A in metastatic breast cancer",
            "Single-arm study. 35 patients enrolled, clinical benefit rate 72%. NCT00000002.",
            pub_types={"Clinical Trial, Phase II"},
        ),
    }
    # LLM: it IS this trial and reports outcome data, but no explicit pick.
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm({"700": {"role": "primary_results", "outcome_data": True}}, primary=None),
    )
    client = FakeFamilyClient(records, exact=["700"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    selection = select_primary_result(family)
    assert selection.status == PrimaryResultStatus.SELECTED
    assert selection.selected_pmid == "700"
    assert selection.needs_review is True  # fallback pick is always flagged


def test_fallback_never_picks_a_safety_or_qol_paper(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "800": _rec("800", "Safety analysis of the FAMILY trial", "Adverse events. NCT00000002."),
        "801": _rec("801", "Quality of life in the FAMILY trial", "PRO outcomes. NCT00000002."),
    }
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm(
            {"800": {"role": "safety", "outcome_data": True},
             "801": {"role": "qol_pro", "outcome_data": True}},
            primary=None,
        ),
    )
    client = FakeFamilyClient(records, exact=["800", "801"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    assert select_primary_result(family).status == PrimaryResultStatus.NOT_FOUND


def test_llm_says_no_primary_paper_gives_not_found(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {"500": _rec("500", "FAMILY trial safety analysis", "Adverse events. NCT00000002.")}
    monkeypatch.setattr(pf, "_llm_link_trial", _fake_llm({"500": {"role": "safety"}}, primary=None))
    client = FakeFamilyClient(records, exact=["500"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    selection = select_primary_result(family)
    assert selection.status == PrimaryResultStatus.NOT_FOUND
    assert family[0].publication_role == PublicationRole.SAFETY


def test_different_trial_candidate_cannot_be_the_pick(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "200": _rec("200", "FAMILY trial primary analysis", "Primary results. NCT00000002.",
                    pub_types={"Randomized Controlled Trial"}),
        "999": _rec("999", "SIBLING trial primary analysis", "Different trial.",
                    nct_ids={"NCT09999999"}),
    }
    monkeypatch.setattr(
        pf,
        "_llm_link_trial",
        _fake_llm(
            {
                "200": {"role": "primary_results", "same_trial": True, "scope": "complete_randomized"},
                "999": {"role": "primary_results", "same_trial": False},
            },
            primary="200",
        ),
    )
    client = FakeFamilyClient(records, exact=["200"], identity=["999"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    by_pmid = {e.pmid: e for e in family}
    assert by_pmid["999"].trial_identity_status == TrialIdentityStatus.REJECTED
    assert select_primary_result(family).selected_pmid == "200"


def test_exact_nct_in_article_overrides_llm_same_trial_false(monkeypatch) -> None:
    """The NCT number in the abstract IS the identity — an LLM 'no' cannot veto it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "300": _rec(
            "300",
            "Drug X plus fulvestrant in metastatic breast cancer: a randomized Phase II study",
            "The primary endpoint was PFS (hazard ratio 0.87, P=0.67). "
            "Trial registration: ClinicalTrials.gov NCT00000002.",
            pub_types={"Clinical Trial, Phase II", "Randomized Controlled Trial"},
        ),
    }
    # LLM wrongly rejects it and names no pick.
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm({"300": {"role": "primary_results", "same_trial": False}}, primary=None),
    )
    client = FakeFamilyClient(records, exact=["300"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    entry = family[0]
    assert entry.trial_identity_status == TrialIdentityStatus.CONFIRMED
    selection = select_primary_result(family)
    assert selection.status == PrimaryResultStatus.SELECTED
    assert selection.selected_pmid == "300"


def test_low_confidence_pick_is_linked_but_flagged_for_review(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    records = {
        "200": _rec("200", "FAMILY trial results", "Results. NCT00000002.",
                    pub_types={"Randomized Controlled Trial"}),
    }
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm({"200": {"role": "primary_results"}}, primary="200", confidence="low"),
    )
    client = FakeFamilyClient(records, exact=["200"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    selection = select_primary_result(family)
    assert selection.status == PrimaryResultStatus.SELECTED
    assert selection.needs_review is True


def test_no_llm_falls_back_to_single_strong_results_candidate(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    records = {
        "200": _rec(
            "200",
            "FAMILY trial primary analysis",
            "120 patients were randomized. Primary results. NCT00000002.",
            pub_types={"Randomized Controlled Trial"},
        )
    }
    client = FakeFamilyClient(records, exact=["200"])

    family, _ = build_publication_family(_trial_row(), client)  # type: ignore[arg-type]
    selection = select_primary_result(family)
    assert selection.status == PrimaryResultStatus.SELECTED
    assert selection.selected_pmid == "200"
    assert selection.method == "llm"  # method not tracked on entry, but selection defaults
    assert selection.needs_review is True  # heuristic pick always flagged


def test_citation_chaining_runs_from_a_result_seed(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[list[str]] = []

    class TrackingClient(FakeFamilyClient):
        def citation_neighbors(self, pmids, max_results=100):
            seen.append(list(pmids))
            return self.citations

    records = {
        "100": _rec("100", "FAMILY protocol", "Design. NCT00000002.",
                    pub_types={"Clinical Trial Protocol"}),
        "200": _rec("200", "FAMILY trial primary analysis", "Primary results. NCT00000002.",
                    pub_types={"Randomized Controlled Trial"}),
    }
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm({"200": {"role": "primary_results"}}, primary="200"),
    )
    row = _trial_row()
    row["ctgov_publication_pmids"] = "100"
    row["ctgov_publication_types"] = "RESULT"
    client = TrackingClient(records, citations=["200"])

    family, _ = build_publication_family(row, client)  # type: ignore[arg-type]
    assert seen and seen[0] == ["100"]  # chained from the RESULT-tagged seed
    assert select_primary_result(family).selected_pmid == "200"


def test_non_numeric_llm_sample_size_helper() -> None:
    assert _int_or_none("not specified") is None
    assert _int_or_none("N/A") is None
    assert _int_or_none("~800 patients") == 800
    assert _int_or_none("1,240") == 1240
    assert _int_or_none(None) is None


def test_trial_match_identity_prefilter_is_tri_state() -> None:
    strong = _rec("1", "FAMILY trial", "Drug A vs Placebo. NCT00000002.")
    same, score, *_ = _trial_match(_trial_row(), strong, {"pubmed_exact_nct"})
    assert same is True and score >= 0.70

    unrelated = PubMedRecord(pmid="2", title="Unrelated lung study", abstract_text="Nothing here.")
    same2, _, *_ = _trial_match(_trial_row(), unrelated, set())
    assert same2 is False


def test_eligibility_requires_confirmed_subtype_and_setting() -> None:
    confirmed = _classify_population(
        "Neoadjuvant therapy for HER2-positive breast cancer",
        ["HER2-positive breast cancer"],
        "Patients with HER2-positive breast cancer eligible for pre-surgical treatment.",
    )
    assert confirmed == ("bc_confirmed", "her2_positive", "neoadjuvant")

    flagged = _classify_population(
        "A study of Drug A in breast cancer",
        ["HER2-positive breast cancer"],
        "Patients with HER2-positive breast cancer.",
    )
    assert flagged[0] == "bc_flagged"
    assert "bc_flagged" in CT_EXCLUDE_POPULATION_CLASSES

    non_breast = _classify_population("A study in lung cancer", ["Lung cancer"], "NSCLC patients.")
    assert non_breast[0] == "non_breast_excluded"


def test_linkage_checkpoint_resumes_without_reprocessing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        pf, "_llm_link_trial",
        _fake_llm({"200": {"role": "primary_results"}}, primary="200"),
    )
    trials = pd.concat([_trial_row().to_frame().T, _trial_row().to_frame().T], ignore_index=True)
    trials.loc[1, "nct_id"] = "NCT00000003"
    records = {
        "200": _rec("200", "FAMILY trial primary analysis", "Primary results.",
                    pub_types={"Randomized Controlled Trial"}, nct_ids=set()),
    }
    checkpoint = tmp_path / "linked_trials.partial.csv"

    link_to_pubmed(
        trials,
        linkage_log=LinkageLog(tmp_path / "linkage.csv"),
        client=FakeFamilyClient(records, exact=["200"]),  # type: ignore[arg-type]
        publication_family_path=tmp_path / "family.csv",
        checkpoint_path=checkpoint,
    )
    assert len(pd.read_csv(checkpoint, dtype=str)) == 2

    class ExplodingClient:
        def __getattr__(self, name):
            raise AssertionError(f"resume should not call client.{name}")

    resumed = link_to_pubmed(
        trials,
        linkage_log=LinkageLog(tmp_path / "linkage2.csv"),
        client=ExplodingClient(),  # type: ignore[arg-type]
        checkpoint_path=checkpoint,
    )
    assert list(resumed["nct_id"]) == ["NCT00000002", "NCT00000003"]


def test_article_gate_does_not_accept_protocol_language_without_results(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    record = PubMedRecord(
        pmid="456",
        title="A randomized phase 3 breast cancer trial",
        abstract_text=(
            "This study is designed to evaluate treatment. "
            "Patients will be randomly assigned and will be enrolled over 24 months."
        ),
        pub_types={"Clinical Trial, Phase III"},
    )

    assert classify_article(record).verdict != ArticleVerdict.ACCEPT
