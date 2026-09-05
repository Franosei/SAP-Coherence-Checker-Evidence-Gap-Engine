from __future__ import annotations

import pandas as pd

from src.pipeline import hr_extractor as hx
from src.pipeline.hr_extractor import extract_effect_measures, extract_hr_from_abstract


def _linked(**over) -> pd.DataFrame:
    row = {
        "nct_id": "NCT00000010",
        "pmid": "111",
        "linkage_confidence": "High",
        "primary_result_status": "SELECTED",
        "registration_date": "2015-01-01",
        "abstract_text": "Background. Methods. Results. Conclusions.",
        "published_results": "",
        "published_conclusion": "",
        "published_endpoint": "progression-free survival",
    }
    row.update(over)
    return pd.DataFrame([row])


def test_regex_extraction_reads_the_results_section_first() -> None:
    res = extract_hr_from_abstract(
        abstract_text="Only background here, no numbers.",
        results_text="PFS was longer with the drug (hazard ratio, 0.62; 95% CI, 0.48 to 0.81).",
    )
    assert res.success and res.measure_type == "HR"
    assert res.hr == 0.62 and res.lci == 0.48 and res.uci == 0.81


def test_llm_fallback_used_when_regex_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls: list[str] = []

    def fake_llm(endpoint, full_text):
        calls.append(full_text)
        return hx.ExtractionResult(
            success=True, hr=0.70, lci=0.55, uci=0.89, pattern_name="llm_hr",
            measure_type="HR", source_text="HR 0.70",
        )

    monkeypatch.setattr(hx, "_llm_extract_effect_measure", fake_llm)
    linked = _linked(
        abstract_text="The primary endpoint favoured treatment (p=0.003).",
        published_results="Median PFS 14 vs 9 months; treatment reduced progression.",
    )
    ems = extract_effect_measures(linked, audit_log_path=tmp_path / "audit.csv")

    assert len(calls) == 1
    assert "Median PFS 14 vs 9 months" in calls[0]  # Results section reached the LLM
    assert len(ems) == 1 and ems[0].hr == 0.70


def test_pcr_trial_with_no_ratio_is_recorded_not_crashed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        hx, "_llm_extract_effect_measure",
        lambda e, t: hx.ExtractionResult(
            success=False, pattern_name="llm_none", measure_type="risk_difference",
            failure_reason="Primary endpoint pCR reported as rates 51% vs 26%, no ratio given.",
        ),
    )
    linked = _linked(
        published_endpoint="pathological complete response",
        published_results="pCR was 51% vs 26% (no odds ratio reported).",
    )
    audit = tmp_path / "audit.csv"
    ems = extract_effect_measures(linked, audit_log_path=audit)

    assert ems == []
    log = pd.read_csv(audit)
    assert (log["extraction_method"] == "llm_none").any()


def test_resume_skips_pairs_already_in_the_audit_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    n = {"calls": 0}

    def fake_llm(e, t):
        n["calls"] += 1
        return hx.ExtractionResult(
            success=True, hr=0.8, lci=0.6, uci=0.95, pattern_name="llm_hr", measure_type="HR",
        )

    monkeypatch.setattr(hx, "_llm_extract_effect_measure", fake_llm)
    linked = _linked(abstract_text="No numeric effect estimate in this abstract at all.")
    audit = tmp_path / "audit.csv"

    first = extract_effect_measures(linked, audit_log_path=audit)
    assert n["calls"] == 1 and len(first) == 1

    second = extract_effect_measures(linked, audit_log_path=audit)
    assert n["calls"] == 1  # not called again
    assert len(second) == 1 and second[0].hr == 0.8  # prior result reconstructed from the log

    # The log is rewritten, never appended — one row per pair_id after re-runs.
    log = pd.read_csv(audit)
    assert len(log) == 1
    assert log["pair_id"].is_unique
