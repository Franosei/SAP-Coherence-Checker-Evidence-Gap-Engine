"""Discover a trial's publications and pick its primary-results paper with an LLM.

Selection is deliberately independent of the registered primary endpoint: the
LLM is never shown any registered outcome measure. It identifies which candidate
reports the trial's **pre-specified primary analysis** using trial identity,
analysis-stage language, and a PubMed publication-type screen. The
registered-vs-published endpoint comparison happens later, in Module 2.

Design (per trial):
  discover candidates  (CT.gov references + exact-NCT search + trial-identity
                        search + citation chaining when a seed exists)
        -> deduplicate (PMID -> DOI -> normalized title)
        -> publication-type screen  (protocol / meta-analysis / editorial ...
                                     are excluded from the results question)
        -> cheap identity prefilter (drop citation-chain noise)
        -> ONE LLM call for the whole shortlist:
             per candidate  {same_trial, reports_randomized_arms,
                             reports_outcome_data, role, analysis_stage,
                             population_scope}
             overall        primary_results_pmid | null, confidence
        -> LINKED (one primary-results pmid) or UNLINKED. No "ambiguous".
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.models.schemas import (
    AnalysisStage,
    PrimaryResultStatus,
    PublicationFamilyEntry,
    PublicationRole,
    TrialIdentityStatus,
)
from src.pipeline.config import LLM_BASE_URL, LLM_MODEL_PRIMARY, llm_sampling_kwargs
from src.pipeline.pubmed_client import PubMedClient, PubMedRecord

logger = logging.getLogger(__name__)

# Forward+backward citation neighbours pulled per trial (only when a seed exists).
_CITATION_NEIGHBOR_LIMIT = 40
# Shortlist size handed to the single per-trial LLM call.
_MAX_CANDIDATES_TO_LLM = 15
# Abstract characters per candidate in the prompt.
_ABSTRACT_CHARS = 3500

# ---------------------------------------------------------------------------
# PubMed publication-type screen
# ---------------------------------------------------------------------------
# "Clinical Trial Protocol" is a stronger negative signal than "Clinical Trial"
# is a positive one, so protocols / evidence syntheses / opinion pieces are
# excluded from the results question outright; RCT-family types make a paper a
# results candidate; everything else is neutral (the LLM still sees it).

_ALWAYS_EXCLUDE_TYPES = {
    "Clinical Trial Protocol",
    "Meta-Analysis",
    "Systematic Review",
    "Network Meta-Analysis",
    "Retracted Publication",
    "Retraction of Publication",
    "Published Erratum",
}
_SOFT_EXCLUDE_TYPES = {
    "Editorial",
    "Comment",
    "News",
    "Case Reports",
    "Guideline",
    "Practice Guideline",
    "Consensus Development Conference",
    "Biography",
    "Portrait",
    "Historical Article",
    "Letter",
    "Review",
    "Address",
    "Congress",
}
_RESULT_LIKE_TYPES = {
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Controlled Clinical Trial",
    "Clinical Trial, Phase I",
    "Clinical Trial, Phase II",
    "Clinical Trial, Phase III",
    "Clinical Trial, Phase IV",
    "Adaptive Clinical Trial",
    "Equivalence Trial",
    "Pragmatic Clinical Trial",
}

_PUBTYPE_RESULTS = "results_candidate"
_PUBTYPE_EXCLUDED = "excluded"
_PUBTYPE_NEUTRAL = "neutral"

_POPULATION_SCOPES = {"complete_randomized", "subset", "unknown"}
_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_LLM_ROLES = {r.value for r in PublicationRole}


def _pubtype_screen(pub_types: set[str]) -> str:
    types = {str(t).strip() for t in pub_types}
    if types & _ALWAYS_EXCLUDE_TYPES:
        return _PUBTYPE_EXCLUDED
    if types & _RESULT_LIKE_TYPES:
        return _PUBTYPE_RESULTS
    if types & _SOFT_EXCLUDE_TYPES:
        return _PUBTYPE_EXCLUDED
    return _PUBTYPE_NEUTRAL


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimarySelection:
    status: PrimaryResultStatus
    selected_pmid: str = ""
    candidate_pmids: tuple[str, ...] = ()
    reason: str = ""
    confidence: str = "low"
    needs_review: bool = True
    method: str = "llm"


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _split(value: object) -> list[str]:
    return [part.strip() for part in _text(value).split("|") if part.strip()]


def _normal_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _int_or_none(value: object) -> Optional[int]:
    """Best-effort integer from arbitrary input ('not specified', 'N/A', '~800')."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else int(value)
    match = re.search(r"\d[\d,]*", str(value))
    return int(match.group(0).replace(",", "")) if match else None


def _coerce_choice(value: object, allowed: set[str], fallback: str) -> str:
    token = re.sub(r"[\s-]+", "_", _text(value).lower())
    return token if token in allowed else fallback


def _coerce_stage(value: object, fallback: AnalysisStage) -> AnalysisStage:
    try:
        return AnalysisStage(str(value).strip().upper())
    except ValueError:
        return fallback


def _coerce_role(value: object) -> PublicationRole:
    token = _text(value).lower()
    return PublicationRole(token) if token in _LLM_ROLES else PublicationRole.OTHER


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _identity_query(row: pd.Series) -> list[str]:
    acronym = _text(row.get("acronym"))
    interventions = _split(row.get("intervention_names"))
    conditions = _split(row.get("conditions"))
    investigators = _split(row.get("investigators"))
    phase = _text(row.get("phase"))
    enrollment = _text(row.get("enrollment"))
    queries: list[str] = []
    if len(acronym) >= 3:
        queries.append(f'"{acronym}"[tiab]')
    if interventions:
        condition = conditions[0] if conditions else "breast cancer"
        queries.append(f'"{interventions[0]}"[tiab] AND "{condition}"[tiab] AND trial[tiab]')
    if investigators and interventions:
        surname = investigators[0].split()[-1]
        queries.append(f'{surname}[au] AND "{interventions[0]}"[tiab]')
    if interventions and conditions and enrollment.isdigit():
        phase_term = f' AND "{phase}"[tiab]' if phase else ""
        queries.append(
            f'"{interventions[0]}"[tiab] AND "{conditions[0]}"[tiab]'
            f"{phase_term} AND {enrollment}[tiab]"
        )
    return queries


def discover_publication_candidates(row: pd.Series, client: PubMedClient) -> dict[str, set[str]]:
    """Registry references + exact-NCT search + trial-identity search."""
    nct_id = _text(row.get("nct_id")).upper()
    candidates: dict[str, set[str]] = {}

    pmids = _split(row.get("ctgov_publication_pmids"))
    types = _split(row.get("ctgov_publication_types"))
    if not pmids:
        legacy_pmid = _text(row.get("ctgov_pmid"))
        if legacy_pmid:
            pmids = [legacy_pmid]
            types = ["RESULT"]
    for index, pmid in enumerate(pmids):
        kind = types[index] if index < len(types) else "UNKNOWN"
        candidates.setdefault(pmid, set()).add(f"ctgov:{kind}")

    try:
        for pmid in client.search_by_trial_id(nct_id):
            candidates.setdefault(pmid, set()).add("pubmed_exact_nct")
    except Exception as exc:
        logger.warning("Exact-NCT PubMed search failed for %s: %s", nct_id, exc)

    for query in _identity_query(row):
        try:
            for pmid in client.search(query, max_results=30):
                candidates.setdefault(pmid, set()).add("trial_identity_search")
        except Exception as exc:
            logger.warning("Trial-identity PubMed search failed for %s: %s", nct_id, exc)
    return candidates


def _deduplicate_records(
    records: dict[str, PubMedRecord],
    sources: dict[str, set[str]],
) -> tuple[dict[str, PubMedRecord], dict[str, set[str]]]:
    """Deduplicate by PMID, then DOI, then normalized title."""
    kept: dict[str, PubMedRecord] = {}
    kept_sources: dict[str, set[str]] = {}
    identity_to_pmid: dict[str, str] = {}
    for pmid, record in records.items():
        identity = f"doi:{record.doi}" if record.doi else f"title:{_normal_title(record.title)}"
        canonical = identity_to_pmid.setdefault(identity, pmid)
        kept.setdefault(canonical, record)
        kept_sources.setdefault(canonical, set()).update(sources.get(pmid, set()))
    return kept, kept_sources


# ---------------------------------------------------------------------------
# Cheap identity prefilter (screening only — the LLM makes the final call)
# ---------------------------------------------------------------------------


def _identity_score(row: pd.Series, record: PubMedRecord, sources: set[str]) -> tuple[float, list[str]]:
    """Coarse 'is this the same trial' score used only to trim citation-chain noise."""
    nct_id = _text(row.get("nct_id")).upper()
    body = f"{record.title}\n{record.abstract_text}".lower()
    evidence: list[str] = []
    score = 0.0

    if nct_id in record.nct_ids or nct_id.lower() in body:
        score += 0.75
        evidence.append("exact NCT identifier in article")

    ref_types = {s.split(":", 1)[1] for s in sources if s.startswith("ctgov:")}
    if "RESULT" in ref_types:
        score += 0.70
        evidence.append("ClinicalTrials.gov RESULT reference")
    elif "DERIVED" in ref_types:
        score += 0.35
        evidence.append("ClinicalTrials.gov DERIVED reference")
    elif "BACKGROUND" in ref_types:
        score += 0.10
        evidence.append("ClinicalTrials.gov BACKGROUND reference")
    if "pubmed_exact_nct" in sources:
        score += 0.35
        evidence.append("PubMed exact-NCT search hit")

    acronym = _text(row.get("acronym"))
    if len(acronym) >= 3 and acronym.lower() in body:
        score += 0.20
        evidence.append("trial acronym match")

    interventions = _split(row.get("intervention_names"))
    matched_int = [v for v in interventions if len(v) > 3 and v.lower() in body]
    if matched_int:
        score += min(0.20, 0.10 * len(matched_int))
        evidence.append("intervention match: " + ", ".join(matched_int[:3]))

    arms = _split(row.get("arm_names"))
    matched_arms = [v for v in arms if len(v) > 3 and v.lower() in body]
    if matched_arms:
        score += min(0.10, 0.05 * len(matched_arms))
        evidence.append("arm match")

    conditions = _split(row.get("conditions"))
    if any(v.lower() in body for v in conditions if len(v) > 4):
        score += 0.05
        evidence.append("condition match")

    investigators = {n.split()[-1].lower() for n in _split(row.get("investigators")) if n}
    if investigators & {a.lower() for a in record.authors}:
        score += 0.10
        evidence.append("investigator/author match")

    other_ncts = record.nct_ids - {nct_id}
    if other_ncts and nct_id not in record.nct_ids:
        score -= 0.40
        evidence.append("different NCT identifier(s): " + ", ".join(sorted(other_ncts)))

    return max(0.0, min(1.0, round(score, 3))), evidence


# Backwards-compatible name kept for tests / callers.
def _trial_match(
    row: pd.Series,
    record: PubMedRecord,
    sources: set[str],
    role_data: Optional[dict] = None,
    *,
    lenient_reject: bool = False,
) -> tuple[Optional[bool], float, list[str], str, list[str], list[str]]:
    score, evidence = _identity_score(row, record, sources)
    reject_below = 0.15 if lenient_reject else 0.30
    same_trial: Optional[bool] = True if score >= 0.70 else (False if score < reject_below else None)
    matched_int = [
        v for v in _split(row.get("intervention_names")) if len(v) > 3 and v.lower() in record.abstract_text.lower()
    ]
    matched_arms = [
        v for v in _split(row.get("arm_names")) if len(v) > 3 and v.lower() in record.abstract_text.lower()
    ]
    return same_trial, score, evidence, "unknown", matched_int, matched_arms


# ---------------------------------------------------------------------------
# The single per-trial LLM linkage call
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    record: PubMedRecord
    sources: set[str] = field(default_factory=set)
    identity_score: float = 0.0
    identity_evidence: list[str] = field(default_factory=list)
    pubtype_screen: str = _PUBTYPE_NEUTRAL

    @property
    def pmid(self) -> str:
        return self.record.pmid


_LINKAGE_SYSTEM_PROMPT = (
    "You identify which published article reports the PRE-SPECIFIED PRIMARY RESULTS "
    "of one clinical trial. The trial may be randomized (parallel arms) OR "
    "single-arm (one cohort, e.g. a phase 2 response-rate study) — both are in "
    "scope. You are given the trial's registry facts and a numbered list of "
    "candidate articles (title, journal, year, PubMed publication types, abstract). "
    "Do NOT consider, infer, guess, or mention any registered outcome measure — you "
    "are not judging endpoint agreement, only which paper is the primary-results "
    "paper.\n\n"
    "For EVERY candidate return: pmid; same_trial (does it report THIS trial — weigh "
    "NCT id, acronym, interventions, design, sample size, enrolment dates, sponsor, "
    "investigators; a different NCT or a clearly different population means false); "
    "reports_randomized_arms (outcomes given by randomized arm — false is normal and "
    "fine for a single-arm trial); reports_outcome_data (actual results, not just "
    "design or rationale); role (one of: primary_results, interim_results, "
    "final_results, updated_results, secondary_endpoint, subgroup_posthoc, safety, "
    "qol_pro, biomarker_translational, long_term_followup, protocol_sap, other); "
    "analysis_stage (PRIMARY, INTERIM, FINAL, UPDATED, LONG_TERM, UNSPECIFIED); "
    "population_scope (complete = all randomized / all enrolled-evaluable; subset = "
    "biomarker / subgroup / per-protocol subset; unknown); reason (one sentence).\n\n"
    "Then set primary_results_pmid to the ONE pmid that first reports this trial's "
    "pre-specified primary outcome for its complete analysis population (all "
    "randomized patients for a randomized trial; the full enrolled/evaluable cohort "
    "for a single-arm trial). Rules: a planned interim analysis is NOT it even if it "
    "reports the primary endpoint; a protocol / design paper is never it; prefer the "
    "earliest complete report and use a later final/updated analysis only when no "
    "earlier one exists; a single-arm trial's main efficacy/response paper counts. "
    "If several candidates qualify, pick the one whose sample size, design and dates "
    "best match the registry. Return null only when NO candidate reports this "
    "trial's primary outcome results.\n\n"
    'Respond with JSON: {"candidates":[{"pmid","same_trial","reports_randomized_arms",'
    '"reports_outcome_data","role","analysis_stage","population_scope","reason"}],'
    '"primary_results_pmid","confidence","reason"}. confidence is high|medium|low.'
)


def _trial_facts_block(row: pd.Series) -> str:
    fields = [
        ("NCT ID", _text(row.get("nct_id"))),
        ("Acronym", _text(row.get("acronym"))),
        ("Official title", _text(row.get("official_title")) or _text(row.get("brief_title"))),
        ("Phase", _text(row.get("phase"))),
        ("Target enrolment", _text(row.get("enrollment"))),
        ("Conditions", "; ".join(_split(row.get("conditions")))),
        ("Interventions", "; ".join(_split(row.get("intervention_names")))),
        ("Arms / groups", "; ".join(_split(row.get("arm_names"))) or "single-arm (no comparator)"),
        ("Lead sponsor", _text(row.get("lead_sponsor"))),
        ("Investigators", "; ".join(_split(row.get("investigators")))),
        ("Start date", _text(row.get("start_date"))),
        ("Primary completion date", _text(row.get("primary_completion_date"))),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def _candidate_block(index: int, cand: _Candidate) -> str:
    r = cand.record
    types = ", ".join(sorted(r.pub_types)) or "(none)"
    header = f"[{index}] PMID {r.pmid} ({r.pub_year or 'n.d.'}) {r.journal}  |  types: {types}"
    return f"{header}\nTitle: {r.title}\nAbstract: {r.abstract_text[:_ABSTRACT_CHARS]}"


def _llm_link_trial(row: pd.Series, shortlist: list[_Candidate]) -> Optional[dict]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.lower().startswith("your_"):
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
        user = (
            "TRIAL\n"
            + _trial_facts_block(row)
            + "\n\nCANDIDATE ARTICLES\n"
            + "\n\n".join(_candidate_block(i, c) for i, c in enumerate(shortlist, start=1))
        )
        response = client.chat.completions.create(
            model=LLM_MODEL_PRIMARY,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LINKAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            **llm_sampling_kwargs(3000),
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("Linkage LLM failed for %s: %s", _text(row.get("nct_id")), exc)
        return None


# ---------------------------------------------------------------------------
# Family assembly
# ---------------------------------------------------------------------------


def _empty_verdict() -> dict:
    return {
        "same_trial": None,
        "reports_randomized_arms": False,
        "reports_outcome_data": False,
        "role": PublicationRole.OTHER,
        "analysis_stage": AnalysisStage.UNSPECIFIED,
        "population_scope": "unknown",
        "reason": "",
        "confidence": "low",
    }


def _verdict_from_llm(raw: dict) -> dict:
    stage = _coerce_stage(raw.get("analysis_stage"), AnalysisStage.UNSPECIFIED)
    role = _coerce_role(raw.get("role"))
    is_interim = stage == AnalysisStage.INTERIM or role == PublicationRole.INTERIM_RESULTS
    same = raw.get("same_trial")
    return {
        "same_trial": bool(same) if same is not None else None,
        "reports_randomized_arms": bool(raw.get("reports_randomized_arms")),
        "reports_outcome_data": bool(raw.get("reports_outcome_data")),
        "role": PublicationRole.INTERIM_RESULTS if is_interim and role == PublicationRole.PRIMARY_RESULTS else role,
        "analysis_stage": AnalysisStage.INTERIM if is_interim else stage,
        "population_scope": _coerce_choice(raw.get("population_scope"), _POPULATION_SCOPES, "unknown"),
        "reason": _text(raw.get("reason")),
        "confidence": _coerce_choice(raw.get("confidence"), _CONFIDENCE_LEVELS, "low"),
    }


def _build_entry(
    nct_id: str,
    cand: _Candidate,
    verdict: dict,
    *,
    is_pick: bool,
    reviewed_by_llm: bool,
) -> PublicationFamilyEntry:
    record = cand.record
    same_trial = verdict["same_trial"]
    has_registry_signal = any(
        s.startswith("ctgov:") or s == "pubmed_exact_nct" for s in cand.sources
    )
    # The exact NCT number appearing in the article (PubMed secondary source id,
    # title, or abstract "Trial registration:" line) IS the trial identity — it
    # overrides an LLM same_trial=false. Only trust it when the article cites a
    # single NCT (a pooled analysis citing several is a different case).
    exact_nct_match = nct_id in record.nct_ids and len(record.nct_ids) == 1

    if cand.pubtype_screen == _PUBTYPE_EXCLUDED:
        identity_status = TrialIdentityStatus.REJECTED
    elif exact_nct_match:
        identity_status = TrialIdentityStatus.CONFIRMED
        same_trial = True
    elif same_trial is True:
        identity_status = TrialIdentityStatus.CONFIRMED
    elif same_trial is False:
        identity_status = TrialIdentityStatus.REJECTED
    elif not reviewed_by_llm and not has_registry_signal:
        # Reached only through speculative routes (citation chaining / identity
        # search), never a registry link or exact-NCT hit, and didn't make the
        # LLM shortlist. Not this trial's paper for our purposes.
        identity_status = TrialIdentityStatus.REJECTED
        same_trial = False
    else:
        identity_status = TrialIdentityStatus.UNCERTAIN

    role = verdict["role"]
    if cand.pubtype_screen == _PUBTYPE_EXCLUDED and role == PublicationRole.OTHER:
        role = (
            PublicationRole.PROTOCOL_SAP
            if "Clinical Trial Protocol" in record.pub_types
            else PublicationRole.OTHER
        )

    needs_review = bool(
        identity_status == TrialIdentityStatus.UNCERTAIN
        or (is_pick and verdict["confidence"] == "low")
        or (reviewed_by_llm and not verdict["reason"] and role == PublicationRole.OTHER)
    )
    ref_types = sorted(s.split(":", 1)[1] for s in cand.sources if s.startswith("ctgov:"))
    reason = verdict["reason"]
    if not reason:
        reason = (
            f"LLM returned no verdict for this shortlisted candidate (id_score={cand.identity_score})."
            if reviewed_by_llm
            else f"Not sent to LLM (pubtype_screen={cand.pubtype_screen}, id_score={cand.identity_score})."
        )

    return PublicationFamilyEntry(
        nct_id=nct_id,
        pmid=record.pmid,
        doi=record.doi,
        title=record.title,
        year=record.pub_year,
        discovery_sources=sorted(cand.sources),
        registry_reference_type=ref_types[0] if ref_types else None,
        registry_result_reference="RESULT" in ref_types,
        same_trial=same_trial,
        trial_identity_status=identity_status,
        trial_match_confidence=cand.identity_score,
        trial_match_evidence=cand.identity_evidence,
        publication_role=role,
        analysis_stage=verdict["analysis_stage"],
        population_scope=verdict["population_scope"],
        is_interim=verdict["analysis_stage"] == AnalysisStage.INTERIM,
        nct_in_article=nct_id in record.nct_ids,
        pubtype_screen=cand.pubtype_screen,
        reports_randomized_arms=verdict["reports_randomized_arms"],
        reports_outcome_data=verdict["reports_outcome_data"],
        is_primary_results_pick=is_pick,
        classification_confidence=verdict["confidence"] if reviewed_by_llm else "low",
        classification_reason=reason,
        human_review_required=needs_review,
    )


def _heuristic_pick(shortlist: list[_Candidate]) -> Optional[str]:
    """No-LLM fallback: one CT.gov-RESULT / exact-NCT results paper reporting randomization."""
    strong = [
        c
        for c in shortlist
        if c.pubtype_screen == _PUBTYPE_RESULTS
        and (c.record.nct_ids or "pubmed_exact_nct" in c.sources or "RESULT" in {
            s.split(":", 1)[1] for s in c.sources if s.startswith("ctgov:")
        })
        and re.search(r"randomi[sz]", c.record.abstract_text, re.I)
    ]
    return strong[0].pmid if len(strong) == 1 else None


def build_publication_family(
    row: pd.Series,
    client: PubMedClient,
    role_cache: Optional[dict[str, dict]] = None,
) -> tuple[list[PublicationFamilyEntry], dict[str, PubMedRecord]]:
    """Discover candidates, screen them, and let one LLM call name the primary paper.

    ``role_cache`` maps PMID -> LLM verdict so a paper reached through several
    trials is classified once. The chosen ``primary_results_pmid`` is encoded on
    the family via ``is_primary_results_pick``; read it with
    :func:`select_primary_result`.
    """
    role_cache = role_cache if role_cache is not None else {}
    nct_id = _text(row.get("nct_id")).upper()

    sources = discover_publication_candidates(row, client)
    records = client.fetch_records_batch(list(sources)) if sources else {}
    records, sources = _deduplicate_records(records, sources)

    # Citation chaining is a last resort. If PubMed's exact-NCT search already
    # returned a paper that carries this trial's NCT number, that IS the trial's
    # publication — chasing ~40 neighbouring citations only adds noise and LLM
    # tokens. Only chain when the direct routes found nothing NCT-anchored.
    have_nct_anchored = any(
        nct_id in rec.nct_ids
        and "pubmed_exact_nct" in sources.get(pmid, set())
        and _pubtype_screen(rec.pub_types) != _PUBTYPE_EXCLUDED
        for pmid, rec in records.items()
    )
    seed_pmids = [
        pmid
        for pmid, s in sources.items()
        if "pubmed_exact_nct" in s or "ctgov:RESULT" in s
    ]
    if seed_pmids and not have_nct_anchored:
        try:
            chained = client.citation_neighbors(seed_pmids[:5], max_results=_CITATION_NEIGHBOR_LIMIT)
        except Exception as exc:
            logger.warning("Citation chaining failed for %s: %s", nct_id, exc)
            chained = []
        new_pmids = [p for p in chained if p not in records]
        if new_pmids:
            for p in new_pmids:
                sources.setdefault(p, set()).add("citation_chain")
            records.update(client.fetch_records_batch(new_pmids))
            records, sources = _deduplicate_records(records, sources)

    candidates: dict[str, _Candidate] = {}
    for pmid, record in records.items():
        srcs = sources.get(pmid, set())
        id_score, id_evidence = _identity_score(row, record, srcs)
        candidates[pmid] = _Candidate(
            record=record,
            sources=srcs,
            identity_score=id_score,
            identity_evidence=id_evidence,
            pubtype_screen=_pubtype_screen(record.pub_types),
        )

    # Shortlist for the LLM: not pubtype-excluded, and either a registry
    # reference / exact-NCT hit or a non-trivial identity score.
    shortlist = sorted(
        (
            c
            for c in candidates.values()
            if c.pubtype_screen != _PUBTYPE_EXCLUDED
            and (
                any(s.startswith("ctgov:") or s == "pubmed_exact_nct" for s in c.sources)
                or c.identity_score >= 0.25
            )
        ),
        key=lambda c: c.identity_score,
        reverse=True,
    )[:_MAX_CANDIDATES_TO_LLM]

    llm_verdicts: dict[str, dict] = {}
    pick_pmid = ""

    # One batched LLM call per trial classifies the whole shortlist and names the
    # primary paper. The pick is trial-specific and cannot be cached; role_cache
    # only backfills per-candidate verdicts the call happens to omit.
    llm_raw: Optional[dict] = _llm_link_trial(row, shortlist) if shortlist else None

    if llm_raw is not None:
        for item in llm_raw.get("candidates", []) or []:
            pmid = _text(item.get("pmid"))
            if pmid in candidates:
                llm_verdicts[pmid] = _verdict_from_llm(item)
        role_cache.update(llm_verdicts)
        raw_pick = _text(llm_raw.get("primary_results_pmid"))
        pick_confidence = _coerce_choice(llm_raw.get("confidence"), _CONFIDENCE_LEVELS, "low")
        if raw_pick in candidates and llm_verdicts.get(raw_pick, {}).get("same_trial") is not False:
            pick_pmid = raw_pick
            llm_verdicts.setdefault(pick_pmid, _empty_verdict())["confidence"] = pick_confidence
    elif shortlist:
        pick_pmid = _heuristic_pick(shortlist) or ""
        if pick_pmid:
            llm_verdicts[pick_pmid] = {
                **_empty_verdict(),
                "same_trial": True,
                "role": PublicationRole.PRIMARY_RESULTS,
                "reports_outcome_data": True,
                "reason": "No LLM available — single strong results candidate (CT.gov/exact-NCT).",
            }

    # Backfill verdicts for shortlisted papers the call omitted (from a prior trial).
    for cand in shortlist:
        if cand.pmid not in llm_verdicts and cand.pmid in role_cache:
            llm_verdicts[cand.pmid] = role_cache[cand.pmid]

    shortlist_pmids = {c.pmid for c in shortlist}
    family: list[PublicationFamilyEntry] = []
    for pmid, cand in candidates.items():
        family.append(
            _build_entry(
                nct_id,
                cand,
                llm_verdicts.get(pmid, _empty_verdict()),
                is_pick=(pmid == pick_pmid),
                reviewed_by_llm=(pmid in llm_verdicts) or (pmid in shortlist_pmids and llm_raw is not None),
            )
        )

    return family, records


# ---------------------------------------------------------------------------
# Selection read-out
# ---------------------------------------------------------------------------


_RESULTS_ROLES = (
    PublicationRole.PRIMARY_RESULTS,
    PublicationRole.FINAL_RESULTS,
    PublicationRole.UPDATED_RESULTS,
)
# Roles that are explicitly NOT the trial's primary-results paper — never
# eligible for the single-confirmed-paper fallback below.
_NON_PRIMARY_ROLES = frozenset(
    {
        PublicationRole.SECONDARY_ENDPOINT,
        PublicationRole.SUBGROUP_POSTHOC,
        PublicationRole.SAFETY,
        PublicationRole.QOL_PRO,
        PublicationRole.BIOMARKER_TRANSLATIONAL,
        PublicationRole.LONG_TERM_FOLLOWUP,
        PublicationRole.EXTENSION_STUDY,
        PublicationRole.PROTOCOL_SAP,
        PublicationRole.INTERIM_RESULTS,
        PublicationRole.UNRESOLVED,
    }
)


def select_primary_result(family: list[PublicationFamilyEntry]) -> PrimarySelection:
    """Read the LLM's decision off the family — LINKED (one pick) or UNLINKED."""
    confirmed = [e for e in family if e.trial_identity_status == TrialIdentityStatus.CONFIRMED]
    picks = [e for e in confirmed if e.is_primary_results_pick]
    if len(picks) == 1:
        pick = picks[0]
        return PrimarySelection(
            status=PrimaryResultStatus.SELECTED,
            selected_pmid=pick.pmid,
            candidate_pmids=(pick.pmid,),
            reason=pick.classification_reason
            or "LLM identified this article as the pre-specified primary analysis.",
            confidence=pick.classification_confidence,
            needs_review=pick.human_review_required,
        )

    # Fallback: the LLM named no pick but there is exactly one confirmed candidate
    # that actually reports this trial's outcome results. Common for single-arm
    # phase 2 trials the model is reluctant to call a "primary analysis".
    if not picks:
        results_papers = [
            e
            for e in confirmed
            if e.publication_role in _RESULTS_ROLES
            and (e.reports_outcome_data or e.nct_in_article)
            and not e.is_interim
        ]
        if not results_papers:
            # Widen to a confirmed paper the LLM left as "other" but which still
            # reports outcome data (single-arm trials it wouldn't call a
            # "primary analysis"); never a safety/QoL/subgroup/etc. paper.
            results_papers = [
                e
                for e in confirmed
                if e.reports_outcome_data
                and not e.is_interim
                and e.publication_role not in _NON_PRIMARY_ROLES
            ]
        # If several qualify, prefer the one(s) that carry the exact NCT number.
        if len(results_papers) > 1:
            nct_tagged = [e for e in results_papers if e.nct_in_article]
            if len(nct_tagged) == 1:
                results_papers = nct_tagged
        if len(results_papers) == 1:
            e = results_papers[0]
            return PrimarySelection(
                status=PrimaryResultStatus.SELECTED,
                selected_pmid=e.pmid,
                candidate_pmids=(e.pmid,),
                reason=(
                    "Confirmed results paper for this trial (exact NCT match) "
                    if e.nct_in_article
                    else "Only one confirmed results paper for this trial; "
                )
                + f"the LLM did not explicitly flag it as the primary analysis "
                f"(role={e.publication_role.value}).",
                confidence="low",
                needs_review=True,
            )

    return PrimarySelection(
        status=PrimaryResultStatus.NOT_FOUND,
        reason="No candidate article reports this trial's pre-specified primary results.",
    )


def family_role_pmids(family: list[PublicationFamilyEntry], *roles: PublicationRole) -> list[str]:
    return [
        entry.pmid
        for entry in family
        if entry.trial_identity_status == TrialIdentityStatus.CONFIRMED
        and entry.publication_role in roles
    ]


def family_to_frame(family: list[PublicationFamilyEntry]) -> pd.DataFrame:
    """Serialize model values and lists into a CSV-friendly long table."""
    rows: list[dict] = []
    for entry in family:
        row = entry.model_dump(mode="json")
        for key, value in row.items():
            if isinstance(value, list):
                row[key] = " | ".join(str(item) for item in value)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(PublicationFamilyEntry.model_fields))
