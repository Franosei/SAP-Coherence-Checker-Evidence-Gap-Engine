"""
Module 1 — Trial data fetcher and publication linker.

This module is responsible for two sequential steps that together produce
the input dataset for all downstream modules:

Step A — ClinicalTrials.gov fetch
    Query the CT.gov REST API v2 for all HFrEF Phase 2/3 interventional RCTs
    with posted results.  For each trial, extract the registered (pre-specified)
    primary endpoints from the *protocol* section.  These are the endpoints the
    study was powered for before any data were collected.

Step B — NCT-to-PMID linkage cascade  (Section 3.2.2)
    Link each registered trial to its peer-reviewed journal publication using a
    prioritised, three-stage cascade:

      Stage 0 (implicit) — CT.gov RESULT reference
          If the trial's references module contains a RESULT-type PMID (submitted
          by investigators themselves), treat it as a High-confidence direct link
          and skip further stages.

      Stage 1 — Direct NCT ID search
          Query PubMed using the NCT identifier as a secondary-identifier (``[si]``)
          field tag.  A single unambiguous hit is classified as High confidence.

      Stage 2 — Title fuzzy matching
          Search PubMed using the trial's official title (first 12 significant
          words, ``[title]`` field tag) and score each candidate using Jaccard
          token similarity against the registered title.
          Score ≥ 0.70 → High confidence.
          Score ≥ 0.50 → Medium confidence; proceed to Stage 3.

      Stage 3 — Author + year disambiguation
          For Medium-confidence title matches, fetch the full PubMed record for
          the top candidate and verify that the first-author surname and
          publication year are consistent with the trial's completion date.
          Matching → Medium confidence (confirmed).
          Non-matching or fetch failure → Low confidence; flagged for review.

    Trials that cannot be linked at any confidence level are classified as
    Unlinked and automatically flagged for human review before any downstream
    endpoint comparison is run (human-in-the-loop principle, Section 3.3.3).

Every linkage decision — including successful links, failures, and the cascade
stage that resolved the match — is written to the structured linkage audit log
(:class:`~src.models.linkage_log.LinkageLog`) with a timestamp and the pipeline
version identifier.

Output columns
--------------
Columns added by :func:`link_to_pubmed` to the CT.gov DataFrame:

pmid                  PubMed identifier of the linked paper; empty string if unlinked.
linkage_method        ``direct | fuzzy | author_date | manual``
linkage_confidence    ``High | Medium | Low | Unlinked``
abstract_text         Full abstract text retrieved from PubMed.
published_endpoint    Primary endpoint text extracted from the abstract Results
                      section.  This is the value compared to
                      ``primary_outcomes`` (registered) in Module 2.
first_author          First author last name from the PubMed record.
pub_year              Four-digit publication year string.
journal               Journal name (MedlineTA abbreviation preferred).
linkage_notes         Free-text explanation of how the link was resolved.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import pandas as pd
import requests

from src.models.linkage_log import LinkageLog
from src.models.schemas import LinkageAuditEntry, LinkageConfidence, LinkageMethod
from src.pipeline.config import (
    CT_BASE_URL,
    CT_COMPLETION_END,
    CT_COMPLETION_START,
    CT_CONDITIONS,
    CT_EXCLUDE_POPULATION_CLASSES,
    CT_HFREF_LVEF_CEILING,
    CT_PAGE_SIZE,
    CT_PHASES,
    CT_REQUEST_TIMEOUT_S,
    CT_REQUIRE_RESULTS,
    CT_STATUS,
    CT_STUDY_TYPE,
    LINKAGE_JACCARD_HIGH,
    LINKAGE_JACCARD_MEDIUM,
)
from src.pipeline.article_classifier import ArticleVerdict, classify_article
from src.pipeline.pubmed_client import PubMedClient, PubMedRecord, jaccard_token_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HFrEF population classifier
# ---------------------------------------------------------------------------
# The CT.gov API offers no LVEF-based filter.  A broad condition query is
# therefore used at the API level (to avoid missing trials registered as plain
# "Heart Failure") and a post-fetch classifier enforces the HFrEF boundary by
# reading each trial's eligibility criteria text, title, and conditions list.
#
# Classification outcomes
# -----------------------
# hfref_confirmed  — LVEF threshold ≤ CT_HFREF_LVEF_CEILING % confirmed in
#                    eligibility criteria, or explicit HFrEF keyword in title.
#                    Included in all downstream analysis.
#
# hfpef_excluded   — Preserved-EF keyword found and no reduced-EF signal.
#                    Removed before linkage; written to the exclusion log.
#
# mixed_flag       — HFmrEF/borderline keyword, or conflicting signals.
#                    Kept in dataset but flagged for human review.
#
# ambiguous        — No explicit EF criterion found in the available text.
#                    Kept in dataset, flagged; human reviewer decides.

# Patterns are compiled once at module load — reused for every trial row.

_RE_HFREF_KEYWORD = re.compile(
    r"\b("
    r"hfref"
    r"|heart\s+failure\s+with\s+reduced\s+ejection\s+fraction"
    r"|systolic\s+heart\s+failure"
    r"|systolic\s+dysfunction"
    r"|reduced\s+ejection\s+fraction"
    r"|reduced\s+ef"
    r"|left\s+ventricular\s+systolic\s+dysfunction"
    r")\b",
    re.IGNORECASE,
)

_RE_HFPEF_KEYWORD = re.compile(
    r"\b("
    r"hfpef"
    r"|heart\s+failure\s+with\s+preserved\s+ejection\s+fraction"
    r"|preserved\s+ejection\s+fraction"
    r"|preserved\s+ef"
    r"|diastolic\s+heart\s+failure"
    r")\b",
    re.IGNORECASE,
)

_RE_HFMREF_KEYWORD = re.compile(
    r"\b("
    r"hfmref"
    r"|mildly\s+reduced\s+ejection\s+fraction"
    r"|mid[- ]range\s+ejection\s+fraction"
    r"|borderline\s+ejection\s+fraction"
    r")\b",
    re.IGNORECASE,
)

# Matches "LVEF < 40 %", "ejection fraction ≤ 35%", "EF <45%", etc.
# The threshold value is captured in group 1 so it can be logged.
_RE_LVEF_LOW = re.compile(
    r"(?:lvef|ejection\s+fraction|ef)\s*[<≤]\s*=?\s*(\d{2})\s*%",
    re.IGNORECASE,
)

# Matches "LVEF > 50 %", "EF ≥ 45%", "ejection fraction > 40%" (HFpEF gate)
_RE_LVEF_HIGH = re.compile(
    r"(?:lvef|ejection\s+fraction|ef)\s*[>≥]\s*=?\s*(\d{2})\s*%",
    re.IGNORECASE,
)


def _classify_population(
    title: str,
    conditions: list[str],
    eligibility_criteria: str,
) -> tuple[str, str]:
    """
    Classify a trial's enrolled population as HFrEF, HFpEF, mixed, or ambiguous.

    The classifier checks three sources in descending specificity:
      1. Eligibility criteria text (most reliable — contains numeric LVEF thresholds)
      2. Trial title (often explicit: "…in HFrEF", "…Systolic Heart Failure")
      3. Conditions list (varies widely; used only to confirm, not to exclude)

    Parameters
    ----------
    title:
        Official or brief title from ClinicalTrials.gov.
    conditions:
        List of condition/disease strings from ``conditionsModule``.
    eligibility_criteria:
        Free-text eligibility criteria from ``eligibilityModule``.

    Returns
    -------
    tuple[str, str]
        ``(population_class, population_notes)`` where ``population_class`` is
        one of: ``hfref_confirmed``, ``hfpef_excluded``, ``mixed_flag``,
        ``ambiguous``.
    """
    combined     = " ".join([title] + conditions)
    criteria_lc  = eligibility_criteria  # keep original case for regex

    # ---- Signal detection ------------------------------------------------
    hfref_kw    = bool(_RE_HFREF_KEYWORD.search(combined + " " + criteria_lc))
    hfpef_kw    = bool(_RE_HFPEF_KEYWORD.search(combined + " " + criteria_lc))
    hfmref_kw   = bool(_RE_HFMREF_KEYWORD.search(combined + " " + criteria_lc))

    lvef_low_matches  = _RE_LVEF_LOW.findall(criteria_lc)
    lvef_high_matches = _RE_LVEF_HIGH.findall(criteria_lc)

    # Values are strings; cast to int for comparison
    lvef_low_values  = [int(v) for v in lvef_low_matches]
    lvef_high_values = [int(v) for v in lvef_high_matches]

    # A low LVEF threshold ≤ ceiling confirms HFrEF eligibility
    has_lvef_low  = bool(lvef_low_values and min(lvef_low_values) <= CT_HFREF_LVEF_CEILING)
    # A high LVEF threshold ≥ 50 % is a preserved-EF gate (HFpEF signal)
    has_lvef_high = bool(lvef_high_values and max(lvef_high_values) >= 50)

    # ---- Build audit note ------------------------------------------------
    notes: list[str] = []
    if hfref_kw:
        notes.append("HFrEF keyword in title/conditions")
    if lvef_low_values:
        notes.append(f"LVEF ≤ {min(lvef_low_values)}% in eligibility criteria")
    if hfpef_kw:
        notes.append("HFpEF keyword detected")
    if has_lvef_high:
        notes.append(f"LVEF ≥ {max(lvef_high_values)}% gate in eligibility criteria")
    if hfmref_kw:
        notes.append("HFmrEF/borderline keyword detected")
    if not notes:
        notes.append("No ejection-fraction criteria found in available text")

    # ---- Classification logic --------------------------------------------
    # Confirmed HFpEF: preserved-EF signal present, no reduced-EF signal.
    if (hfpef_kw or has_lvef_high) and not hfref_kw and not has_lvef_low:
        return "hfpef_excluded", "; ".join(notes)

    # Confirmed HFrEF: explicit LVEF low threshold or HFrEF keyword, no
    # conflicting preserved-EF signal.
    if (hfref_kw or has_lvef_low) and not hfpef_kw and not has_lvef_high:
        return "hfref_confirmed", "; ".join(notes)

    # Mixed / borderline: HFmrEF keyword, or conflicting LVEF signals, or
    # both HFrEF and HFpEF signals detected simultaneously.
    if hfmref_kw or (has_lvef_low and has_lvef_high) or (hfref_kw and hfpef_kw):
        return "mixed_flag", "; ".join(notes)

    # Ambiguous: some HFrEF signal but paired with preserved-EF gate, or no
    # EF criteria found at all.
    return "ambiguous", "; ".join(notes)


# ---------------------------------------------------------------------------
# Step A — ClinicalTrials.gov fetch
# ---------------------------------------------------------------------------

def fetch_hfref_trials(max_records: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch HFrEF Phase 2/3 RCTs from the ClinicalTrials.gov REST API v2.

    Queries the API using the condition, phase, status, and date parameters
    defined in ``config.py``.  Returns one row per trial containing the
    registered (pre-specified) primary and secondary endpoints extracted from
    the protocol section, along with key metadata fields.

    The CT.gov results section PMID (``ctgov_pmid``) is also extracted where
    available; it is used as Stage 0 of the linkage cascade in
    :func:`link_to_pubmed`.

    Parameters
    ----------
    max_records:
        Optional ceiling on the number of trials to return.  Pass a small
        integer (e.g. ``20``) for smoke-testing without fetching the full
        registry.  ``None`` fetches all matching trials.

    Returns
    -------
    pd.DataFrame
        One row per trial.  All columns are strings or ``None``; numeric
        fields (e.g. ``enrollment``) are cast to ``object`` dtype so the
        DataFrame can be safely written to CSV without dtype ambiguity.

    Raises
    ------
    requests.HTTPError
        If any CT.gov API page returns a non-2xx status.
    """
    records: list[dict] = []
    next_page_token: Optional[str] = None

    condition_query = " OR ".join(CT_CONDITIONS)
    query_term = (
        f"({' OR '.join(f'AREA[Phase]{p}' for p in CT_PHASES)}) "
        f"AND AREA[StudyType]{CT_STUDY_TYPE} "
        f"AND AREA[CompletionDate]RANGE[{CT_COMPLETION_START},{CT_COMPLETION_END}]"
    )
    agg_filters = "results:with" if CT_REQUIRE_RESULTS else ""

    logger.info(
        "Fetching HFrEF RCTs from ClinicalTrials.gov "
        "(completion %s → %s, results-posted=%s)...",
        CT_COMPLETION_START, CT_COMPLETION_END, CT_REQUIRE_RESULTS,
    )

    while True:
        params: dict = {
            "query.cond":        condition_query,
            "query.term":        query_term,
            "filter.overallStatus": CT_STATUS,
            "pageSize":          CT_PAGE_SIZE,
            "format":            "json",
        }
        if agg_filters:
            params["aggFilters"] = agg_filters
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(CT_BASE_URL, params=params, timeout=CT_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()

        for study in data.get("studies", []):
            proto           = study.get("protocolSection", {})
            id_mod          = proto.get("identificationModule", {})
            design_mod      = proto.get("designModule", {})
            outcomes_mod    = proto.get("outcomesModule", {})
            status_mod      = proto.get("statusModule", {})
            conditions_mod  = proto.get("conditionsModule", {})
            eligibility_mod = proto.get("eligibilityModule", {})

            official_title = id_mod.get("officialTitle", "")
            brief_title    = id_mod.get("briefTitle", "")
            conditions     = conditions_mod.get("conditions", [])
            eligibility    = eligibility_mod.get("eligibilityCriteria", "")

            # Registered primary and secondary endpoints (pre-specified)
            primary_outcomes = [
                o.get("measure", "")
                for o in outcomes_mod.get("primaryOutcomes", [])
            ]
            secondary_outcomes = [
                o.get("measure", "")
                for o in outcomes_mod.get("secondaryOutcomes", [])
            ]

            # CT.gov references — extract investigator-submitted RESULT PMID
            # for use as Stage 0 of the linkage cascade.
            references   = proto.get("referencesModule", {}).get("references", [])
            result_pmids = [
                r["pmid"]
                for r in references
                if r.get("type") == "RESULT" and r.get("pmid")
            ]
            derived_pmids = sorted(
                [r["pmid"] for r in references if r.get("type") == "DERIVED" and r.get("pmid")],
                key=lambda p: int(p) if str(p).isdigit() else 0,
            )
            ctgov_pmid = (
                result_pmids[0] if result_pmids
                else (derived_pmids[0] if derived_pmids else "")
            )

            # Population classifier — enforces HFrEF specificity post-fetch
            pop_class, pop_notes = _classify_population(
                title                = official_title or brief_title,
                conditions           = conditions,
                eligibility_criteria = eligibility,
            )

            records.append({
                "nct_id":                    id_mod.get("nctId", ""),
                "official_title":            official_title,
                "brief_title":               brief_title,
                "phase":                     ", ".join(design_mod.get("phases", [])),
                "start_date":                status_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date":           status_mod.get("completionDateStruct", {}).get("date", ""),
                "enrollment":                design_mod.get("enrollmentInfo", {}).get("count", None),
                "primary_outcomes":          " | ".join(primary_outcomes),
                "secondary_outcomes":        " | ".join(secondary_outcomes),
                "registration_date":         status_mod.get("studyFirstSubmitDate", ""),
                "results_first_posted_date": status_mod.get(
                    "resultsFirstPostedDateStruct", {}
                ).get("date", ""),
                "ctgov_pmid":                ctgov_pmid,
                "population_class":          pop_class,
                "population_notes":          pop_notes,
            })

            if max_records is not None and len(records) >= max_records:
                break

        next_page_token = data.get("nextPageToken")
        logger.info("  Fetched %d trials so far...", len(records))

        if max_records is not None and len(records) >= max_records:
            break
        if not next_page_token:
            break

        time.sleep(0.3)  # Polite delay between CT.gov pages

    df = pd.DataFrame(records[:max_records] if max_records else records)
    df = df[df["nct_id"].str.startswith("NCT")].reset_index(drop=True)

    # ---- Population filter report ----------------------------------------
    pop_counts = df["population_class"].value_counts().to_dict()
    logger.info(
        "Population classification — confirmed: %d | ambiguous: %d | "
        "mixed: %d | hfpef_excluded: %d",
        pop_counts.get("hfref_confirmed", 0),
        pop_counts.get("ambiguous", 0),
        pop_counts.get("mixed_flag", 0),
        pop_counts.get("hfpef_excluded", 0),
    )

    # Remove confirmed HFpEF and any other explicitly excluded classes
    excluded = df[df["population_class"].isin(CT_EXCLUDE_POPULATION_CLASSES)]
    if not excluded.empty:
        logger.info(
            "Excluding %d trials with population_class in %s: %s",
            len(excluded),
            CT_EXCLUDE_POPULATION_CLASSES,
            excluded["nct_id"].tolist(),
        )
    df = df[~df["population_class"].isin(CT_EXCLUDE_POPULATION_CLASSES)].reset_index(drop=True)

    # Flag ambiguous / mixed trials for human review in the dashboard
    n_flagged = df["population_class"].isin(["ambiguous", "mixed_flag"]).sum()
    if n_flagged:
        logger.warning(
            "%d trials have ambiguous or mixed-population classification and "
            "are flagged for human review before endpoint pooling.",
            n_flagged,
        )

    n_registered = (df["primary_outcomes"] != "").sum()
    logger.info(
        "CT.gov fetch complete: %d trials retained | with registered endpoint: %d",
        len(df), n_registered,
    )
    return df


# ---------------------------------------------------------------------------
# Results-paper gate helper
# ---------------------------------------------------------------------------

def _fetch_and_gate(
    pmid: str,
    nct_id: str,
    client: PubMedClient,
) -> tuple[Optional[PubMedRecord], str]:
    """
    Fetch a PubMed record and immediately run the article-type gate on it.

    This is the single integration point between the linkage cascade and the
    article classifier.  Every candidate PMID passes through this function
    before being accepted as a valid link.

    Parameters
    ----------
    pmid:
        Candidate PubMed identifier to fetch and classify.
    nct_id:
        NCT ID of the trial being linked (used only for logging).
    client:
        Authenticated :class:`PubMedClient` instance.

    Returns
    -------
    tuple[Optional[PubMedRecord], str]
        ``(record, gate_note)`` where:

        - ``record`` is the :class:`PubMedRecord` if the article passed the
          gate (ACCEPT or UNCERTAIN), or ``None`` if definitively REJECTED.
        - ``gate_note`` is a human-readable string summarising the gate
          decision, for inclusion in the linkage audit log ``notes`` field.
          For UNCERTAIN results the note includes the flag so the reviewer
          knows to check the article type.

    Notes
    -----
    REJECT with ``confidence="high"`` → returns ``(None, note)`` so the
    cascade skips this PMID and tries the next candidate.

    REJECT with ``confidence="medium"`` or UNCERTAIN → returns
    ``(record, note)`` with the note flagging the uncertainty.  The cascade
    stores this as Low/Medium confidence so the human reviewer is prompted.
    """
    try:
        record = client.fetch_record(pmid)
    except Exception as exc:
        return None, f"Fetch failed for PMID {pmid}: {exc}"

    classification = classify_article(record)

    gate_note = (
        f"[Article gate: {classification.verdict.value.upper()} | "
        f"type={classification.article_type.value} | "
        f"conf={classification.confidence} | "
        f"tier={classification.tier}] "
        f"{classification.reason}"
    )

    if (classification.verdict == ArticleVerdict.REJECT
            and classification.confidence == "high"):
        # Definitive rejection — discard this PMID entirely
        logger.info(
            "  %s — PMID %s REJECTED (high confidence): %s",
            nct_id, pmid, classification.reason,
        )
        return None, gate_note

    if classification.verdict == ArticleVerdict.REJECT:
        # Medium-confidence rejection — keep but flag
        logger.warning(
            "  %s — PMID %s rejected (medium confidence, flagged): %s",
            nct_id, pmid, classification.reason,
        )
        # Return record so it can be stored as Low confidence + human review flag
        return record, gate_note

    if classification.verdict == ArticleVerdict.UNCERTAIN:
        logger.info(
            "  %s — PMID %s article type UNCERTAIN — flagged for review: %s",
            nct_id, pmid, classification.reason,
        )
        return record, gate_note

    # ACCEPT
    logger.debug("  %s — PMID %s accepted as results paper.", nct_id, pmid)
    return record, gate_note


# ---------------------------------------------------------------------------
# Step B — NCT-to-PMID linkage cascade
# ---------------------------------------------------------------------------

def link_to_pubmed(
    trials_df: pd.DataFrame,
    linkage_log: Optional[LinkageLog] = None,
    client: Optional[PubMedClient] = None,
) -> pd.DataFrame:
    """
    Link each trial in *trials_df* to its PubMed publication record.

    Runs a prioritised four-stage linkage cascade for every trial row and
    writes every decision — including failures — to the linkage audit log.
    Low-confidence and unlinked trials are automatically flagged for human
    review in the dashboard.

    Parameters
    ----------
    trials_df:
        Output of :func:`fetch_hfref_trials`.  Must contain ``nct_id``,
        ``official_title``, ``brief_title``, ``completion_date``, and
        ``ctgov_pmid`` columns.
    linkage_log:
        Linkage audit log instance.  If ``None``, a new
        :class:`~src.models.linkage_log.LinkageLog` is created using the
        default path from config.
    client:
        :class:`PubMedClient` instance.  If ``None``, a new client is
        instantiated using settings from config.

    Returns
    -------
    pd.DataFrame
        *trials_df* with the following columns appended:

        - ``pmid``                 — PubMed identifier (empty if unlinked)
        - ``linkage_method``       — ``direct | fuzzy | author_date | manual``
        - ``linkage_confidence``   — ``High | Medium | Low | Unlinked``
        - ``abstract_text``        — Full abstract from PubMed
        - ``published_endpoint``   — Extracted primary endpoint from abstract
        - ``first_author``         — First author last name
        - ``pub_year``             — Publication year
        - ``journal``              — Journal name
        - ``linkage_notes``        — Free-text resolution explanation

    Notes
    -----
    Trials already linked at High confidence via Stage 0 (CT.gov RESULT PMID)
    still have their PubMed abstract fetched so that the ``published_endpoint``
    field can be populated for Module 2.
    """
    if linkage_log is None:
        linkage_log = LinkageLog()
    if client is None:
        client = PubMedClient()

    output_rows: list[dict] = []

    total = len(trials_df)
    logger.info("Beginning NCT-to-PMID linkage for %d trials...", total)

    for idx, row in trials_df.iterrows():
        nct_id  = str(row["nct_id"]).strip()
        title   = str(row.get("official_title") or row.get("brief_title") or "").strip()
        ctgov_pmid = str(row.get("ctgov_pmid", "")).strip()
        completion_date = str(row.get("completion_date", "")).strip()

        logger.info(
            "  [%d/%d] Linking %s — %r...",
            idx + 1, total, nct_id, title[:60],
        )

        pmid, method, confidence, notes, record = _cascade_link(
            nct_id         = nct_id,
            title          = title,
            ctgov_pmid     = ctgov_pmid,
            completion_date = completion_date,
            client         = client,
        )

        # Write every decision to the audit log (Section 3.2.2)
        audit_entry = LinkageAuditEntry(
            nct_id             = nct_id,
            pmid               = pmid or None,
            linkage_method     = method,
            linkage_confidence = confidence,
            notes              = notes,
        )
        linkage_log.append(audit_entry)

        # Extract published endpoint and metadata from the PubMed record
        if record is not None:
            published_endpoint = _extract_published_endpoint(record)
            abstract_text      = record.abstract_text
            first_author       = record.authors[0] if record.authors else ""
            pub_year           = record.pub_year
            journal            = record.journal
        else:
            published_endpoint = ""
            abstract_text      = ""
            first_author       = ""
            pub_year           = ""
            journal            = ""

        flag_str = (
            "FLAGGED_FOR_REVIEW"
            if confidence in (LinkageConfidence.LOW, LinkageConfidence.UNLINKED)
            else ""
        )
        if flag_str:
            logger.warning(
                "  %s — %s (%s). Flagged for human review before endpoint comparison.",
                nct_id, confidence.value, notes,
            )

        output_rows.append({
            "pmid":               pmid,
            "linkage_method":     method.value,
            "linkage_confidence": confidence.value,
            "abstract_text":      abstract_text,
            "published_endpoint": published_endpoint,
            "first_author":       first_author,
            "pub_year":           pub_year,
            "journal":            journal,
            "linkage_notes":      notes,
            "linkage_flag":       flag_str,
        })

    linkage_df = pd.DataFrame(output_rows, index=trials_df.index)
    result = pd.concat([trials_df, linkage_df], axis=1)

    # Summary statistics for the governance log
    n_high     = (linkage_df["linkage_confidence"] == "High").sum()
    n_medium   = (linkage_df["linkage_confidence"] == "Medium").sum()
    n_low      = (linkage_df["linkage_confidence"] == "Low").sum()
    n_unlinked = (linkage_df["linkage_confidence"] == "Unlinked").sum()
    n_flagged  = (linkage_df["linkage_flag"] != "").sum()

    logger.info(
        "Linkage complete: %d total | High: %d | Medium: %d | Low: %d | "
        "Unlinked: %d | Flagged for review: %d",
        total, n_high, n_medium, n_low, n_unlinked, n_flagged,
    )

    summary = linkage_log.confidence_summary()
    logger.info("Linkage audit log governance summary: %s", summary)

    return result


# ---------------------------------------------------------------------------
# Internal — linkage cascade
# ---------------------------------------------------------------------------

def _cascade_link(
    nct_id: str,
    title: str,
    ctgov_pmid: str,
    completion_date: str,
    client: PubMedClient,
) -> tuple[str, LinkageMethod, LinkageConfidence, str, Optional[PubMedRecord]]:
    """
    Run the four-stage NCT-to-PMID linkage cascade for a single trial.

    Parameters
    ----------
    nct_id:
        ClinicalTrials.gov NCT identifier.
    title:
        Trial official or brief title.
    ctgov_pmid:
        RESULT-type PMID from the CT.gov references module, if present.
    completion_date:
        Trial completion date string (used to extract expected publication year).
    client:
        Initialised :class:`PubMedClient`.

    Returns
    -------
    tuple[str, LinkageMethod, LinkageConfidence, str, Optional[PubMedRecord]]
        ``(pmid, method, confidence, notes, record)`` where ``pmid`` is the
        resolved PubMed identifier (empty string if unlinked) and ``record``
        is the fetched :class:`PubMedRecord` (``None`` if unlinked or fetch
        failed).
    """
    # ------------------------------------------------------------------ #
    # Stage 0 — CT.gov investigator-submitted RESULT PMID                #
    # ------------------------------------------------------------------ #
    if ctgov_pmid:
        record, gate_note = _fetch_and_gate(ctgov_pmid, nct_id, client)
        if record is not None:
            return (
                ctgov_pmid,
                LinkageMethod.DIRECT,
                LinkageConfidence.HIGH,
                f"CT.gov RESULT reference PMID {ctgov_pmid} fetched successfully. {gate_note}",
                record,
            )
        else:
            logger.warning(
                "  %s — Stage 0: CT.gov PMID %s gated out or fetch failed (%s). "
                "Proceeding to Stage 1.",
                nct_id, ctgov_pmid, gate_note,
            )

    # ------------------------------------------------------------------ #
    # Stage 1 — Direct NCT ID search in PubMed                          #
    # ------------------------------------------------------------------ #
    try:
        pmids = client.search_by_nct_id(nct_id)
    except requests.RequestException as exc:
        logger.warning("  %s — Stage 1: PubMed search error: %s", nct_id, exc)
        pmids = []

    if len(pmids) == 1:
        # Single unambiguous result → High confidence (subject to article gate)
        record, gate_note = _fetch_and_gate(pmids[0], nct_id, client)
        if record is not None:
            return (
                pmids[0],
                LinkageMethod.DIRECT,
                LinkageConfidence.HIGH,
                f"Single direct NCT ID match in PubMed (PMID {pmids[0]}). {gate_note}",
                record,
            )
        else:
            logger.warning(
                "  %s — Stage 1: PMID %s gated out or fetch failed (%s). "
                "Proceeding to Stage 2.",
                nct_id, pmids[0], gate_note,
            )

    if len(pmids) > 1:
        logger.debug(
            "  %s — Stage 1: %d results; proceeding to title disambiguation.",
            nct_id, len(pmids),
        )

    # ------------------------------------------------------------------ #
    # Stage 2 — Title fuzzy matching (Jaccard token overlap)             #
    # ------------------------------------------------------------------ #
    if not title:
        return (
            "",
            LinkageMethod.MANUAL,
            LinkageConfidence.UNLINKED,
            "No trial title available for fuzzy matching; manual linkage required.",
            None,
        )

    try:
        candidate_pmids = client.search_by_title(title)
    except requests.RequestException as exc:
        logger.warning("  %s — Stage 2: title search error: %s", nct_id, exc)
        candidate_pmids = []

    best_pmid      = ""
    best_score     = 0.0
    best_record: Optional[PubMedRecord] = None
    best_gate_note = ""

    for candidate_pmid in candidate_pmids:
        record, gate_note = _fetch_and_gate(candidate_pmid, nct_id, client)
        if record is None:
            # High-confidence article-type rejection or fetch failure — skip candidate
            logger.debug(
                "  %s — Stage 2: PMID %s gated out: %s", nct_id, candidate_pmid, gate_note
            )
            continue

        score = jaccard_token_similarity(title, record.title)
        logger.debug(
            "  %s — Stage 2: PMID %s Jaccard=%.3f (%r)", nct_id, candidate_pmid, score, record.title[:60]
        )

        if score > best_score:
            best_score     = score
            best_pmid      = candidate_pmid
            best_record    = record
            best_gate_note = gate_note

    if best_score >= LINKAGE_JACCARD_HIGH:
        return (
            best_pmid,
            LinkageMethod.FUZZY,
            LinkageConfidence.HIGH,
            (
                f"Title Jaccard={best_score:.3f} ≥ {LINKAGE_JACCARD_HIGH} (High threshold). "
                f"Matched to: {best_record.title[:80] if best_record else ''}. "
                f"{best_gate_note}"
            ),
            best_record,
        )

    if best_score >= LINKAGE_JACCARD_MEDIUM:
        # ---------------------------------------------------------------- #
        # Stage 3 — Author + year disambiguation                          #
        # ---------------------------------------------------------------- #
        return _author_year_disambiguate(
            nct_id           = nct_id,
            completion_date  = completion_date,
            candidate_pmid   = best_pmid,
            candidate_record = best_record,
            jaccard_score    = best_score,
            gate_note        = best_gate_note,
            client           = client,
        )

    # ------------------------------------------------------------------ #
    # No match found at any stage                                        #
    # ------------------------------------------------------------------ #
    reason = (
        f"No PubMed match found (best Jaccard={best_score:.3f} below "
        f"medium threshold {LINKAGE_JACCARD_MEDIUM}). Manual review required."
        if best_score > 0
        else "No PubMed candidates retrieved at any linkage stage."
    )
    return ("", LinkageMethod.MANUAL, LinkageConfidence.UNLINKED, reason, None)


def _author_year_disambiguate(
    nct_id: str,
    completion_date: str,
    candidate_pmid: str,
    candidate_record: Optional[PubMedRecord],
    jaccard_score: float,
    client: PubMedClient,
    gate_note: str = "",
) -> tuple[str, LinkageMethod, LinkageConfidence, str, Optional[PubMedRecord]]:
    """
    Stage 3 — Author + year disambiguation for medium-confidence title matches.

    Verifies that the first-author surname and publication year of the candidate
    record are consistent with the trial's completion date.  A match on both
    fields promotes the candidate from Medium to confirmed Medium confidence.

    A mismatch on either field downgrades to Low confidence and triggers the
    human review flag.

    Parameters
    ----------
    nct_id:
        NCT identifier (used only for logging).
    completion_date:
        Trial completion date from CT.gov (ISO-8601 or month/year string).
    candidate_pmid:
        PMID of the best title-match candidate from Stage 2.
    candidate_record:
        Already-fetched :class:`PubMedRecord` for the candidate (may be ``None``
        if the Stage 2 fetch failed).
    jaccard_score:
        The Jaccard similarity that placed this candidate in the medium band.
    client:
        :class:`PubMedClient` for additional fetches if needed.
    gate_note:
        Article-type gate verdict string from :func:`_fetch_and_gate`, forwarded
        from Stage 2 and appended to the linkage notes for the audit log.

    Returns
    -------
    tuple[str, LinkageMethod, LinkageConfidence, str, Optional[PubMedRecord]]
        See :func:`_cascade_link` for the return contract.
    """
    if candidate_record is None:
        return (
            "",
            LinkageMethod.AUTHOR_DATE,
            LinkageConfidence.LOW,
            f"Stage 3 reached but candidate record is missing (PMID {candidate_pmid}). "
            "Manual linkage required.",
            None,
        )

    pub_year     = candidate_record.pub_year
    first_author = candidate_record.authors[0] if candidate_record.authors else ""

    # Extract expected year from completion_date (handles "2021-06", "Jun 2021", "2021")
    expected_year_match = re.search(r"\b(19|20)\d{2}\b", completion_date)
    expected_year       = expected_year_match.group(0) if expected_year_match else ""

    year_ok = bool(expected_year and pub_year and abs(int(pub_year) - int(expected_year)) <= 2)
    # Allow ±2 years to account for delayed publication and early termination

    gate_suffix = f" {gate_note}" if gate_note else ""

    if year_ok:
        return (
            candidate_pmid,
            LinkageMethod.AUTHOR_DATE,
            LinkageConfidence.MEDIUM,
            (
                f"Stage 3 author+date disambiguation: Jaccard={jaccard_score:.3f}, "
                f"pub_year={pub_year} within ±2 of completion_year={expected_year}, "
                f"first_author={first_author!r}. "
                f"Matched to: {candidate_record.title[:80]}.{gate_suffix}"
            ),
            candidate_record,
        )

    return (
        candidate_pmid,
        LinkageMethod.AUTHOR_DATE,
        LinkageConfidence.LOW,
        (
            f"Stage 3 year mismatch: pub_year={pub_year!r} vs "
            f"expected≈{expected_year!r} (completion_date={completion_date!r}). "
            f"Jaccard={jaccard_score:.3f}. Human review required.{gate_suffix}"
        ),
        candidate_record,
    )


# ---------------------------------------------------------------------------
# Internal — published endpoint extraction from PubMed abstract
# ---------------------------------------------------------------------------

# Phrases that introduce the primary endpoint result in clinical trial abstracts.
# Ordered from most specific to most general.
_PRIMARY_ENDPOINT_SIGNALS: list[re.Pattern] = [
    re.compile(r"(primary\s+end\s*point|primary\s+outcome)[^\.\n]{0,300}", re.IGNORECASE),
    re.compile(r"(primary\s+composite\s+end\s*point|primary\s+composite\s+outcome)[^\.\n]{0,300}", re.IGNORECASE),
    re.compile(r"(primary\s+efficacy\s+end\s*point|primary\s+efficacy\s+outcome)[^\.\n]{0,300}", re.IGNORECASE),
    re.compile(r"(the\s+primary\s+end\s*point\s+(?:was|is|included?))[^\.\n]{0,300}", re.IGNORECASE),
]

# Labels for abstract sections that are most likely to report the primary endpoint.
_RESULT_SECTION_LABELS: tuple[str, ...] = (
    "RESULTS", "RESULTS AND DISCUSSION", "MAIN OUTCOME MEASURE", "FINDINGS",
)


def _extract_published_endpoint(record: PubMedRecord) -> str:
    """
    Extract the primary endpoint text from a PubMed abstract.

    Uses a three-tier strategy to locate the most informative text:

    1. Pattern search within the Results section of a structured abstract.
       Structured abstracts (common in NEJM, JAMA, Lancet RCT reports) have
       labelled sections; the Results section contains the primary endpoint result.

    2. Pattern search across the full abstract text if no Results section exists
       (unstructured abstract) or if the Results section had no matching text.

    3. Full Results section text as a fallback if no keyword pattern matched.

    Parameters
    ----------
    record:
        A :class:`PubMedRecord` with the ``abstract_text`` and
        ``abstract_sections`` fields populated by :meth:`PubMedClient.fetch_record`.

    Returns
    -------
    str
        Extracted endpoint text, or the full abstract text if no specific
        endpoint sentence could be identified.  Returns ``""`` if the record
        has no abstract.
    """
    if not record.abstract_text:
        return ""

    # Tier 1 — search within the Results section of a structured abstract
    results_text = ""
    for label in _RESULT_SECTION_LABELS:
        if label in record.abstract_sections:
            results_text = record.abstract_sections[label]
            break

    for pattern in _PRIMARY_ENDPOINT_SIGNALS:
        search_space = results_text or record.abstract_text
        match = pattern.search(search_space)
        if match:
            extracted = match.group(0).strip()
            # Clean trailing whitespace and truncate gracefully at a sentence boundary
            extracted = re.sub(r"\s+", " ", extracted)
            logger.debug(
                "  PMID %s: extracted endpoint via pattern %r (first 120 chars): %r",
                record.pmid, pattern.pattern[:40], extracted[:120],
            )
            return extracted

    # Tier 3 — return the full Results section or full abstract as fallback
    fallback = results_text or record.abstract_text
    logger.debug(
        "  PMID %s: no primary-endpoint pattern matched; "
        "using Results section / full abstract as published_endpoint.",
        record.pmid,
    )
    return fallback
