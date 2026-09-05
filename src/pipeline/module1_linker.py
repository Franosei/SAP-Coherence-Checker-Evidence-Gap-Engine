"""
Module 1 — Trial data fetcher and publication linker.

This module is responsible for two sequential steps that together produce
the input dataset for all downstream modules:

Step A — ClinicalTrials.gov fetch
    Query the CT.gov REST API v2 for all breast cancer Phase 2/3 interventional
    RCTs with posted results.  For each trial, extract the registered
    (pre-specified) primary endpoints from the *protocol* section.  These are
    the endpoints the study was powered for before any data were collected.

Step B — Publication-family construction
    Discover candidates through registry references, exact-NCT and identity
    searches, and citation chaining. Classify each article's role independently
    of the registered endpoint, verify trial identity, and retain one row per
    publication. Select a primary paper only when exactly one article explicitly
    reports the complete randomized primary analysis.

    Trials that cannot be linked at any confidence level are classified as
    Unlinked and automatically flagged for human review before any downstream
    endpoint comparison is run (human-in-the-loop principle, Section 3.3.3).

Every linkage decision — including all candidate verdicts — is written to the structured linkage audit log
(:class:`~src.models.linkage_log.LinkageLog`) with a timestamp and the pipeline
version identifier.

Output columns
--------------
Columns added by :func:`link_to_pubmed` to the CT.gov DataFrame:

pmid                  Primary-results PMID the LLM committed to; blank if none.
primary_result_status ``SELECTED | NOT_FOUND`` (binary — low-confidence picks are
                      SELECTED with ``PRIMARY_RESULTS_NEEDS_REVIEW`` in linkage_flag)
publication_family_count
                      Number of confirmed/uncertain family members (identity-
                      rejected candidates are excluded from the count but kept
                      in ``publication_family.csv`` for audit).
publication_family_rejected_count
                      Candidates dropped by the trial-identity hard gate.
interim_stage_pmids   Family PMIDs whose analysis stage is interim; never
                      eligible for primary selection regardless of role.
linkage_method        ``publication_family | manual``
linkage_confidence    ``High | Medium | Low | Unlinked``
abstract_text         Full abstract text retrieved from PubMed.
published_results    Results/FINDINGS section text from a structured PubMed
                      abstract, when available.
published_conclusion Conclusion/INTERPRETATION section text from a structured
                      PubMed abstract, when available.
published_endpoint    Primary endpoint text extracted from the abstract Results
                      section.  This is the value compared to
                      the historical registered outcome in Module 2.
first_author          First author last name from the PubMed record.
pub_year              Four-digit publication year string.
journal               Journal name (MedlineTA abbreviation preferred).
linkage_notes         Free-text explanation of how the link was resolved.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.models.linkage_log import LinkageLog
from src.models.schemas import (
    LinkageAuditEntry,
    LinkageConfidence,
    LinkageMethod,
    PrimaryResultStatus,
    PublicationRole,
    TrialIdentityStatus,
)
from src.pipeline.config import (
    CT_BASE_URL,
    CT_COMPLETION_END,
    CT_COMPLETION_START,
    CT_CONDITIONS,
    CT_EXCLUDE_POPULATION_CLASSES,
    CT_PAGE_SIZE,
    CT_PHASES,
    CT_REQUEST_TIMEOUT_S,
    CT_REQUIRE_RESULTS,
    CT_STATUS,
    CT_STUDY_TYPE,
)
from src.pipeline.publication_family import (
    build_publication_family,
    family_role_pmids,
    family_to_frame,
    select_primary_result,
)
from src.pipeline.pubmed_client import PubMedClient, PubMedRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Breast cancer population classifier  (2-D: subtype × treatment setting)
# ---------------------------------------------------------------------------
# The CT.gov API has no breast-cancer subtype filter.  A broad condition query
# is used at the API level and this post-fetch classifier assigns each trial a
# two-dimensional label read from its eligibility criteria, title, and
# conditions list.
#
# Dimension 1 — Tumour subtype  (bc_subtype column)
# --------------------------------------------------
# her2_positive    — HER2+ / HER2-amplified / HER2-overexpressing
# hr_positive      — HR+/HER2- (ER+ and/or PR+, HER2-negative)
# tnbc             — Triple-negative (ER-, PR-, HER2-)
# unknown_subtype  — Subtype not determinable → trial is ineligible
#
# Dimension 2 — Treatment setting  (bc_setting column)
# -----------------------------------------------------
# neoadjuvant      — Pre-surgical (primary) treatment
# adjuvant         — Post-surgical treatment
# metastatic       — Advanced / metastatic / recurrent disease
# unknown_setting  — Setting not determinable → trial is ineligible
#
# Eligibility = confirmed subtype AND confirmed setting. Trials missing either
# ("bc_flagged") and non-breast trials ("non_breast_excluded") are both
# hard-excluded during the fetch; only their NCT IDs are logged.

# Patterns compiled once at import — reused for every trial row.

_RE_BC_HER2_POS = re.compile(
    r"\b("
    r"her2[- ]positive"
    r"|her2[- ]amplified"
    r"|her2[- ]overexpressing"
    r"|her2[+]"
    r"|erbb2[- ]positive"
    r"|erbb2[- ]amplified"
    r"|trastuzumab.{0,40}eligible"  # surrogate signal
    r")\b",
    re.IGNORECASE,
)

_RE_BC_HER2_NEG = re.compile(
    r"\b("
    r"her2[- ]negative"
    r"|her2[- ]low"
    r"|her2\s*[-]\s*(er|pr)\s*positive"  # common shorthand
    r")\b",
    re.IGNORECASE,
)

_RE_BC_TNBC = re.compile(
    r"\b("
    r"triple[- ]negative"
    r"|tnbc"
    r"|er[- ]negative.*pr[- ]negative.*her2[- ]negative"
    r"|er\s*negative\s*and\s*pr\s*negative"  # sometimes HER2 implied
    r")\b",
    re.IGNORECASE,
)

_RE_BC_HR_POS = re.compile(
    r"\b("
    r"hormone\s+receptor[- ]positive"
    r"|hr[+]"
    r"|hr[- ]positive"
    r"|estrogen\s+receptor[- ]positive"
    r"|er[+]"
    r"|er[- ]positive"
    r"|progesterone\s+receptor[- ]positive"
    r"|pr[+]"
    r"|pr[- ]positive"
    r"|luminal"  # luminal A/B subtypes
    r")\b",
    re.IGNORECASE,
)

_RE_BC_NEOADJUVANT = re.compile(
    r"\b("
    r"neoadjuvant"
    r"|pre[- ]surgical"
    r"|pre[- ]operative"
    r"|primary\s+(systemic\s+)?therapy"
    r"|primary\s+treatment"
    r"|preoperative\s+chemotherapy"
    r")\b",
    re.IGNORECASE,
)

_RE_BC_ADJUVANT = re.compile(
    r"\b("
    r"adjuvant"
    r"|post[- ]surgical"
    r"|post[- ]operative"
    r"|after\s+(surgery|resection|mastectomy|lumpectomy)"
    r")\b",
    re.IGNORECASE,
)

_RE_BC_METASTATIC = re.compile(
    r"\b("
    r"metastatic"
    r"|advanced"
    r"|unresectable"
    r"|locally\s+advanced"
    r"|recurrent"
    r"|stage\s+iv"
    r"|stage\s+4"
    r")\b",
    re.IGNORECASE,
)

# Guard against non-breast oncology trials slipping through the broad query.
_RE_BC_BREAST = re.compile(
    r"\b(breast)\b",
    re.IGNORECASE,
)


def _classify_population(
    title: str,
    conditions: list[str],
    eligibility_criteria: str,
) -> tuple[str, str, str]:
    """
    Classify a trial's population along two breast cancer dimensions.

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
    tuple[str, str, str]
        ``(population_class, bc_subtype, bc_setting)`` where:
        - ``population_class`` is ``"bc_confirmed"`` (included — a confirmed
          subtype AND a confirmed setting), ``"non_breast_excluded"``
          (removed — broad search returned a non-breast trial), or
          ``"bc_flagged"`` (removed — subtype or setting not determinable, so
          the trial fails the subtype×setting eligibility criterion).
        - ``bc_subtype`` is one of: ``her2_positive``, ``hr_positive``,
          ``tnbc``, ``unknown_subtype``.
        - ``bc_setting`` is one of: ``neoadjuvant``, ``adjuvant``,
          ``metastatic``, ``unknown_setting``.
    """
    combined = " ".join([title] + conditions + [eligibility_criteria])

    # ---- Guard: is this actually a breast cancer trial? ------------------
    if not _RE_BC_BREAST.search(combined):
        return "non_breast_excluded", "unknown_subtype", "unknown_setting"

    # ---- Subtype detection -----------------------------------------------
    is_tnbc = bool(_RE_BC_TNBC.search(combined))
    is_her2_pos = bool(_RE_BC_HER2_POS.search(combined))
    is_her2_neg = bool(_RE_BC_HER2_NEG.search(combined))
    is_hr_pos = bool(_RE_BC_HR_POS.search(combined))

    # TNBC takes priority: ER-/PR-/HER2- by definition
    if is_tnbc:
        bc_subtype = "tnbc"
    elif is_her2_pos and not is_her2_neg:
        bc_subtype = "her2_positive"
    elif is_hr_pos and (is_her2_neg or not is_her2_pos):
        bc_subtype = "hr_positive"
    else:
        bc_subtype = "unknown_subtype"

    # ---- Setting detection -----------------------------------------------
    is_neoadjuvant = bool(_RE_BC_NEOADJUVANT.search(combined))
    is_adjuvant = bool(_RE_BC_ADJUVANT.search(combined))
    is_metastatic = bool(_RE_BC_METASTATIC.search(combined))

    if is_neoadjuvant and not is_adjuvant and not is_metastatic:
        bc_setting = "neoadjuvant"
    elif is_adjuvant and not is_neoadjuvant and not is_metastatic:
        bc_setting = "adjuvant"
    elif is_metastatic:
        bc_setting = "metastatic"
    elif is_neoadjuvant and is_adjuvant:
        # Both signals present — typical of neo+adjuvant sequential trials
        bc_setting = "neoadjuvant"
    else:
        bc_setting = "unknown_setting"

    # ---- Overall population class ----------------------------------------
    # Eligibility requires a confirmed subtype AND a confirmed setting. Trials
    # missing either dimension are "bc_flagged" and hard-excluded at fetch
    # (see CT_EXCLUDE_POPULATION_CLASSES).
    if bc_subtype == "unknown_subtype" or bc_setting == "unknown_setting":
        population_class = "bc_flagged"
    else:
        population_class = "bc_confirmed"

    return population_class, bc_subtype, bc_setting


# ---------------------------------------------------------------------------
# Step A — ClinicalTrials.gov fetch
# ---------------------------------------------------------------------------

_REFERENCE_PRIORITY = {"RESULT": 0, "DERIVED": 1, "BACKGROUND": 2}


def _extract_ctgov_publications(references: list[dict]) -> list[tuple[str, str]]:
    """Return unique ``(PMID, reference type)`` pairs from CT.gov references."""
    publications: dict[str, str] = {}
    for reference in references:
        pmid = str(reference.get("pmid", "")).strip()
        kind = str(reference.get("type", "")).strip().upper()
        if not pmid.isdigit():
            continue
        current = publications.get(pmid)
        if current is None or _REFERENCE_PRIORITY.get(kind, 99) < _REFERENCE_PRIORITY.get(
            current, 99
        ):
            publications[pmid] = kind or "UNKNOWN"

    return sorted(
        publications.items(),
        key=lambda item: (_REFERENCE_PRIORITY.get(item[1], 99), int(item[0])),
    )


def _format_registered_outcome(outcome: dict) -> str:
    """Preserve the registered measure and timeframe used in coherence checks."""
    measure = str(outcome.get("measure", "")).strip()
    timeframe = str(outcome.get("timeFrame", "")).strip()
    if not measure:
        return ""
    return f"{measure} [Time Frame: {timeframe}]" if timeframe else measure


def fetch_breast_cancer_trials(max_records: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch breast cancer Phase 2/3 RCTs from the ClinicalTrials.gov REST API v2.

    Queries the API using the condition, phase, status, and date parameters
    defined in ``config.py``.  Returns one row per trial containing the
    registered (pre-specified) primary and secondary endpoints extracted from
    the protocol section, along with key metadata fields.

    All PubMed references linked by ClinicalTrials.gov are retained with their
    reference types. PubMed metadata is used later to select the primary
    results publication.

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
        "Fetching breast cancer RCTs from ClinicalTrials.gov "
        "(completion %s to %s, results-posted=%s)...",
        CT_COMPLETION_START,
        CT_COMPLETION_END,
        CT_REQUIRE_RESULTS,
    )

    while True:
        params: dict = {
            "query.cond": condition_query,
            "query.term": query_term,
            "filter.overallStatus": CT_STATUS,
            "pageSize": CT_PAGE_SIZE,
            "format": "json",
        }
        if agg_filters:
            params["aggFilters"] = agg_filters
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(CT_BASE_URL, params=params, timeout=CT_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()

        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            design_mod = proto.get("designModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            status_mod = proto.get("statusModule", {})
            conditions_mod = proto.get("conditionsModule", {})
            eligibility_mod = proto.get("eligibilityModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            sponsors_mod = proto.get("sponsorCollaboratorsModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            official_title = id_mod.get("officialTitle", "")
            brief_title = id_mod.get("briefTitle", "")
            conditions = conditions_mod.get("conditions", [])
            eligibility = eligibility_mod.get("eligibilityCriteria", "")
            intervention_names = [
                intervention.get("name", "")
                for intervention in arms_mod.get("interventions", [])
                if intervention.get("name")
            ]
            arm_names = [
                arm.get("label", "") for arm in arms_mod.get("armGroups", []) if arm.get("label")
            ]
            investigators = [
                official.get("name", "")
                for official in contacts_mod.get("overallOfficials", [])
                if official.get("name")
            ]
            sites = [
                " / ".join(
                    str(location.get(key, "")).strip()
                    for key in ("facility", "city", "country")
                    if location.get(key)
                )
                for location in contacts_mod.get("locations", [])
            ]

            # Registered primary and secondary endpoints (pre-specified)
            primary_outcomes = [
                value
                for outcome in outcomes_mod.get("primaryOutcomes", [])
                if (value := _format_registered_outcome(outcome))
            ]
            secondary_outcomes = [
                value
                for outcome in outcomes_mod.get("secondaryOutcomes", [])
                if (value := _format_registered_outcome(outcome))
            ]

            publications = _extract_ctgov_publications(
                proto.get("referencesModule", {}).get("references", [])
            )

            # Population classifier — assigns breast cancer subtype and setting
            pop_class, bc_subtype, bc_setting = _classify_population(
                title=official_title or brief_title,
                conditions=conditions,
                eligibility_criteria=eligibility,
            )

            records.append(
                {
                    "nct_id": id_mod.get("nctId", ""),
                    "official_title": official_title,
                    "brief_title": brief_title,
                    "acronym": id_mod.get("acronym", ""),
                    "phase": ", ".join(design_mod.get("phases", [])),
                    "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                    "primary_completion_date": status_mod.get(
                        "primaryCompletionDateStruct", {}
                    ).get("date", ""),
                    "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                    "enrollment": design_mod.get("enrollmentInfo", {}).get("count", None),
                    "conditions": " | ".join(conditions),
                    "intervention_names": " | ".join(intervention_names),
                    "arm_names": " | ".join(arm_names),
                    "investigators": " | ".join(investigators),
                    "sites": " | ".join(value for value in sites if value),
                    "lead_sponsor": sponsors_mod.get("leadSponsor", {}).get("name", ""),
                    "primary_outcomes": " | ".join(primary_outcomes),
                    "secondary_outcomes": " | ".join(secondary_outcomes),
                    "registration_date": status_mod.get("studyFirstSubmitDate", ""),
                    "results_first_posted_date": status_mod.get(
                        "resultsFirstPostedDateStruct", {}
                    ).get("date", ""),
                    # ``ctgov_pmid`` remains as a compatibility alias for older
                    # dashboard/output files. The plural columns are authoritative.
                    "ctgov_pmid": publications[0][0] if publications else "",
                    "ctgov_publication_pmids": " | ".join(pmid for pmid, _ in publications),
                    "ctgov_publication_types": " | ".join(kind for _, kind in publications),
                    "ctgov_publication_urls": " | ".join(
                        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid, _ in publications
                    ),
                    "population_class": pop_class,
                    "bc_subtype": bc_subtype,
                    "bc_setting": bc_setting,
                }
            )

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
    subtype_cts = df["bc_subtype"].value_counts().to_dict()
    setting_cts = df["bc_setting"].value_counts().to_dict()
    logger.info(
        "Population classification — bc_confirmed: %d | bc_flagged: %d | non_breast_excluded: %d",
        pop_counts.get("bc_confirmed", 0),
        pop_counts.get("bc_flagged", 0),
        pop_counts.get("non_breast_excluded", 0),
    )
    logger.info(
        "Subtype breakdown — HER2+: %d | HR+/HER2-: %d | TNBC: %d | unknown: %d",
        subtype_cts.get("her2_positive", 0),
        subtype_cts.get("hr_positive", 0),
        subtype_cts.get("tnbc", 0),
        subtype_cts.get("unknown_subtype", 0),
    )
    logger.info(
        "Setting breakdown — neoadjuvant: %d | adjuvant: %d | metastatic: %d | unknown: %d",
        setting_cts.get("neoadjuvant", 0),
        setting_cts.get("adjuvant", 0),
        setting_cts.get("metastatic", 0),
        setting_cts.get("unknown_setting", 0),
    )

    # Hard-exclude ineligible trials: non-breast, and breast trials the
    # classifier cannot place on both the subtype and setting axes. Eligibility
    # is a confirmed subtype AND a confirmed setting.
    for reason in CT_EXCLUDE_POPULATION_CLASSES:
        dropped = df[df["population_class"].eq(reason)]
        if not dropped.empty:
            logger.info(
                "Excluding %d trials (%s): %s",
                len(dropped),
                reason,
                dropped["nct_id"].tolist(),
            )
    df = df[~df["population_class"].isin(CT_EXCLUDE_POPULATION_CLASSES)].reset_index(drop=True)

    n_registered = (df["primary_outcomes"] != "").sum()
    logger.info(
        "CT.gov fetch complete: %d eligible breast cancer trials retained "
        "(confirmed subtype × setting) | with registered endpoint: %d",
        len(df),
        n_registered,
    )
    return df


def fetch_hfref_trials(max_records: Optional[int] = None) -> pd.DataFrame:
    """
    Backward-compatible alias retained for older callers.

    The v3 pipeline now targets breast cancer. Existing scripts that still
    import ``fetch_hfref_trials`` are redirected to
    :func:`fetch_breast_cancer_trials` so the entrypoint does not break during
    the indication transition.
    """
    logger.warning(
        "fetch_hfref_trials() is deprecated in v3.0 and now redirects to "
        "fetch_breast_cancer_trials()."
    )
    return fetch_breast_cancer_trials(max_records=max_records)


# ---------------------------------------------------------------------------
# Step B — select a results paper from CT.gov-linked publications
# ---------------------------------------------------------------------------


def link_to_pubmed(
    trials_df: pd.DataFrame,
    linkage_log: Optional[LinkageLog] = None,
    client: Optional[PubMedClient] = None,
    publication_family_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build each trial's publication family and let one LLM call name its primary paper.

    The LLM never sees the registered endpoint. Its decision is binary: SELECTED
    (a committed primary-results PMID) or NOT_FOUND. Low-confidence SELECTED
    picks stay SELECTED but carry ``needs_review``.

    ``checkpoint_path`` enables per-trial resume: each trial's combined row is
    appended as it completes, and a re-run skips trials already recorded there.
    """
    linkage_log = linkage_log or LinkageLog()
    client = client or PubMedClient()
    role_cache: dict[str, dict] = {}
    output_rows: list[dict] = []

    done_rows: dict[str, dict] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        prev = pd.read_csv(checkpoint_path, dtype=str, keep_default_na=False)
        done_rows = {str(r["nct_id"]).strip(): r for r in prev.to_dict("records")}
        logger.info(
            "Linkage checkpoint found: %d trial(s) already done in %s — resuming.",
            len(done_rows),
            checkpoint_path,
        )

    total = len(trials_df)
    logger.info("Building publication families for %d trials...", total)
    for position, (_, row) in enumerate(trials_df.iterrows(), start=1):
        nct_id = str(row["nct_id"]).strip()
        if nct_id in done_rows:
            output_rows.append(done_rows[nct_id])
            continue
        logger.info("  [%d/%d] Discovering publications for %s", position, total, nct_id)
        try:
            family, records = build_publication_family(row, client, role_cache=role_cache)
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Sustained network outage (retries in the client are already
            # exhausted). Abort instead of checkpointing this trial as "done
            # with error" — everything processed so far is saved, and a re-run
            # resumes here once connectivity is back.
            logger.error(
                "Network unavailable at trial %d/%d (%s). Stopping — re-run "
                "`python run_pipeline.py` to resume from this NCT. Error: %s",
                position,
                total,
                nct_id,
                exc,
            )
            raise
        except Exception as exc:
            logger.exception("Publication-family construction failed for %s", nct_id)
            family, records = [], {}
            discovery_error = str(exc)
        else:
            discovery_error = ""
        selection = select_primary_result(family)

        family_members = [
            entry for entry in family if entry.trial_identity_status != TrialIdentityStatus.REJECTED
        ]
        rejected_count = len(family) - len(family_members)
        interim_stage_pmids = [entry.pmid for entry in family_members if entry.is_interim]

        selected = selection.status == PrimaryResultStatus.SELECTED
        pmid = selection.selected_pmid if selected else ""
        record = records.get(pmid)
        if not selected:
            confidence = LinkageConfidence.UNLINKED
        elif selection.confidence == "high" and not selection.needs_review:
            confidence = LinkageConfidence.HIGH
        elif selection.confidence == "medium":
            confidence = LinkageConfidence.MEDIUM
        else:
            confidence = LinkageConfidence.LOW
        method = LinkageMethod.PUBLICATION_FAMILY if selected else LinkageMethod.MANUAL

        primary_pmids = family_role_pmids(family, PublicationRole.PRIMARY_RESULTS)
        final_pmids = family_role_pmids(family, PublicationRole.FINAL_RESULTS)
        interim_pmids = family_role_pmids(family, PublicationRole.INTERIM_RESULTS)
        updated_pmids = family_role_pmids(
            family, PublicationRole.UPDATED_RESULTS, PublicationRole.LONG_TERM_FOLLOWUP
        )
        secondary_pmids = family_role_pmids(
            family,
            PublicationRole.SECONDARY_ENDPOINT,
            PublicationRole.SUBGROUP_POSTHOC,
            PublicationRole.SAFETY,
            PublicationRole.QOL_PRO,
            PublicationRole.BIOMARKER_TRANSLATIONAL,
            PublicationRole.EXTENSION_STUDY,
        )
        subgroup_pmids = family_role_pmids(family, PublicationRole.SUBGROUP_POSTHOC)
        safety_pmids = family_role_pmids(family, PublicationRole.SAFETY)
        qol_pmids = family_role_pmids(family, PublicationRole.QOL_PRO)
        biomarker_pmids = family_role_pmids(family, PublicationRole.BIOMARKER_TRANSLATIONAL)
        long_term_pmids = family_role_pmids(family, PublicationRole.LONG_TERM_FOLLOWUP)
        extension_pmids = family_role_pmids(family, PublicationRole.EXTENSION_STUDY)
        protocol_pmids = family_role_pmids(family, PublicationRole.PROTOCOL_SAP)
        other_pmids = family_role_pmids(family, PublicationRole.OTHER)
        unresolved_pmids = family_role_pmids(family, PublicationRole.UNRESOLVED)
        candidate_details = [
            {
                "pmid": entry.pmid,
                "trial_match_confidence": entry.trial_match_confidence,
                "classification_confidence": entry.classification_confidence,
            }
            for entry in family
            if entry.pmid in selection.candidate_pmids
        ]
        notes = (
            f"Publication family: {len(family_members)} member(s), {rejected_count} rejected. "
            f"Primary-result status={selection.status.value} "
            f"(method={selection.method}, confidence={selection.confidence}, "
            f"needs_review={selection.needs_review}). {selection.reason}"
            + (f" Discovery error: {discovery_error}" if discovery_error else "")
        )
        linkage_log.append(
            LinkageAuditEntry(
                nct_id=nct_id,
                pmid=pmid or None,
                linkage_method=method,
                linkage_confidence=confidence,
                notes=notes,
            )
        )

        if record:
            published_endpoint = _extract_published_endpoint(record)
            published_results = _extract_results_text(record)
            published_conclusion = _extract_conclusion_text(record)
        else:
            published_endpoint = published_results = published_conclusion = ""

        linkage_row = {
                "pmid": pmid,
                "publication_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "publication_family_count": len(family_members),
                "publication_family_rejected_count": rejected_count,
                "primary_result_status": selection.status.value,
                "primary_result_candidates": " | ".join(selection.candidate_pmids),
                "primary_result_candidate_details": json.dumps(candidate_details),
                "interim_stage_pmids": " | ".join(interim_stage_pmids),
                "publication_family_human_review_required": (
                    selection.status != PrimaryResultStatus.SELECTED or selection.needs_review
                ),
                "primary_results_pmid": pmid,
                "primary_results_pmids": " | ".join(primary_pmids),
                "final_results_pmid": final_pmids[0] if len(final_pmids) == 1 else "",
                "final_results_pmids": " | ".join(final_pmids),
                "interim_results_pmids": " | ".join(interim_pmids),
                "updated_results_pmids": " | ".join(updated_pmids),
                "secondary_results_pmids": " | ".join(secondary_pmids),
                "subgroup_pmids": " | ".join(subgroup_pmids),
                "safety_pmids": " | ".join(safety_pmids),
                "qol_pmids": " | ".join(qol_pmids),
                "biomarker_pmids": " | ".join(biomarker_pmids),
                "long_term_followup_pmids": " | ".join(long_term_pmids),
                "extension_study_pmids": " | ".join(extension_pmids),
                "protocol_sap_pmids": " | ".join(protocol_pmids),
                "other_pmids": " | ".join(other_pmids),
                "unresolved_pmids": " | ".join(unresolved_pmids),
                "linkage_method": method.value,
                "linkage_confidence": confidence.value,
                "article_type": PublicationRole.PRIMARY_RESULTS.value if selected else "",
                "pubmed_publication_types": (
                    " | ".join(sorted(record.pub_types)) if record else ""
                ),
                "abstract_text": record.abstract_text if record else "",
                "published_results": published_results,
                "published_conclusion": published_conclusion,
                "published_endpoint": published_endpoint,
                "first_author": record.authors[0] if record and record.authors else "",
                "pub_year": record.pub_year if record else "",
                "pub_date": record.pub_date if record else "",
                "journal": record.journal if record else "",
                "linkage_notes": notes,
                "linkage_flag": (
                    "PUBLICATION_DISCOVERY_ERROR"
                    if discovery_error
                    else (
                        "NO_PRIMARY_RESULTS"
                        if not selected
                        else ("PRIMARY_RESULTS_NEEDS_REVIEW" if selection.needs_review else "")
                    )
                ),
            }

        combined = {**row.to_dict(), **linkage_row}
        output_rows.append(combined)

        if checkpoint_path is not None:
            _append_csv_row(checkpoint_path, combined)
        if publication_family_path is not None:
            _append_csv_frame(publication_family_path, family_to_frame(family))

    # output_rows holds exactly one row per trial, appended in trials_df order
    # (resumed rows in place), so the frame is already aligned.
    return pd.DataFrame(output_rows).reset_index(drop=True)


def _append_csv_row(path: Path, row: dict) -> None:
    """Append one row, writing the header only when the file is first created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_csv_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------------
# Internal — published endpoint extraction from PubMed abstract
# ---------------------------------------------------------------------------

# Phrases that introduce the primary endpoint result in clinical trial abstracts.
# Ordered from most specific to most general.
_PRIMARY_ENDPOINT_SIGNALS: list[re.Pattern] = [
    re.compile(r"(primary\s+end\s*point|primary\s+outcome)[^\.\n]{0,300}", re.IGNORECASE),
    re.compile(
        r"(primary\s+composite\s+end\s*point|primary\s+composite\s+outcome)[^\.\n]{0,300}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(primary\s+efficacy\s+end\s*point|primary\s+efficacy\s+outcome)[^\.\n]{0,300}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(the\s+primary\s+end\s*point\s+(?:was|is|included?))[^\.\n]{0,300}", re.IGNORECASE
    ),
]

# Labels for abstract sections that are most likely to report the primary endpoint.
_RESULT_SECTION_LABELS: tuple[str, ...] = (
    "RESULTS",
    "RESULT",
    "RESULTS AND DISCUSSION",
    "MAIN OUTCOME MEASURE",
    "MAIN OUTCOME MEASURES",
    "MAIN RESULTS",
    "FINDINGS",
    "OUTCOMES",
)

_CONCLUSION_SECTION_LABELS: tuple[str, ...] = (
    "CONCLUSION",
    "CONCLUSIONS",
    "CONCLUSIONS AND RELEVANCE",
    "INTERPRETATION",
    "INTERPRETATIONS",
    "DISCUSSION",
)


def _first_section_text(record: PubMedRecord, labels: tuple[str, ...]) -> str:
    """Return structured abstract text for the requested labels."""
    sections = record.abstract_sections or {}
    parts = [sections[label] for label in labels if sections.get(label)]
    return "\n".join(parts)


def _extract_results_text(record: PubMedRecord) -> str:
    """Return PubMed Results/FINDINGS text when available."""
    structured = getattr(record, "results_text", "")
    return structured or _first_section_text(record, _RESULT_SECTION_LABELS)


def _extract_conclusion_text(record: PubMedRecord) -> str:
    """Return PubMed Conclusion/Interpretation text when available."""
    structured = getattr(record, "conclusion_text", "")
    return structured or _first_section_text(record, _CONCLUSION_SECTION_LABELS)


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
    results_text = _extract_results_text(record)

    for pattern in _PRIMARY_ENDPOINT_SIGNALS:
        search_space = results_text or record.abstract_text
        match = pattern.search(search_space)
        if match:
            extracted = match.group(0).strip()
            # Clean trailing whitespace and truncate gracefully at a sentence boundary
            extracted = re.sub(r"\s+", " ", extracted)
            logger.debug(
                "  PMID %s: extracted endpoint via pattern %r (first 120 chars): %r",
                record.pmid,
                pattern.pattern[:40],
                extracted[:120],
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
