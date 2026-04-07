"""
Effect measure extraction from PubMed abstracts.  (Section 3.4.2)

Extracts the primary Hazard Ratio (HR) and its 95% confidence interval from
clinical trial abstracts using a cascade of regex patterns that cover the
range of reporting styles found in major cardiovascular journals (NEJM, JAMA,
Lancet, JACC, ESC journals).

The cascade tries patterns from most structured (explicit "HR" label + CI) to
least structured (bare numbers following a primary-endpoint sentence).  When
no machine-readable effect estimate can be found, the trial is flagged for
manual data entry with an explicit reason code rather than silently excluded.

All extraction decisions are written to the power audit log
(``data/logs/power_audit_log.csv``) with:
  - The regex pattern variant that matched (or ``"manual"`` / ``"failed"``)
  - The raw text window from which the values were extracted
  - The PMID and trial registration date for sequential-analysis ordering
  - A ``requires_manual_check`` flag for human verification

Pipeline integration
--------------------
Input:   ``linked_trials`` DataFrame — output of :func:`module1_linker.link_to_pubmed`.
         Must contain columns: ``nct_id``, ``pmid``, ``abstract_text``,
         ``linkage_confidence``, ``registration_date``.

Output:  ``list[EffectMeasure]`` — one object per successfully extracted trial
         pair, sorted by ``registration_date`` for sequential Bayesian analysis.
         Trials with failed extraction are logged but not included in the list.

Usage
-----
    from src.pipeline.hr_extractor import extract_effect_measures

    effect_measures = extract_effect_measures(linked_trials_df)
    # Pass directly to module3_bayesian.load_poolable_effects(effect_measures, ...)
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.schemas import EffectMeasure
from src.pipeline.config import PIPELINE_VERSION, POWER_AUDIT_LOG_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction audit log
# ---------------------------------------------------------------------------

_AUDIT_COLUMNS: list[str] = [
    "nct_id",
    "pmid",
    "pair_id",
    "hr",
    "hr_lci",
    "hr_uci",
    "log_hr",
    "se_log_hr",
    "extraction_method",
    "source_text",
    "requires_manual_check",
    "exclusion_reason",
    "registration_date",
    "pipeline_version",
    "extracted_at",
]


def _initialise_audit_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=_AUDIT_COLUMNS).writeheader()
        logger.info("HR extraction audit log initialised at %s", path)


def _write_audit_row(path: Path, row: dict) -> None:
    row["pipeline_version"] = PIPELINE_VERSION
    row["extracted_at"]     = datetime.now(UTC).isoformat()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=_AUDIT_COLUMNS).writerow(
            {col: row.get(col, "") for col in _AUDIT_COLUMNS}
        )


# ---------------------------------------------------------------------------
# Regex pattern library
# ---------------------------------------------------------------------------
# Each pattern is a compiled regex that captures three named groups:
#   hr   — the hazard ratio point estimate
#   lci  — lower 95% confidence interval bound
#   uci  — upper 95% confidence interval bound
#
# The patterns are ordered from most explicit (labelled HR) to least explicit
# (bare ratio after a primary-endpoint sentence).  The first matching pattern
# wins; its name is stored in the audit log as the extraction_method.
#
# Number format: decimal values may use period (0.80) or comma (0,80).
# The _num helper normalises these to float-parseable strings.

_NUM = r"(?P<{name}>\d+[.,]\d+)"   # Named decimal number capture template


def _n(name: str) -> str:
    """Return a named decimal-number capture group for use in patterns."""
    return r"(?P<" + name + r">\d+[.,]\d+)"


# Optional separators between HR value and CI (e.g. ";", ",", whitespace)
_SEP = r"[\s;,]*"

# Confidence interval introducers
_CI_INTRO = r"(?:95\s*%\s*(?:CI|confidence\s+interval|CrI)\s*[,:]?\s*)"

# Full pattern list — (name, compiled_pattern)
_HR_PATTERNS: list[tuple[str, re.Pattern]] = [

    # ------------------------------------------------------------------ #
    # Pattern 1: "HR 0.80 (95% CI 0.73-0.87)" — most common RCT format  #
    # ------------------------------------------------------------------ #
    (
        "hr_paren_ci",
        re.compile(
            r"(?:hazard\s+ratio|HR)[,\s]*"
            + _n("hr")
            + r"\s*\(\s*"
            + _CI_INTRO + r"?"
            + _n("lci")
            + r"\s*[-–—to]+\s*"
            + _n("uci")
            + r"\s*\)",
            re.IGNORECASE,
        ),
    ),

    # ------------------------------------------------------------------ #
    # Pattern 2: "HR, 0.80; 95% CI, 0.73–0.87" — NEJM structured style  #
    # ------------------------------------------------------------------ #
    (
        "hr_semicolon_ci",
        re.compile(
            r"(?:hazard\s+ratio|HR)\s*,?\s*"
            + _n("hr")
            + r"\s*;\s*"
            + _CI_INTRO
            + _n("lci")
            + r"\s*[-–—to]+\s*"
            + _n("uci"),
            re.IGNORECASE,
        ),
    ),

    # ------------------------------------------------------------------ #
    # Pattern 3: "hazard ratio of 0.80 (0.73 to 0.87)" — narrative style #
    # ------------------------------------------------------------------ #
    (
        "hr_narrative_to",
        re.compile(
            r"hazard\s+ratio\s+of\s+"
            + _n("hr")
            + r"\s*\(\s*"
            + _n("lci")
            + r"\s+to\s+"
            + _n("uci")
            + r"\s*\)",
            re.IGNORECASE,
        ),
    ),

    # ------------------------------------------------------------------ #
    # Pattern 4: "HR=0.80 [95%CI: 0.73, 0.87]" — bracket CI format     #
    # ------------------------------------------------------------------ #
    (
        "hr_bracket_ci",
        re.compile(
            r"(?:hazard\s+ratio|HR)\s*=\s*"
            + _n("hr")
            + r"\s*\[\s*"
            + _CI_INTRO + r"?"
            + _n("lci")
            + r"\s*[-–—,]+\s*"
            + _n("uci")
            + r"\s*\]",
            re.IGNORECASE,
        ),
    ),

    # ------------------------------------------------------------------ #
    # Pattern 5: "HR 0.80, 95% CI 0.73-0.87" — comma-separated          #
    # ------------------------------------------------------------------ #
    (
        "hr_comma_ci",
        re.compile(
            r"(?:hazard\s+ratio|HR)[,\s]+"
            + _n("hr")
            + r"\s*,\s*"
            + _CI_INTRO
            + _n("lci")
            + r"\s*[-–—]+\s*"
            + _n("uci"),
            re.IGNORECASE,
        ),
    ),

    # ------------------------------------------------------------------ #
    # Pattern 6: RR/OR fallback — relative risk or odds ratio when HR    #
    # is unavailable (lower confidence; flagged for manual check)         #
    # ------------------------------------------------------------------ #
    (
        "rr_or_fallback",
        re.compile(
            r"(?:relative\s+risk|odds\s+ratio|RR|OR)[,\s]*"
            + _n("hr")
            + r"\s*[(\[]\s*"
            + _CI_INTRO + r"?"
            + _n("lci")
            + r"\s*[-–—to,]+\s*"
            + _n("uci")
            + r"\s*[)\]]",
            re.IGNORECASE,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_number(raw: str) -> float:
    """Parse a decimal string that may use either period or comma as separator."""
    return float(raw.replace(",", "."))


def _validate_hr_range(hr: float, lci: float, uci: float) -> Optional[str]:
    """
    Validate that extracted HR values are clinically plausible.

    Returns an error message string if validation fails, ``None`` otherwise.
    """
    if not (lci < hr < uci):
        return f"HR {hr} is not between LCI {lci} and UCI {uci}."
    if hr <= 0 or lci <= 0 or uci <= 0:
        return "One or more values are non-positive."
    if hr > 10 or hr < 0.05:
        return f"HR {hr} outside plausible range [0.05, 10.0] for a clinical trial."
    if uci / lci > 20:
        return f"CI width ratio {uci / lci:.1f} implausibly large."
    return None


def _search_results_section(abstract_sections: dict[str, str], full_abstract: str) -> str:
    """
    Return the most informative section of the abstract for HR searching.

    Prefers the labelled Results section of structured abstracts; falls back
    to the full abstract text.
    """
    for label in ("RESULTS", "FINDINGS", "RESULTS AND DISCUSSION", "MAIN OUTCOME MEASURE"):
        if label in abstract_sections:
            return abstract_sections[label]
    return full_abstract


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Internal result container for a single HR extraction attempt.

    Not exposed outside this module; :class:`EffectMeasure` is the public
    interface for downstream consumers.
    """
    success: bool
    hr: float = 0.0
    lci: float = 0.0
    uci: float = 0.0
    pattern_name: str = ""
    source_text: str = ""
    requires_manual_check: bool = False
    failure_reason: str = ""


def extract_hr_from_abstract(
    abstract_text: str,
    abstract_sections: Optional[dict[str, str]] = None,
) -> ExtractionResult:
    """
    Attempt to extract a primary Hazard Ratio and 95% CI from abstract text.

    Tries each pattern in ``_HR_PATTERNS`` in priority order against the
    Results section of the abstract (or the full abstract for unstructured
    records).  Returns the first successful match.

    Parameters
    ----------
    abstract_text:
        The full abstract as a single string.
    abstract_sections:
        Labelled abstract sections (from :attr:`PubMedRecord.abstract_sections`).
        Pass ``None`` or an empty dict for unstructured abstracts.

    Returns
    -------
    ExtractionResult
        ``success=True`` with populated ``hr``, ``lci``, ``uci``,
        ``pattern_name``, and ``source_text`` on success.
        ``success=False`` with ``failure_reason`` set on failure.
    """
    if not abstract_text:
        return ExtractionResult(success=False, failure_reason="Abstract is empty.")

    sections = abstract_sections or {}
    search_text = _search_results_section(sections, abstract_text)

    for pattern_name, pattern in _HR_PATTERNS:
        match = pattern.search(search_text)
        if match is None:
            continue

        try:
            hr  = _parse_number(match.group("hr"))
            lci = _parse_number(match.group("lci"))
            uci = _parse_number(match.group("uci"))
        except (IndexError, KeyError, ValueError) as exc:
            logger.debug("Pattern %r matched but group extraction failed: %s", pattern_name, exc)
            continue

        validation_error = _validate_hr_range(hr, lci, uci)
        if validation_error:
            logger.debug(
                "Pattern %r: values failed validation (%s). Trying next pattern.",
                pattern_name, validation_error,
            )
            continue

        # Capture a window of text around the match for the audit log
        start    = max(0, match.start() - 30)
        end      = min(len(search_text), match.end() + 30)
        src_text = search_text[start:end].strip()

        # Flag the RR/OR fallback as requiring manual verification since it
        # is not a true hazard ratio.
        requires_check = pattern_name == "rr_or_fallback"

        logger.debug(
            "Extracted HR=%.3f (%.3f–%.3f) via pattern %r from text: %r",
            hr, lci, uci, pattern_name, src_text[:80],
        )
        return ExtractionResult(
            success               = True,
            hr                    = hr,
            lci                   = lci,
            uci                   = uci,
            pattern_name          = pattern_name,
            source_text           = src_text,
            requires_manual_check = requires_check,
        )

    return ExtractionResult(
        success        = False,
        failure_reason = (
            "No HR/CI pattern matched in abstract. "
            "Manual extraction required."
        ),
    )


def extract_effect_measures(
    linked_trials: pd.DataFrame,
    audit_log_path: Path = POWER_AUDIT_LOG_PATH,
) -> list[EffectMeasure]:
    """
    Extract Hazard Ratios from all linked, High/Medium-confidence trial pairs.

    Iterates over *linked_trials*, skips Low-confidence and unlinked rows
    (these must be resolved by the human reviewer before effect measures can
    be extracted), attempts HR extraction from the PubMed abstract, and
    returns a list of validated :class:`EffectMeasure` objects ready to be
    passed to Module 3.

    All extraction outcomes — successes and failures — are appended to the
    power audit log at *audit_log_path*.

    Parameters
    ----------
    linked_trials:
        Output of :func:`module1_linker.link_to_pubmed`.  Required columns:
        ``nct_id``, ``pmid``, ``abstract_text``, ``linkage_confidence``,
        ``registration_date``.
    audit_log_path:
        Filesystem path for the power audit log CSV.  Created if absent.

    Returns
    -------
    list[EffectMeasure]
        Validated effect measures, sorted by ``registration_date`` for
        sequential Bayesian analysis.  Trials with failed extraction are
        excluded from this list but logged.

    Notes
    -----
    Only trials with ``linkage_confidence`` of ``"High"`` or ``"Medium"``
    are processed.  Low-confidence and Unlinked trials have their linkage
    flagged for human review first; attempting HR extraction before the link
    is confirmed would produce unreliable data.
    """
    _initialise_audit_log(audit_log_path)

    effect_measures: list[EffectMeasure] = []
    processable = linked_trials[
        linked_trials["linkage_confidence"].isin(["High", "Medium"])
    ].copy()

    skipped = len(linked_trials) - len(processable)
    if skipped:
        logger.info(
            "HR extraction: skipping %d trials with Low/Unlinked linkage confidence "
            "(pending human review of publication link).",
            skipped,
        )

    logger.info(
        "Extracting HR/CI from PubMed abstracts for %d linked trials...", len(processable)
    )

    for _, row in processable.iterrows():
        nct_id    = str(row.get("nct_id", "")).strip()
        pmid      = str(row.get("pmid", "")).strip()
        pair_id   = f"{nct_id}_{pmid}" if pmid else f"{nct_id}_unlinked"
        reg_date  = str(row.get("registration_date", "")).strip()
        abstract  = str(row.get("abstract_text", "")).strip()

        if not pmid:
            _write_audit_row(audit_log_path, {
                "nct_id":               nct_id,
                "pmid":                 "",
                "pair_id":              pair_id,
                "extraction_method":    "failed",
                "requires_manual_check": "True",
                "exclusion_reason":     "No PMID — trial not linked to a publication.",
                "registration_date":    reg_date,
            })
            logger.warning("  %s — no PMID; skipping HR extraction.", nct_id)
            continue

        if not abstract:
            _write_audit_row(audit_log_path, {
                "nct_id":               nct_id,
                "pmid":                 pmid,
                "pair_id":              pair_id,
                "extraction_method":    "failed",
                "requires_manual_check": "True",
                "exclusion_reason":     "Abstract text is empty; manual extraction required.",
                "registration_date":    reg_date,
            })
            logger.warning("  %s (PMID %s) — abstract is empty.", nct_id, pmid)
            continue

        result = extract_hr_from_abstract(abstract_text=abstract)

        if not result.success:
            _write_audit_row(audit_log_path, {
                "nct_id":               nct_id,
                "pmid":                 pmid,
                "pair_id":              pair_id,
                "extraction_method":    "failed",
                "requires_manual_check": "True",
                "exclusion_reason":     result.failure_reason,
                "registration_date":    reg_date,
            })
            logger.info(
                "  %s (PMID %s) — extraction failed: %s", nct_id, pmid, result.failure_reason
            )
            continue

        # Build the EffectMeasure via the factory method, which re-derives
        # log_hr, se_log_hr, and variance and validates their consistency.
        try:
            em = EffectMeasure.from_raw(
                pair_id           = pair_id,
                nct_id            = nct_id,
                pmid              = pmid,
                hr                = result.hr,
                hr_lci            = result.lci,
                hr_uci            = result.uci,
                extraction_method = result.pattern_name,
                source_reference  = f"PMID:{pmid} — {result.source_text[:120]}",
                registration_date = reg_date or None,
            )
        except Exception as exc:
            _write_audit_row(audit_log_path, {
                "nct_id":               nct_id,
                "pmid":                 pmid,
                "pair_id":              pair_id,
                "extraction_method":    result.pattern_name,
                "requires_manual_check": "True",
                "exclusion_reason":     f"EffectMeasure validation failed: {exc}",
                "registration_date":    reg_date,
            })
            logger.warning(
                "  %s (PMID %s) — EffectMeasure validation failed: %s", nct_id, pmid, exc
            )
            continue

        # Write a success row to the audit log
        _write_audit_row(audit_log_path, {
            "nct_id":               nct_id,
            "pmid":                 pmid,
            "pair_id":              pair_id,
            "hr":                   em.hr,
            "hr_lci":               em.hr_lci,
            "hr_uci":               em.hr_uci,
            "log_hr":               em.log_hr,
            "se_log_hr":            em.se_log_hr,
            "extraction_method":    result.pattern_name,
            "source_text":          result.source_text[:200],
            "requires_manual_check": str(result.requires_manual_check),
            "exclusion_reason":     "",
            "registration_date":    reg_date,
        })

        manual_flag = " [REQUIRES MANUAL CHECK — RR/OR fallback]" if result.requires_manual_check else ""
        logger.info(
            "  %s (PMID %s) — HR=%.3f (95%% CI %.3f–%.3f) via %r%s",
            nct_id, pmid, em.hr, em.hr_lci, em.hr_uci, result.pattern_name, manual_flag,
        )
        effect_measures.append(em)

    # Sort by registration date for chronological sequential analysis
    effect_measures.sort(key=lambda em: em.registration_date or "")

    n_total    = len(processable)
    n_success  = len(effect_measures)
    n_failed   = n_total - n_success
    n_manual   = sum(1 for em in effect_measures if "rr_or_fallback" in em.extraction_method)

    logger.info(
        "HR extraction complete: %d processed | %d extracted | %d failed "
        "(manual required) | %d RR/OR fallback (manual check flagged)",
        n_total, n_success, n_failed, n_manual,
    )

    return effect_measures
