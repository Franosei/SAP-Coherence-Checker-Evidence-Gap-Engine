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
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.schemas import EffectMeasure
from src.pipeline.config import (
    EFFECT_MEASURE_LOG_PATH,
    LLM_BASE_URL,
    LLM_MODEL_PRIMARY,
    PIPELINE_VERSION,
    llm_sampling_kwargs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction audit log
# ---------------------------------------------------------------------------

_AUDIT_COLUMNS: list[str] = [
    "nct_id",
    "pmid",
    "pair_id",
    "measure_type",
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

# extraction_method values that count as "already attempted" for resume — a
# bare "failed" is NOT here so a regex-only failure is retried by the LLM.
_DONE_METHODS = {
    "hr_paren_ci",
    "hr_semicolon_ci",
    "hr_narrative_to",
    "hr_bracket_ci",
    "hr_comma_ci",
    "rr_or_fallback",
    "llm_hr",
    "llm_or",
    "llm_rr",
    "llm_none",
    "manual",
    "no_pmid",
    "empty_abstract",
}


def _stamp(row: dict) -> dict:
    row.setdefault("pipeline_version", PIPELINE_VERSION)
    row.setdefault("extracted_at", datetime.now(UTC).isoformat())
    return row


def _write_audit_log(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole audit log — one row per pair_id, latest attempt wins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_pair: dict[str, dict] = {}
    for row in rows:
        pid = str(row.get("pair_id", "")).strip()
        if pid:
            by_pair[pid] = row
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_AUDIT_COLUMNS)
        writer.writeheader()
        for row in by_pair.values():
            writer.writerow({col: row.get(col, "") for col in _AUDIT_COLUMNS})


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

_NUM = r"(?P<{name}>\d+[.,]\d+)"  # Named decimal number capture template


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
            + _CI_INTRO
            + r"?"
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
            + _CI_INTRO
            + r"?"
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
            + _CI_INTRO
            + r"?"
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
    Internal result container for a single effect-measure extraction attempt.

    Not exposed outside this module; :class:`EffectMeasure` is the public
    interface for downstream consumers.
    """

    success: bool
    hr: float = 0.0
    lci: float = 0.0
    uci: float = 0.0
    pattern_name: str = ""
    measure_type: str = "HR"  # HR | OR | RR | risk_difference | none
    source_text: str = ""
    requires_manual_check: bool = False
    failure_reason: str = ""


def _combined_text(abstract: str, results_text: str, conclusion_text: str) -> str:
    """Abstract + labelled Results + Conclusion, de-duplicated, for the LLM/regex."""
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in (abstract, results_text, conclusion_text):
        chunk = (chunk or "").strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            parts.append(chunk)
    return "\n\n".join(parts)


_LLM_EXTRACT_SYSTEM = (
    "You extract the PRIMARY-endpoint effect estimate from a randomized clinical "
    "trial report, for a meta-analysis. You are given the endpoint the publication "
    "reports as primary and the abstract + Results text. Find the between-group "
    "effect estimate for THAT endpoint.\n\n"
    'Return JSON: {"primary_endpoint","measure_type","point_estimate","ci_lower",'
    '"ci_upper","ci_pct","time_to_event","quote","note"}.\n'
    "measure_type is one of: HR (time-to-event endpoints only — PFS, OS, DFS, EFS, "
    "iDFS, RFS, DRFS, time to progression/recurrence), OR, RR, risk_difference, or "
    "none. Use none when the report gives no ratio or difference with numbers for "
    "the primary endpoint (only rates, only a p-value, or the estimate is in a "
    "figure/table not shown). For pCR / ORR / response-rate primary endpoints use "
    "OR or RR exactly as the paper reports; if only the two rates are given, use "
    "risk_difference with point_estimate = interventionRate - controlRate as a "
    "proportion and null CI. point_estimate/ci_lower/ci_upper are numbers or null. "
    "Never invent numbers. quote is the verbatim sentence containing the estimate."
)


def _llm_extract_effect_measure(published_endpoint: str, full_text: str) -> ExtractionResult:
    """LLM fallback: pull the primary-endpoint effect estimate from the full text."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.lower().startswith("your_"):
        return ExtractionResult(success=False, failure_reason="No LLM available; manual extraction required.")
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
        user = (
            f"PUBLICATION'S PRIMARY ENDPOINT: {published_endpoint or '(not stated — infer it)'}\n\n"
            f"ABSTRACT + RESULTS:\n{full_text[:12000]}"
        )
        response = client.chat.completions.create(
            model=LLM_MODEL_PRIMARY,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            **llm_sampling_kwargs(1200),
        )
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("Effect-measure LLM failed: %s", exc)
        return ExtractionResult(success=False, failure_reason=f"LLM error: {exc}")

    measure = str(data.get("measure_type", "none")).strip().lower()
    note = str(data.get("note", "")).strip()
    quote = str(data.get("quote", "")).strip()

    if measure not in {"hr", "or", "rr"}:
        reason = note or f"LLM: primary-endpoint effect is '{measure}', not a poolable ratio."
        return ExtractionResult(
            success=False,
            pattern_name="llm_none",
            measure_type=measure or "none",
            source_text=quote[:200],
            failure_reason=reason,
        )

    try:
        hr = float(data["point_estimate"])
        lci = float(data["ci_lower"])
        uci = float(data["ci_upper"])
    except (KeyError, TypeError, ValueError):
        return ExtractionResult(
            success=False,
            pattern_name="llm_none",
            measure_type=measure,
            source_text=quote[:200],
            failure_reason=note or "LLM found a ratio but no usable point estimate + 95% CI.",
        )

    if lci > uci:
        lci, uci = uci, lci
    validation_error = _validate_hr_range(hr, lci, uci)
    if validation_error:
        return ExtractionResult(
            success=False,
            pattern_name="llm_none",
            measure_type=measure,
            source_text=quote[:200],
            failure_reason=f"LLM values failed validation: {validation_error}",
        )

    return ExtractionResult(
        success=True,
        hr=hr,
        lci=lci,
        uci=uci,
        pattern_name=f"llm_{measure}",
        measure_type=measure.upper(),
        source_text=(quote or note)[:200],
        requires_manual_check=measure in {"or", "rr"},
    )


def extract_hr_from_abstract(
    abstract_text: str,
    abstract_sections: Optional[dict[str, str]] = None,
    results_text: str = "",
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
    if not abstract_text and not results_text:
        return ExtractionResult(success=False, failure_reason="Abstract is empty.")

    sections = abstract_sections or {}
    # Search the labelled Results section (from the structured PubMed abstract)
    # first, then the full abstract.
    search_text = "\n\n".join(
        t for t in (results_text, _search_results_section(sections, abstract_text)) if t
    ) or abstract_text

    for pattern_name, pattern in _HR_PATTERNS:
        match = pattern.search(search_text)
        if match is None:
            continue

        try:
            hr = _parse_number(match.group("hr"))
            lci = _parse_number(match.group("lci"))
            uci = _parse_number(match.group("uci"))
        except (IndexError, KeyError, ValueError) as exc:
            logger.debug("Pattern %r matched but group extraction failed: %s", pattern_name, exc)
            continue

        validation_error = _validate_hr_range(hr, lci, uci)
        if validation_error:
            logger.debug(
                "Pattern %r: values failed validation (%s). Trying next pattern.",
                pattern_name,
                validation_error,
            )
            continue

        # Capture a window of text around the match for the audit log
        start = max(0, match.start() - 30)
        end = min(len(search_text), match.end() + 30)
        src_text = search_text[start:end].strip()

        # Flag the RR/OR fallback as requiring manual verification since it
        # is not a true hazard ratio.
        requires_check = pattern_name == "rr_or_fallback"

        logger.debug(
            "Extracted HR=%.3f (%.3f–%.3f) via pattern %r from text: %r",
            hr,
            lci,
            uci,
            pattern_name,
            src_text[:80],
        )
        return ExtractionResult(
            success=True,
            hr=hr,
            lci=lci,
            uci=uci,
            pattern_name=pattern_name,
            measure_type="OR" if pattern_name == "rr_or_fallback" else "HR",
            source_text=src_text,
            requires_manual_check=requires_check,
        )

    return ExtractionResult(
        success=False,
        failure_reason=("No HR/CI pattern matched in abstract. Manual extraction required."),
    )


def _effect_from_row(r: dict) -> Optional[EffectMeasure]:
    try:
        hr, lci, uci = float(r["hr"]), float(r["hr_lci"]), float(r["hr_uci"])
    except (KeyError, ValueError, TypeError):
        return None
    if not (hr > 0 and lci > 0 and uci > 0):
        return None
    try:
        return EffectMeasure.from_raw(
            pair_id=str(r.get("pair_id", "")).strip(),
            nct_id=str(r.get("nct_id", "")).strip(),
            pmid=str(r.get("pmid", "")).strip(),
            hr=hr,
            hr_lci=lci,
            hr_uci=uci,
            extraction_method=str(r.get("extraction_method", "")).strip() or "manual",
            source_reference=f"PMID:{r.get('pmid', '')} — {str(r.get('source_text', ''))[:120]}",
            registration_date=str(r.get("registration_date", "")).strip() or None,
        )
    except Exception:
        return None


def _load_audit_state(path: Path) -> tuple[dict[str, dict], set[str]]:
    """Resume support. Returns ({pair_id: latest audit row}, {pair_ids already attempted})."""
    if not path.exists():
        return {}, set()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    except Exception:
        return {}, set()
    if "extraction_method" not in df.columns or "pair_id" not in df.columns:
        return {}, set()
    by_pair: dict[str, dict] = {}
    done: set[str] = set()
    for _, r in df.iterrows():
        pair_id = str(r.get("pair_id", "")).strip()
        if not pair_id:
            continue
        by_pair[pair_id] = r.to_dict()  # last row for a pair wins
    for pair_id, r in by_pair.items():
        if str(r.get("extraction_method", "")).strip() in _DONE_METHODS:
            done.add(pair_id)
    return by_pair, done


def extract_effect_measures(
    linked_trials: pd.DataFrame,
    audit_log_path: Path = EFFECT_MEASURE_LOG_PATH,
) -> list[EffectMeasure]:
    """Extract the primary-endpoint effect estimate for each linked SELECTED trial.

    Regex cascade first (precise for the explicit ``HR 0.65 (95% CI ...)`` form);
    on failure an LLM reads the full abstract + Results + Conclusion and returns
    the primary-endpoint HR / OR / RR, or reports that no poolable ratio exists
    (e.g. a pCR trial reporting only rates).

    The audit log (``data/logs/effect_measure_log.csv``) holds exactly one row
    per trial-publication pair and is **rewritten** each run — never appended,
    so re-runs cannot duplicate rows. A re-run resumes: pairs whose recorded
    ``extraction_method`` is in ``_DONE_METHODS`` are not re-attempted (LLM
    calls are not repaid); bare ``failed`` rows ARE retried.
    """
    prior_rows, done_pairs = _load_audit_state(audit_log_path)
    audit_rows: dict[str, dict] = dict(prior_rows)  # pair_id -> row; overwritten below

    gate = ~linked_trials["linkage_confidence"].isin(["Unlinked"])
    if "primary_result_status" in linked_trials.columns:
        gate &= linked_trials["primary_result_status"].eq("SELECTED")
    processable = linked_trials[gate].copy()

    skipped = len(linked_trials) - len(processable)
    if skipped:
        logger.info(
            "HR extraction: skipping %d trials without a SELECTED primary-results paper.",
            skipped,
        )
    if done_pairs:
        logger.info("HR extraction: %d pair(s) already recorded — resuming.", len(done_pairs))

    n_regex = n_llm = n_none = n_fail = 0

    def _record(pair_id: str, row: dict) -> None:
        audit_rows[pair_id] = _stamp(row)

    for _, row in processable.iterrows():
        nct_id = str(row.get("nct_id", "")).strip()
        pmid = str(row.get("pmid", "")).strip()
        pair_id = f"{nct_id}_{pmid}" if pmid else f"{nct_id}_unlinked"
        if pair_id in done_pairs:
            continue
        reg_date = str(row.get("registration_date", "")).strip()
        abstract = str(row.get("abstract_text", "")).strip()
        results_text = str(row.get("published_results", "")).strip()
        conclusion_text = str(row.get("published_conclusion", "")).strip()
        published_endpoint = str(row.get("published_endpoint", "")).strip()
        full_text = _combined_text(abstract, results_text, conclusion_text)

        base = {"nct_id": nct_id, "pmid": pmid, "pair_id": pair_id, "registration_date": reg_date}

        if not pmid:
            _record(pair_id, {**base, "extraction_method": "no_pmid", "requires_manual_check": "True",
                              "exclusion_reason": "No PMID — trial not linked to a publication."})
            n_fail += 1
            continue
        if not full_text:
            _record(pair_id, {**base, "extraction_method": "empty_abstract",
                              "requires_manual_check": "True",
                              "exclusion_reason": "No abstract / Results text available."})
            logger.warning("  %s (PMID %s) — no text to extract from.", nct_id, pmid)
            n_fail += 1
            continue

        result = extract_hr_from_abstract(abstract_text=abstract, results_text=results_text)
        via_llm = not result.success
        if via_llm:
            result = _llm_extract_effect_measure(published_endpoint, full_text)

        if not result.success:
            method = result.pattern_name or "failed"
            _record(pair_id, {**base, "measure_type": result.measure_type,
                              "extraction_method": method, "requires_manual_check": "True",
                              "exclusion_reason": result.failure_reason})
            if method == "llm_none":
                n_none += 1
                logger.info("  %s (PMID %s) — no poolable ratio: %s", nct_id, pmid, result.failure_reason)
            else:
                n_fail += 1
                logger.info("  %s (PMID %s) — extraction failed: %s", nct_id, pmid, result.failure_reason)
            continue

        try:
            em = EffectMeasure.from_raw(
                pair_id=pair_id,
                nct_id=nct_id,
                pmid=pmid,
                hr=result.hr,
                hr_lci=result.lci,
                hr_uci=result.uci,
                extraction_method=result.pattern_name,
                source_reference=f"PMID:{pmid} [{result.measure_type}] — {result.source_text[:120]}",
                registration_date=reg_date or None,
            )
        except Exception as exc:
            _record(pair_id, {**base, "measure_type": result.measure_type,
                              "extraction_method": result.pattern_name,
                              "requires_manual_check": "True",
                              "exclusion_reason": f"EffectMeasure validation failed: {exc}"})
            logger.warning("  %s (PMID %s) — EffectMeasure validation failed: %s", nct_id, pmid, exc)
            n_fail += 1
            continue

        _record(pair_id, {**base, "measure_type": result.measure_type, "hr": em.hr,
                          "hr_lci": em.hr_lci, "hr_uci": em.hr_uci, "log_hr": em.log_hr,
                          "se_log_hr": em.se_log_hr, "extraction_method": result.pattern_name,
                          "source_text": result.source_text[:200],
                          "requires_manual_check": str(result.requires_manual_check),
                          "exclusion_reason": ""})
        flag = " [manual check — not a true HR]" if result.requires_manual_check else ""
        logger.info(
            "  %s (PMID %s) — %s=%.3f (95%% CI %.3f–%.3f) via %s%s",
            nct_id, pmid, result.measure_type, em.hr, em.hr_lci, em.hr_uci, result.pattern_name, flag,
        )
        n_llm += int(via_llm)
        n_regex += int(not via_llm)

    _write_audit_log(audit_log_path, list(audit_rows.values()))
    effect_measures = [em for em in (_effect_from_row(r) for r in audit_rows.values()) if em]
    effect_measures.sort(key=lambda em: em.registration_date or "")

    logger.info(
        "HR extraction complete: %d poolable effect measure(s) from %d audited pair(s) | "
        "this run: %d regex, %d LLM, %d no-ratio (pCR/ORR etc.), %d failed",
        len(effect_measures), len(audit_rows), n_regex, n_llm, n_none, n_fail,
    )
    return effect_measures
