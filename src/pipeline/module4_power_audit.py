"""
Module 4 — Power audit (Section 3.5).

For each registered trial, back-calculates the implied effect size assumption
from the enrolled sample size. Compares this to the Bayesian posterior mean
available at the time of trial registration (using only evidence published
before that date). The gap is the trial's optimism bias.

All inputs and computations are logged. Trials with missing inputs are
explicitly excluded with a reason code.
"""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

from src.models.schemas import PowerAuditEntry
from src.pipeline.config import POWER_AUDIT_LOG_PATH

logger = logging.getLogger(__name__)

_COLUMNS = [
    "nct_id",
    "enrollment",
    "assumed_hr",
    "assumed_event_rate",
    "alpha",
    "power",
    "posterior_hr_at_registration",
    "optimism_bias",
    "excluded_reason",
    "inputs_source",
    "audit_timestamp",
]


# ---------------------------------------------------------------------------
# Effect size back-calculation
# ---------------------------------------------------------------------------

def _back_calculate_hr(
    enrollment: int,
    event_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Back-calculate the HR assumption implied by the registered enrollment figure,
    assuming a log-rank test with balanced allocation.

    Uses the Freedman (1982) approximation:
      n_events = (z_alpha/2 + z_beta)^2 / (log HR)^2 * 4
    Solving for HR given n_events = enrollment * event_rate.

    Returns the implied HR (< 1 for a beneficial effect).
    """
    n_events = enrollment * event_rate
    if n_events <= 0:
        raise ValueError(f"Computed n_events={n_events:.1f} is non-positive.")

    z_alpha = abs(stats_norm_ppf(1 - alpha / 2))
    z_beta = abs(stats_norm_ppf(1 - power))

    # (z_alpha + z_beta)^2 = n_events * (log HR)^2 / 4
    # |log HR| = (z_alpha + z_beta) * 2 / sqrt(n_events)
    log_hr_abs = (z_alpha + z_beta) * 2.0 / math.sqrt(n_events)
    implied_hr = math.exp(-log_hr_abs)   # beneficial direction (HR < 1)
    return round(implied_hr, 4)


def stats_norm_ppf(p: float) -> float:
    from scipy.stats import norm  # type: ignore
    return float(norm.ppf(p))


# ---------------------------------------------------------------------------
# Posterior HR at registration date
# ---------------------------------------------------------------------------

def get_posterior_hr_at_date(
    registration_date: str,
    sequential_results: list[dict],
) -> Optional[float]:
    """
    Return the pooled HR from the most recent sequential model fit whose
    last-added trial was registered *before* registration_date.

    sequential_results is the output of module3_bayesian.run_sequential_analysis().
    """
    if not registration_date or not sequential_results:
        return None

    try:
        reg_dt = datetime.fromisoformat(registration_date.replace("/", "-")[:10])
    except ValueError:
        return None

    best_hr: Optional[float] = None
    for result in sequential_results:
        # Each result's 'nct_id' was the last trial added
        # We need the registration date of that trial — stored in the sequential df
        # Caller must ensure nct_registration_dates is populated in result if needed
        result_date_str = result.get("registration_date", "")
        if not result_date_str:
            continue
        try:
            result_dt = datetime.fromisoformat(result_date_str.replace("/", "-")[:10])
        except ValueError:
            continue
        if result_dt < reg_dt:
            best_hr = result["mu_mean"]

    return best_hr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_power_audit(
    trials_df: pd.DataFrame,
    sequential_results: list[dict],
    assumed_event_rate: float = 0.15,  # default fallback for time-to-event oncology endpoints
    alpha: float = 0.05,
    power: float = 0.80,
    inputs_source: str = "registry",
) -> list[PowerAuditEntry]:
    """
    Run the power audit for every trial in trials_df.

    Parameters
    ----------
    trials_df : pd.DataFrame
        Must contain: nct_id, enrollment, registration_date.
    sequential_results : list[dict]
        Output of module3_bayesian.run_sequential_analysis().
    assumed_event_rate : float
        Default event rate; overridden per-trial if 'event_rate' column present.
    """
    entries: list[PowerAuditEntry] = []
    rows_for_log: list[dict] = []

    for _, row in trials_df.iterrows():
        nct_id: str = row["nct_id"]
        excluded_reason: Optional[str] = None
        assumed_hr: Optional[float] = None
        optimism_bias: Optional[float] = None
        posterior_hr: Optional[float] = None

        # Determine enrollment
        enrollment_raw = row.get("enrollment")
        try:
            enrollment = int(float(str(enrollment_raw)))
            if enrollment <= 0:
                raise ValueError("non-positive enrollment")
        except (ValueError, TypeError):
            excluded_reason = "missing_or_invalid_enrollment"
            enrollment = 0

        # Determine event rate (per-trial if available)
        event_rate = float(row.get("event_rate", assumed_event_rate))

        if not excluded_reason:
            try:
                assumed_hr = _back_calculate_hr(enrollment, event_rate, alpha, power)
            except Exception as exc:
                excluded_reason = f"back_calculation_failed: {exc}"

        # Posterior HR at registration
        if not excluded_reason:
            reg_date = str(row.get("registration_date", ""))
            posterior_hr = get_posterior_hr_at_date(reg_date, sequential_results)
            if posterior_hr is not None and assumed_hr is not None:
                optimism_bias = round(assumed_hr - posterior_hr, 4)

        entry = PowerAuditEntry(
            nct_id=nct_id,
            enrollment=max(enrollment, 1),
            assumed_hr=assumed_hr or 1.0,
            assumed_event_rate=event_rate,
            alpha=alpha,
            power=power,
            posterior_hr_at_registration=posterior_hr,
            optimism_bias=optimism_bias,
            excluded_reason=excluded_reason,
            inputs_source=inputs_source,
        )
        entries.append(entry)

        rows_for_log.append(
            {
                **entry.model_dump(),
                "audit_timestamp": datetime.utcnow().isoformat(),
            }
        )

        if excluded_reason:
            logger.warning("Power audit: %s excluded — %s", nct_id, excluded_reason)
        else:
            logger.debug(
                "Power audit: %s assumed HR=%.3f posterior HR=%s bias=%s",
                nct_id,
                assumed_hr,
                f"{posterior_hr:.3f}" if posterior_hr else "N/A",
                f"{optimism_bias:.3f}" if optimism_bias is not None else "N/A",
            )

    _write_power_audit_log(rows_for_log)
    included = sum(1 for e in entries if e.excluded_reason is None)
    logger.info(
        "Power audit complete: %d included, %d excluded",
        included,
        len(entries) - included,
    )
    return entries


def _write_power_audit_log(rows: list[dict]) -> None:
    POWER_AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not POWER_AUDIT_LOG_PATH.exists()
    with open(POWER_AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _COLUMNS})
