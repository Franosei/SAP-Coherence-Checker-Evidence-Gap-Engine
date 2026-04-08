"""
Evidence gap scorecard (Section 3.6).

Synthesises all four modules. Every scorecard cell links to its provenance:
which trials contributed, which AI classifications were used, which were
human-overridden, and the final human-confirmed status.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.schemas import EvidenceStrength, ScorecardEntry
from src.pipeline.config import DECISION_LOG_PATH, LINKAGE_LOG_PATH
from src.pipeline.validation import compute_ai_calibration as _compute_ai_calibration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence strength classification rules
# ---------------------------------------------------------------------------

def _classify_evidence_strength(
    pooled_hr: Optional[float],
    cri_lower: Optional[float],
    cri_upper: Optional[float],
    n_trials: int,
    human_confirmed_switch_rate_pct: float,
) -> EvidenceStrength:
    """
    Classify evidence strength based on pooled estimate, credible interval,
    trial count, and outcome switching prevalence.
    """
    if n_trials == 0 or pooled_hr is None:
        return EvidenceStrength.GAP

    # CrI entirely excludes null (HR=1)?
    excludes_null = cri_upper is not None and cri_lower is not None and cri_upper < 1.0

    if n_trials >= 5 and excludes_null and human_confirmed_switch_rate_pct < 20:
        return EvidenceStrength.STRONG
    if n_trials >= 3 and excludes_null:
        return EvidenceStrength.MODERATE
    if n_trials >= 2:
        return EvidenceStrength.WEAK
    return EvidenceStrength.GAP


# ---------------------------------------------------------------------------
# AI calibration metrics
# ---------------------------------------------------------------------------

def compute_ai_calibration(
    decision_log: pd.DataFrame,
    gold_standard: pd.DataFrame,
) -> dict:
    """
    Backward-compatible wrapper for the validation workflow helper.
    """
    return _compute_ai_calibration(decision_log, gold_standard)


# ---------------------------------------------------------------------------
# Endpoint clustering
# ---------------------------------------------------------------------------

# Ordered list of (cluster_name, [keyword_patterns]).
# A pair is assigned to the FIRST matching cluster; unmatched pairs fall into
# "other_endpoints".  Patterns are matched case-insensitively against the
# registered endpoint text.
_CLUSTER_RULES: list[tuple[str, list[str]]] = [
    ("mace_composite", [
        "composite", "mace", "major adverse cardiovascular",
        "cardiovascular death.*hospitali", "hospitali.*cardiovascular death",
        "death.*heart failure", "heart failure.*death",
        "worsening heart failure.*death", "death.*worsening heart failure",
    ]),
    ("all_cause_mortality", [
        "all.cause mort", "all.cause death", "all cause mort", "all cause death",
        "death from any cause", "death due to any cause",
    ]),
    ("cv_mortality", [
        "cardiovascular death", "cardiovascular mort", "cv death", "cv mort",
        "cardiac death", "cardiac mort",
    ]),
    ("hf_hospitalization", [
        "heart failure hospitali", "hf hospitali", "hospitali.*heart failure",
        "worsening heart failure", "acute decompensated", "urgent visit.*heart failure",
        "heart failure.*urgent visit",
    ]),
    ("exercise_capacity", [
        "6.minute walk", "six.minute walk", "6mwt", "peak vo2", "vo2 max",
        "exercise capacity", "exercise tolerance", "cardiopulmonary exercise",
    ]),
    ("nt_probnp_biomarkers", [
        "nt.probnp", "nt probnp", r"\bbnp\b", "natriuretic peptide",
        "troponin", "galectin", "st2",
    ]),
    ("quality_of_life", [
        "kccq", "quality of life", "qol", "mlhfq", "minnesota living",
        "patient.reported", "symptom", "kansas city",
    ]),
    ("lvef_remodeling", [
        "ejection fraction", "lvef", "left ventricular ejection",
        "end.diastolic volume", "end.systolic volume", "lv remodel",
        "reverse remodel",
    ]),
]


def cluster_endpoints(
    matched_df: pd.DataFrame,
    registered_col: str = "registered_endpoint",
    pair_id_col: str = "pair_id",
) -> dict[str, list[str]]:
    """
    Assign each trial pair to an endpoint cluster based on keyword matching
    against its registered endpoint text.

    Parameters
    ----------
    matched_df :
        DataFrame that must contain ``pair_id`` and ``registered_endpoint``
        columns.  Typically the output of ``run_endpoint_matching()``.
    registered_col :
        Column name containing the pre-specified registered endpoint text.
    pair_id_col :
        Column name containing the unique pair identifier.

    Returns
    -------
    dict[str, list[str]]
        Mapping of ``cluster_name`` to the list of ``pair_id`` values assigned
        to that cluster.  Every pair_id appears in exactly one cluster.
        Unmatched pairs land in ``"other_endpoints"``.

    Notes
    -----
    - Matching is case-insensitive and uses Python ``re.search``.
    - The first matching cluster wins (rules are ordered from most specific
      to most general).
    - Pairs with a blank or missing registered endpoint go to
      ``"missing_endpoint"`` so they are explicitly visible in the scorecard
      rather than silently ignored.
    """
    import re as _re

    clusters: dict[str, list[str]] = {name: [] for name, _ in _CLUSTER_RULES}
    clusters["other_endpoints"] = []
    clusters["missing_endpoint"] = []

    for _, row in matched_df.iterrows():
        pair_id = str(row.get(pair_id_col, "")).strip()
        text    = str(row.get(registered_col, "")).strip().lower()

        if not text or text in {"", "none", "nan"}:
            clusters["missing_endpoint"].append(pair_id)
            continue

        assigned = False
        for cluster_name, patterns in _CLUSTER_RULES:
            for pattern in patterns:
                if _re.search(pattern, text, _re.IGNORECASE):
                    clusters[cluster_name].append(pair_id)
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            clusters["other_endpoints"].append(pair_id)

    # Drop empty clusters to keep the scorecard tidy
    clusters = {k: v for k, v in clusters.items() if v}

    logger.info(
        "Endpoint clustering complete: %d clusters, %d pairs total.",
        len(clusters),
        sum(len(v) for v in clusters.values()),
    )
    for name, ids in sorted(clusters.items()):
        logger.info("  %-30s %d pair(s)", name, len(ids))

    return clusters


# ---------------------------------------------------------------------------
# Per-cluster scorecard construction
# ---------------------------------------------------------------------------

def build_scorecard(
    endpoint_clusters: dict[str, list[str]],   # cluster_name → [pair_ids]
    decision_log_path=DECISION_LOG_PATH,
    linkage_log_path=LINKAGE_LOG_PATH,
    bayesian_summaries: Optional[dict[str, dict]] = None,
    power_audit_summary: Optional[dict[str, float]] = None,
    gold_standard_path: Optional[str] = None,
    within_trial_variances: Optional[dict[str, float]] = None,
) -> list[ScorecardEntry]:
    """
    Build one ScorecardEntry per endpoint cluster.

    Parameters
    ----------
    endpoint_clusters : mapping of cluster name to list of pair_ids in that cluster
    bayesian_summaries : {cluster_name: summarise_posterior() output}
    power_audit_summary : {nct_id: optimism_bias}
    gold_standard_path : optional path to gold_standard.csv for AI calibration
    within_trial_variances : {pair_id: se_log_hr ** 2}
        Within-trial variances (σ²_i) derived from the effect measures.
        Used for the correct I² = τ² / (τ² + σ²_typical) formula.
        If None, I² is not computed.
    """
    dl = pd.read_csv(decision_log_path, dtype=str, keep_default_na=False)
    ll = pd.read_csv(linkage_log_path, dtype=str, keep_default_na=False)

    scorecard: list[ScorecardEntry] = []

    for cluster_name, pair_ids in endpoint_clusters.items():
        cluster_dl = dl[dl["pair_id"].isin(pair_ids)]

        trials_included = cluster_dl[cluster_dl["human_poolable"].str.lower() == "true"].shape[0]
        trials_excluded = cluster_dl[cluster_dl["human_poolable"].str.lower() == "false"].shape[0]

        # Unlinked: from linkage log
        nct_ids_in_cluster = cluster_dl["pair_id"].str.split("_").str[0].unique().tolist()
        trials_unlinked = ll[
            ll["nct_id"].isin(nct_ids_in_cluster)
            & ll["linkage_confidence"].isin(["Unlinked"])
        ].shape[0]

        # Human override rate
        reviewed = cluster_dl[cluster_dl["human_reviewed"].isin(["yes", "spot_check"])]
        overrides = reviewed[reviewed["human_decision"] == "override"]
        override_rate = (
            round(len(overrides) / max(len(reviewed), 1) * 100, 1) if not reviewed.empty else 0.0
        )

        # Mean SSI
        ssi_vals = pd.to_numeric(cluster_dl["ssi"], errors="coerce").dropna()
        mean_ssi = round(float(ssi_vals.mean()), 2) if not ssi_vals.empty else 0.0

        # Human-confirmed switch rate
        confirmed_switch = cluster_dl[
            cluster_dl["human_final_class"].isin(
                ["minor_modification", "moderate_switch", "major_switch"]
            )
        ]
        switch_rate = round(
            len(confirmed_switch) / max(len(cluster_dl), 1) * 100, 1
        )

        # AI-human agreement
        agreed = cluster_dl[
            cluster_dl["llm_switch_type"].notna()
            & (cluster_dl["llm_switch_type"] == cluster_dl["human_final_class"])
        ]
        ai_human_agreement = round(
            len(agreed) / max(cluster_dl["llm_switch_type"].notna().sum(), 1) * 100, 1
        )

        # Bayesian results
        bayes = (bayesian_summaries or {}).get(cluster_name, {})
        pooled_hr = bayes.get("pooled_hr")
        cri_lower = bayes.get("pooled_hr_cri_lower")
        cri_upper = bayes.get("pooled_hr_cri_upper")
        tau = bayes.get("tau_mean")

        # I² = τ² / (τ² + σ²_typical) × 100
        # σ²_typical is the median within-trial variance (se_log_hr²) for
        # trials in this cluster.  SSI is a UI routing metric — it must NOT
        # be used here.  If variances are unavailable the field is left None.
        i_squared: Optional[float] = None
        if tau is not None and within_trial_variances:
            cluster_vars = [
                within_trial_variances[pid]
                for pid in pair_ids
                if pid in within_trial_variances
            ]
            if cluster_vars:
                sigma2_typical = float(np.median(cluster_vars))
                if sigma2_typical > 0:
                    i_squared = round(
                        tau ** 2 / (tau ** 2 + sigma2_typical) * 100, 1
                    )

        # Mean optimism bias from power audit
        if power_audit_summary:
            nct_biases = [
                power_audit_summary[n]
                for n in nct_ids_in_cluster
                if n in power_audit_summary and power_audit_summary[n] is not None
            ]
            mean_bias = round(float(np.mean(nct_biases)), 3) if nct_biases else None
        else:
            mean_bias = None

        evidence_strength = _classify_evidence_strength(
            pooled_hr, cri_lower, cri_upper, trials_included, switch_rate
        )

        entry = ScorecardEntry(
            endpoint_cluster=cluster_name,
            trials_included=trials_included,
            trials_excluded=trials_excluded,
            trials_unlinked=trials_unlinked,
            human_override_rate_pct=override_rate,
            mean_ssi=mean_ssi,
            human_confirmed_switch_rate_pct=switch_rate,
            ai_human_agreement_rate_pct=ai_human_agreement,
            pooled_hr=pooled_hr,
            pooled_hr_cri_lower=cri_lower,
            pooled_hr_cri_upper=cri_upper,
            between_trial_tau=tau,
            i_squared_pct=i_squared,
            mean_optimism_bias=mean_bias,
            evidence_strength=evidence_strength,
        )
        scorecard.append(entry)
        logger.info("Scorecard: %s — %s", cluster_name, evidence_strength.value)

    return scorecard
