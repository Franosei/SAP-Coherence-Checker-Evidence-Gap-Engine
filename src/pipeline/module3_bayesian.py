"""
Module 3 — Bayesian evidence accumulation (Section 3.4).

Implements a random-effects Bayesian meta-analysis in PyMC.
Operates EXCLUSIVELY on trial pairs where human_poolable=TRUE in the
decision log — the provenance of the input dataset is fully traceable.

Sequential analysis: the model is re-fitted one trial at a time in
chronological order, producing a posterior after each addition.
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.models.schemas import EffectMeasure
from src.pipeline.config import (
    BAYES_CHAINS,
    BAYES_DRAWS,
    BAYES_PRIOR_MU_MEAN,
    BAYES_PRIOR_MU_SD,
    BAYES_PRIOR_TAU_SD,
    BAYES_TARGET_ACCEPT,
    BAYES_TRACE_DIR,
    BAYES_WARMUP,
    DECISION_LOG_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def load_poolable_effects(
    effect_measures: list[EffectMeasure],
    decision_log_path: Path = DECISION_LOG_PATH,
) -> pd.DataFrame:
    """
    Filter effect_measures to only those where human_poolable=True in the
    decision log. Returns a DataFrame sorted by registration_date.
    """
    import pandas as _pd

    dl = _pd.read_csv(decision_log_path, dtype=str, keep_default_na=False)
    poolable_ids = set(
        dl.loc[dl["human_poolable"].str.lower() == "true", "pair_id"].tolist()
    )

    rows = []
    for em in effect_measures:
        if em.pair_id in poolable_ids:
            rows.append(
                {
                    "pair_id": em.pair_id,
                    "nct_id": em.nct_id,
                    "pmid": em.pmid,
                    "log_hr": em.log_hr,
                    "se_log_hr": em.se_log_hr,
                    "variance": em.variance,
                    "registration_date": em.registration_date or "",
                    "extraction_method": em.extraction_method,
                }
            )

    if not rows:
        logger.warning("No poolable effect measures found after decision log filter.")
        return _pd.DataFrame()

    df = _pd.DataFrame(rows)
    df = df.sort_values("registration_date").reset_index(drop=True)
    logger.info("%d poolable trial pairs loaded for Bayesian model.", len(df))
    return df


# ---------------------------------------------------------------------------
# Core Bayesian model
# ---------------------------------------------------------------------------

def fit_random_effects_model(
    log_hrs: np.ndarray,
    se_log_hrs: np.ndarray,
    label: str = "full_dataset",
) -> az.InferenceData:
    """
    Fit a Bayesian random-effects meta-analysis model (Section 3.4.3).

    Parameters
    ----------
    log_hrs : array of log(HR) values, one per trial
    se_log_hrs : array of SE(log HR) values
    label : identifier used for trace file naming

    Returns
    -------
    ArviZ InferenceData object
    """
    n_trials = len(log_hrs)
    logger.info("Fitting Bayesian model on %d trials [%s]...", n_trials, label)

    with pm.Model():
        # Priors (Section 3.4.3)
        mu = pm.Normal("mu", mu=BAYES_PRIOR_MU_MEAN, sigma=BAYES_PRIOR_MU_SD)
        tau = pm.HalfNormal("tau", sigma=BAYES_PRIOR_TAU_SD)

        # Trial-level effects
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_trials)

        # Likelihood: observed log(HR) ~ Normal(theta_i, sigma_i^2)
        pm.Normal(
            "y",
            mu=theta,
            sigma=se_log_hrs,
            observed=log_hrs,
        )

        idata = pm.sample(
            draws=BAYES_DRAWS,
            tune=BAYES_WARMUP,
            chains=BAYES_CHAINS,
            target_accept=BAYES_TARGET_ACCEPT,
            progressbar=True,
            return_inferencedata=True,
        )

    _save_trace(idata, label)
    return idata


def _save_trace(idata: az.InferenceData, label: str) -> None:
    BAYES_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = BAYES_TRACE_DIR / f"trace_{label}.nc"
    idata.to_netcdf(str(path))
    logger.info("Bayesian trace saved: %s", path)


# ---------------------------------------------------------------------------
# Sequential analysis
# ---------------------------------------------------------------------------

def run_sequential_analysis(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Fit the Bayesian model incrementally, adding one trial at a time in
    chronological order (Section 3.4.3 — sequential analysis).

    Returns a list of dicts, each with:
    {
        'n_trials': int,
        'nct_id': str,
        'mu_mean': float,
        'mu_hdi_lower': float,
        'mu_hdi_upper': float,
        'tau_mean': float,
        'idata': InferenceData,
    }
    """
    results = []

    for i in range(1, len(df) + 1):
        subset = df.iloc[:i]
        log_hrs = subset["log_hr"].values.astype(float)
        se_log_hrs = subset["se_log_hr"].values.astype(float)
        label = f"sequential_n{i}_{subset.iloc[-1]['nct_id']}"

        idata = fit_random_effects_model(log_hrs, se_log_hrs, label=label)
        posterior_mu = idata.posterior["mu"].values.flatten()

        hdi = az.hdi(idata, var_names=["mu"], hdi_prob=0.95)["mu"].values
        tau_mean = float(idata.posterior["tau"].values.flatten().mean())

        results.append(
            {
                "n_trials":          i,
                "nct_id":            subset.iloc[-1]["nct_id"],
                # registration_date of the last-added trial — required by
                # module4_power_audit.get_posterior_hr_at_date() so that the
                # optimism bias can be computed for each trial.
                "registration_date": str(subset.iloc[-1].get("registration_date", "")),
                "mu_mean":           float(np.exp(posterior_mu.mean())),  # HR scale
                "mu_hdi_lower":      float(np.exp(hdi[0])),
                "mu_hdi_upper":      float(np.exp(hdi[1])),
                "tau_mean":          tau_mean,
                "idata":             idata,
            }
        )

        logger.info(
            "Sequential [n=%d]: pooled HR=%.3f (95%% CrI %.3f–%.3f), tau=%.3f",
            i,
            results[-1]["mu_mean"],
            results[-1]["mu_hdi_lower"],
            results[-1]["mu_hdi_upper"],
            tau_mean,
        )

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarise_posterior(idata: az.InferenceData) -> dict:
    """
    Extract the key summary statistics for reporting in the scorecard.
    """
    posterior_mu = idata.posterior["mu"].values.flatten()
    posterior_tau = idata.posterior["tau"].values.flatten()

    hdi_mu = az.hdi(idata, var_names=["mu"], hdi_prob=0.95)["mu"].values
    r_hat = az.rhat(idata)["mu"].item()

    # I² approximation: τ² / (τ² + σ²_typical)
    # We use median within-trial variance as σ²_typical
    # (full computation requires the data, deferred to caller)

    return {
        "pooled_hr": float(np.exp(posterior_mu.mean())),
        "pooled_hr_cri_lower": float(np.exp(hdi_mu[0])),
        "pooled_hr_cri_upper": float(np.exp(hdi_mu[1])),
        "tau_mean": float(posterior_tau.mean()),
        "tau_sd": float(posterior_tau.std()),
        "r_hat_mu": round(r_hat, 3),
    }
