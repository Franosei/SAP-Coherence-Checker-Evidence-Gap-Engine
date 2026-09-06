"""
Pipeline configuration.

Single source of truth for every threshold, model identifier, API setting,
and filesystem path used across the SAP Coherence Checker pipeline.

Environment variables are read from a ``.env`` file in the project root.
Copy ``.env.example`` to ``.env`` and populate all required values before
running the pipeline or dashboard.

Sections
--------
- Project paths
- Embedding similarity thresholds    (Section 3.3.1)
- LLM settings                       (Section 3.3.2)
- ClinicalTrials.gov query params    (Section 3.2.1)
- Breast cancer population classifier
- PubMed / NCBI E-utilities          (Section 3.1)
- NCT-to-PMID linkage thresholds     (Section 3.2.2)
- Bayesian model parameters          (Section 3.4.3)
- Validation targets                 (Section 5.2)
- Human review governance            (Section 5.3)
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths and environment loading
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = ROOT_DIR / ".env"

# Load once so CLI runs, tests, and the dashboard all resolve the same
# environment without requiring a manual dotenv call at each entry point.
load_dotenv(ENV_FILE, override=False)

DATA_DIR: Path = ROOT_DIR / "data"
LOGS_DIR: Path = DATA_DIR / "logs"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"
GOLD_STANDARD_PATH: Path = DATA_DIR / "gold_standard" / "gold_standard.csv"
INTER_RATER_REVIEW_PATH: Path = DATA_DIR / "gold_standard" / "inter_rater_review.csv"
DECISION_LOG_PATH: Path = LOGS_DIR / "decision_log.csv"
LINKAGE_LOG_PATH: Path = LOGS_DIR / "linkage_audit_log.csv"
PUBLICATION_FAMILY_PATH: Path = OUTPUTS_DIR / "publication_family.csv"
# Per-trial linkage checkpoint. Appended as each trial completes so an
# interrupted linkage run resumes from the first unfinished NCT.
LINKAGE_CHECKPOINT_PATH: Path = OUTPUTS_DIR / "linked_trials.partial.csv"
REGISTRY_HISTORY_PATH: Path = OUTPUTS_DIR / "registry_history.csv"
# Module 4 effect-measure extraction audit (one row per trial-publication pair,
# rewritten each run — never appended).
EFFECT_MEASURE_LOG_PATH: Path = LOGS_DIR / "effect_measure_log.csv"
# Module 4 optimism-bias audit (one row per trial, rewritten each run).
POWER_AUDIT_LOG_PATH: Path = LOGS_DIR / "power_audit_log.csv"
EFFECT_MEASURES_PATH: Path = OUTPUTS_DIR / "effect_measures.json"
SWITCHING_SUMMARY_PATH: Path = OUTPUTS_DIR / "switching_summary.csv"
BAYES_TRACE_DIR: Path = LOGS_DIR / "bayes_traces"

# Bump this whenever a pipeline change would alter existing log entries.
# A new version tag is required before any re-run that modifies the log.
PIPELINE_VERSION: str = "pipeline_v4.5"

# ---------------------------------------------------------------------------
# Endpoint-matching routing (Section 3.3.1)
# ---------------------------------------------------------------------------

# v4.1: every trial-publication pair is adjudicated by the LLM. Cosine
# similarity of endpoint embeddings is still computed and stored as an
# informational reference (and for the calibration view), but it no longer
# routes anything — cosine on short endpoint phrases diverged too often from
# the clinical judgement the switch-type classification needs.
ENDPOINT_MATCHING_LLM_ONLY: bool = os.getenv(
    "ENDPOINT_MATCHING_LLM_ONLY", "true"
).strip().lower() not in {"0", "false", "no"}

# Human-in-the-loop policy for Module 2 verdicts (simplified per user directive,
# Sept 2026): the task is simple — does the publication report the registered
# primary endpoint? A human is needed ONLY when the model itself is not
# confident either way, or the comparison couldn't be made at all (no
# publication text, or no registered endpoint on the trial side). A confident
# verdict is accepted directly, WHETHER OR NOT it is a switch — the switch/
# concordant distinction is not, on its own, a reason to route to a human.
# A deterministic spot-check sample of confident verdicts is still drawn (see
# SPOT_CHECK_RATE) so an AI-human agreement rate can be reported.
ENDPOINT_AUTO_ACCEPT: bool = os.getenv("ENDPOINT_AUTO_ACCEPT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}

ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD", "0.5")
)

# Retained only for the (now unused) _route_from_score helper and the
# calibration plot's reference bands.
SIMILARITY_AUTO_CONCORDANT: float = 0.90
SIMILARITY_LLM_LOWER: float = 0.50

# ---------------------------------------------------------------------------
# Embedding model settings (Section 3.3.1 — Layer 1)
# ---------------------------------------------------------------------------
# OpenAI text-embedding-3-small — 1 536-dimensional unit vectors.
# Cosine similarity = dot product (no normalisation step needed).
# Pricing (2025): $0.00002 / 1 000 tokens. Full run of ~200 pairs < $0.05.

EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE: int = 64  # Max texts per embeddings API call
EMBEDDING_COST_PER_1K_USD: float = 0.000_020

# ---------------------------------------------------------------------------
# LLM settings (Section 3.3.2)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_PRIMARY: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Raised from 1_024 — a reasoning model spends part of this budget on hidden
# reasoning tokens before the visible JSON, and Module 2's structured
# endpoint_comparisons schema was already truncating ("Unterminated string")
# some responses at the old ceiling.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE: float = 0.0
LLM_COST_CEILING_USD: float = 50.0

# Reasoning effort for reasoning-capable models (o-series, gpt-5 family, ...).
# Ignored (and not sent) for classic chat models — see LLM_MODEL_IS_REASONING.
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "medium")

# Classic chat models (gpt-4*, gpt-3.5*) take `temperature` + `max_tokens` and
# reject `reasoning_effort`. Reasoning models reject a non-default
# `temperature`, use `max_completion_tokens` (which also counts hidden
# reasoning tokens) instead of `max_tokens`, and accept `reasoning_effort`.
# This is a name-based heuristic — override with LLM_MODEL_IS_REASONING=false
# in .env if a future non-"gpt-4*/gpt-3*" model turns out not to be one.
LLM_MODEL_IS_REASONING: bool = os.getenv(
    "LLM_MODEL_IS_REASONING",
    "true" if not LLM_MODEL_PRIMARY.lower().startswith(("gpt-4", "gpt-3")) else "false",
).strip().lower() in {"1", "true", "yes"}


def llm_sampling_kwargs(max_output_tokens: int) -> dict:
    """Provider-correct sampling kwargs for a ``chat.completions.create`` call
    against ``LLM_MODEL_PRIMARY``. Centralised so every call site (linkage,
    endpoint adjudication, HR extraction, article classification) stays
    consistent when the model is switched.
    """
    if LLM_MODEL_IS_REASONING:
        return {
            "max_completion_tokens": max_output_tokens,
            "reasoning_effort": LLM_REASONING_EFFORT,
        }
    return {"max_tokens": max_output_tokens, "temperature": LLM_TEMPERATURE}


# Token cost estimates (USD per 1 000 tokens). Calibrated for gpt-4o-mini —
# THESE ARE STALE for any other model (including the current LLM_MODEL_PRIMARY
# if it has been changed) until updated to that model's real published
# pricing. Cost figures logged/reported by the pipeline are only meaningful
# once these are corrected.
LLM_COST_PER_1K_INPUT_USD: float = float(os.getenv("LLM_COST_PER_1K_INPUT_USD", "0.000150"))
LLM_COST_PER_1K_OUTPUT_USD: float = float(os.getenv("LLM_COST_PER_1K_OUTPUT_USD", "0.000600"))

# ---------------------------------------------------------------------------
# ClinicalTrials.gov query parameters (Section 3.2.1)
# ---------------------------------------------------------------------------

CT_BASE_URL: str = "https://clinicaltrials.gov/api/v2/studies"

# Condition query terms — intentionally broad at the API level so we do not
# miss trials registered simply as "breast cancer".  The post-fetch population
# classifier (``_classify_population`` in module1_linker) enforces the finer
# subtype boundary (HER2+, HR+/HER2-, TNBC) and treatment setting
# (neoadjuvant, adjuvant, metastatic).
CT_CONDITIONS: list[str] = [
    "breast cancer",
    "breast neoplasm",
    "breast carcinoma",
    "HER2-positive breast cancer",
    "triple negative breast cancer",
]

CT_PHASES: list[str] = ["PHASE2", "PHASE3"]
CT_STUDY_TYPE: str = "INTERVENTIONAL"
CT_STATUS: str = "COMPLETED"
CT_PAGE_SIZE: int = 100
CT_REQUEST_TIMEOUT_S: int = 30


# Date window: 15 years back → today. Only trials with posted results.
def _fmt_date(d: _dt.date) -> str:
    return d.strftime("%m/%d/%Y")


CT_COMPLETION_END: str = _fmt_date(_dt.date.today())
CT_COMPLETION_START: str = _fmt_date(_dt.date.today().replace(year=_dt.date.today().year - 15))
CT_REQUIRE_RESULTS: bool = True  # aggFilters=results:with

# ---------------------------------------------------------------------------
# Breast cancer population classifier (post-fetch filter)
# ---------------------------------------------------------------------------
# The classifier reads each trial's eligibility criteria text, title, and
# conditions list to assign a 2-D label: subtype × treatment setting.
#
# Subtype classes  (bc_subtype column)
# ------------------------------------
# her2_positive    — HER2-positive (HER2+) / HER2-amplified
# hr_positive      — HR+/HER2-negative (ER+ and/or PR+, HER2-)
# tnbc             — Triple-negative (ER-, PR-, HER2-)
# unknown_subtype  — Subtype not determinable from available text
#
# Setting classes  (bc_setting column)
# -------------------------------------
# neoadjuvant      — Pre-surgical treatment
# adjuvant         — Post-surgical treatment
# metastatic       — Advanced / metastatic disease
# unknown_setting  — Setting not determinable from available text
#
# Eligibility is a confirmed subtype AND a confirmed setting. Trials the
# classifier cannot place on both axes ("bc_flagged") are hard-excluded at
# fetch, alongside non-breast trials ("non_breast_excluded"). Only their NCT
# IDs are logged.
CT_EXCLUDE_POPULATION_CLASSES: list[str] = ["non_breast_excluded", "bc_flagged"]

# ---------------------------------------------------------------------------
# PubMed / NCBI E-utilities (Section 3.1)
# ---------------------------------------------------------------------------

PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_REQUEST_TIMEOUT_S: int = 30

# NCBI API key raises the rate limit from 3 req/s to 10 req/s.
# Register free at: https://www.ncbi.nlm.nih.gov/account/
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")

# Sleep interval between NCBI requests to stay within rate limits.
# 0.11 s → ~9 req/s (within 10/s limit when key is present).
# 0.34 s → ~3 req/s (within 3/s limit without key).
PUBMED_RATE_LIMIT_S: float = 0.11 if NCBI_API_KEY else 0.34

# ---------------------------------------------------------------------------
# Bayesian model parameters (Section 3.4.3)
# ---------------------------------------------------------------------------

# fit_random_effects_model computes the (mu, tau) posterior EXACTLY by 1-D
# numerical marginalisation (no MCMC) — the whole 44-fit sequence runs in ~2 s.
# CHAINS x DRAWS is just how many posterior draws to synthesise for arviz.
BAYES_CHAINS: int = 4
BAYES_DRAWS: int = 1_000
BAYES_WARMUP: int = 1_000  # retained for config compatibility; unused by the analytic fit
BAYES_TARGET_ACCEPT: float = 0.9  # retained for config compatibility; unused by the analytic fit
# Sequential analysis fits the model at n = stride, 2*stride, ... N (plus N).
# stride=1 fits after every trial (finest "when did evidence cross" resolution);
# raise it to trade resolution for wall-clock (each fit is minutes in Python mode).
BAYES_SEQUENTIAL_STRIDE: int = int(os.getenv("BAYES_SEQUENTIAL_STRIDE", "1"))
BAYES_PRIOR_MU_MEAN: float = 0.0  # Weakly informative, centred on null
BAYES_PRIOR_MU_SD: float = 0.5
BAYES_PRIOR_TAU_SD: float = 0.5  # HalfNormal — allows meaningful heterogeneity

# ---------------------------------------------------------------------------
# Validation targets (Section 5.2)
# ---------------------------------------------------------------------------

VALIDATION_TARGET_AUC: float = 0.80
VALIDATION_TARGET_PRECISION: float = 0.75
VALIDATION_TARGET_RECALL: float = 0.80
VALIDATION_LLM_LOW_CONF_FLAG_RATE: float = 0.80

# ---------------------------------------------------------------------------
# Human review governance thresholds (Section 5.3)
# ---------------------------------------------------------------------------

# Override rate > 30 % → AI pipeline is poorly calibrated; halt and recalibrate.
OVERRIDE_RATE_HIGH_THRESHOLD: float = 0.30

# Override rate < 5 % with high spot-check error → potential automation bias.
OVERRIDE_RATE_LOW_THRESHOLD: float = 0.05

# Fraction of AUTO-ACCEPTED verdicts drawn (deterministically) into the human
# spot-check sample so an AI-human agreement rate can be reported. Every
# outcome-switch verdict is reviewed regardless of this rate.
SPOT_CHECK_RATE: float = float(os.getenv("SPOT_CHECK_RATE", "0.15"))
