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
- Embedding similarity thresholds  (Section 3.3.1)
- LLM settings                     (Section 3.3.2)
- ClinicalTrials.gov query params  (Section 3.2.1)
- PubMed / NCBI E-utilities        (Section 3.1)
- NCT-to-PMID linkage thresholds   (Section 3.2.2)
- Bayesian model parameters        (Section 3.4.3)
- Validation targets               (Section 5.2)
- Human review governance          (Section 5.3)
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

DATA_DIR: Path           = ROOT_DIR / "data"
LOGS_DIR: Path           = DATA_DIR / "logs"
OUTPUTS_DIR: Path        = DATA_DIR / "outputs"
GOLD_STANDARD_PATH: Path = DATA_DIR / "gold_standard" / "gold_standard.csv"
INTER_RATER_REVIEW_PATH: Path = DATA_DIR / "gold_standard" / "inter_rater_review.csv"
DECISION_LOG_PATH: Path  = LOGS_DIR / "decision_log.csv"
LINKAGE_LOG_PATH: Path   = LOGS_DIR / "linkage_audit_log.csv"
POWER_AUDIT_LOG_PATH: Path = LOGS_DIR / "power_audit_log.csv"
BAYES_TRACE_DIR: Path    = LOGS_DIR / "bayes_traces"

# Bump this whenever a pipeline change would alter existing log entries.
# A new version tag is required before any re-run that modifies the log.
PIPELINE_VERSION: str = "pipeline_v2.0"

# ---------------------------------------------------------------------------
# Embedding similarity routing thresholds (Section 3.3.1)
# ---------------------------------------------------------------------------

# Pairs at or above this score are auto-classified as concordant; LLM skipped.
SIMILARITY_AUTO_CONCORDANT: float = 0.90

# Pairs below this score are auto-classified as major switch; LLM skipped.
# Everything between SIMILARITY_LLM_LOWER and SIMILARITY_AUTO_CONCORDANT
# is routed to the LLM layer.
SIMILARITY_LLM_LOWER: float = 0.50

# ---------------------------------------------------------------------------
# Embedding model settings (Section 3.3.1 — Layer 1)
# ---------------------------------------------------------------------------
# OpenAI text-embedding-3-small — 1 536-dimensional unit vectors.
# Cosine similarity = dot product (no normalisation step needed).
# Pricing (2025): $0.00002 / 1 000 tokens. Full run of ~200 pairs < $0.05.

EMBEDDING_MODEL: str             = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE: int        = 64      # Max texts per embeddings API call
EMBEDDING_COST_PER_1K_USD: float = 0.000_020

# ---------------------------------------------------------------------------
# LLM settings (Section 3.3.2)
# ---------------------------------------------------------------------------
LLM_PROVIDER: str        = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_PRIMARY: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_BASE_URL: str        = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MAX_TOKENS: int      = 1_024
LLM_TEMPERATURE: float   = 0.0
LLM_COST_CEILING_USD: float = 50.0

# Token cost estimates (USD per 1 000 tokens) for the default gpt-4o-mini.
# Update these if switching to a different model.
LLM_COST_PER_1K_INPUT_USD: float  = 0.000_150
LLM_COST_PER_1K_OUTPUT_USD: float = 0.000_600

# ---------------------------------------------------------------------------
# ClinicalTrials.gov query parameters (Section 3.2.1)
# ---------------------------------------------------------------------------

CT_BASE_URL: str  = "https://clinicaltrials.gov/api/v2/studies"

# Condition query terms — intentionally broad at the API level so we do not
# miss trials registered as plain "Heart Failure" whose HFrEF specificity only
# appears in the eligibility criteria (LVEF threshold).  The post-fetch
# population classifier (``_classify_population`` in module1_linker) enforces
# the stricter HFrEF boundary.
CT_CONDITIONS: list[str] = [
    "heart failure with reduced ejection fraction",
    "HFrEF",
    "systolic heart failure",
    "left ventricular systolic dysfunction",
    "heart failure",   # catch-all for trials using the generic condition label
]

CT_PHASES: list[str]      = ["PHASE2", "PHASE3"]
CT_STUDY_TYPE: str        = "INTERVENTIONAL"
CT_STATUS: str            = "COMPLETED"
CT_PAGE_SIZE: int         = 100
CT_REQUEST_TIMEOUT_S: int = 30

# Date window: 15 years back → today. Only trials with posted results.
def _fmt_date(d: _dt.date) -> str:
    return d.strftime("%m/%d/%Y")

CT_COMPLETION_END: str   = _fmt_date(_dt.date.today())
CT_COMPLETION_START: str = _fmt_date(
    _dt.date.today().replace(year=_dt.date.today().year - 15)
)
CT_REQUIRE_RESULTS: bool = True  # aggFilters=results:with

# ---------------------------------------------------------------------------
# HFrEF population classifier thresholds (post-fetch filter)
# ---------------------------------------------------------------------------
# The classifier reads each trial's eligibility criteria text and title to
# determine whether the enrolled population is HFrEF-specific.
#
# LVEF ceiling: trials requiring LVEF ≤ this value are classified as HFrEF.
# Standard HFrEF threshold is 40 %; some landmark trials used 35 % or 45 %.
# Setting this to 45 % captures all accepted HFrEF definitions while excluding
# HFpEF trials (typically LVEF ≥ 50 %) and most HFmrEF trials (LVEF 40–49 %).
CT_HFREF_LVEF_CEILING: int = 45

# Trials classified as these statuses are removed from the dataset before
# linkage and are written to the exclusion log with their reason code.
# "hfpef_excluded" → confirmed HFpEF population, incompatible for pooling.
# "ambiguous"      → kept in dataset but flagged; human reviewer decides.
CT_EXCLUDE_POPULATION_CLASSES: list[str] = ["hfpef_excluded"]

# ---------------------------------------------------------------------------
# PubMed / NCBI E-utilities (Section 3.1)
# ---------------------------------------------------------------------------

PUBMED_BASE_URL: str        = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_REQUEST_TIMEOUT_S: int = 30

# NCBI API key raises the rate limit from 3 req/s to 10 req/s.
# Register free at: https://www.ncbi.nlm.nih.gov/account/
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")

# Sleep interval between NCBI requests to stay within rate limits.
# 0.11 s → ~9 req/s (within 10/s limit when key is present).
# 0.34 s → ~3 req/s (within 3/s limit without key).
PUBMED_RATE_LIMIT_S: float = 0.11 if NCBI_API_KEY else 0.34

# Maximum candidate PMIDs returned by a PubMed title search (Stage 2 linkage).
PUBMED_SEARCH_MAX_RESULTS: int = 5

# ---------------------------------------------------------------------------
# NCT-to-PMID linkage cascade thresholds (Section 3.2.2)
# ---------------------------------------------------------------------------
# Jaccard token similarity thresholds for title-based fuzzy matching.
# Score ≥ HIGH  → High confidence match (no further disambiguation needed).
# Score ≥ MEDIUM but < HIGH → Medium confidence; author+year check applied.
# Score <  MEDIUM → Low/Unlinked; flagged for human review.

LINKAGE_JACCARD_HIGH: float   = 0.70
LINKAGE_JACCARD_MEDIUM: float = 0.50

# ---------------------------------------------------------------------------
# Bayesian model parameters (Section 3.4.3)
# ---------------------------------------------------------------------------

BAYES_CHAINS: int          = 4
BAYES_DRAWS: int           = 2_000
BAYES_WARMUP: int          = 1_000
BAYES_TARGET_ACCEPT: float = 0.90
BAYES_PRIOR_MU_MEAN: float = 0.0   # Weakly informative, centred on null
BAYES_PRIOR_MU_SD: float   = 0.5
BAYES_PRIOR_TAU_SD: float  = 0.5   # HalfNormal — allows meaningful heterogeneity

# ---------------------------------------------------------------------------
# Validation targets (Section 5.2)
# ---------------------------------------------------------------------------

VALIDATION_TARGET_AUC: float            = 0.80
VALIDATION_TARGET_PRECISION: float      = 0.75
VALIDATION_TARGET_RECALL: float         = 0.80
VALIDATION_LLM_LOW_CONF_FLAG_RATE: float = 0.80

# ---------------------------------------------------------------------------
# Human review governance thresholds (Section 5.3)
# ---------------------------------------------------------------------------

# Override rate > 30 % → AI pipeline is poorly calibrated; halt and recalibrate.
OVERRIDE_RATE_HIGH_THRESHOLD: float = 0.30

# Override rate < 5 % with high spot-check error → potential automation bias.
OVERRIDE_RATE_LOW_THRESHOLD: float  = 0.05

# Fraction of auto-classified pairs sampled for spot-check validation.
SPOT_CHECK_RATE: float = 0.10
