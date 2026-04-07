# SAP Coherence Checker & Evidence Gap Engine

Human-in-the-loop pipeline for linking HFrEF trials to publications, detecting endpoint switching, and surfacing evidence gaps with an auditable review trail.

## Architecture

1. `module1_linker.py`
   ClinicalTrials.gov to PubMed linkage with an append-only linkage audit log.
2. `module2_endpoint_matcher.py`
   Embedding similarity routing plus OpenAI adjudication for ambiguous endpoint pairs.
3. `module3_bayesian.py`
   Bayesian evidence accumulation over human-confirmed poolable trial pairs.
4. `module4_power_audit.py`
   Assumed effect-size back-calculation versus evidence-informed posterior estimates.
5. `src/dashboard/app.py`
   Shiny for Python dashboard for review, provenance, calibration, and export.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` with:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

The application now loads `.env` automatically from `src/pipeline/config.py`, so the CLI pipeline, tests, and dashboard share the same OpenAI settings.

## Running the pipeline

For a standard local run that generates the dashboard inputs in one step:

```bash
python run_pipeline.py --fresh-run --max-trials 20
```

This performs:

1. `ClinicalTrials.gov` fetch
2. true `PubMed` linkage cascade with linkage audit log
3. registered-versus-published endpoint matching
4. HR extraction from the linked publication
5. optional Bayesian pooling on human-confirmed poolable pairs

You can also run the modules directly:

```python
from src.pipeline.module1_linker import fetch_hfref_trials, link_to_pubmed
from src.pipeline.module2_endpoint_matcher import run_endpoint_matching

trials = fetch_hfref_trials(max_records=20)
linked = link_to_pubmed(trials)
matched = run_endpoint_matching(linked)
```

## Running the dashboard

```bash
python -m shiny run --reload src/dashboard/app.py
```

On Windows, the repo also includes a convenience launcher that uses port `8010`
to avoid common conflicts on `8000`:

```powershell
.\run_dashboard.ps1
```

The dashboard includes:

- Overview
- Linkage-aware review queue backed by the decision log
- Review Queue
- Decision Log
- AI Calibration
- Evidence Scorecard
- Power Audit
- Provenance
- Export

## Running tests

```bash
python -m pytest
python -m pytest --cov=src --cov-report=term-missing
```

## Project structure

```text
src/
  dashboard/
    app.py
    helpers.py
    www/style.css
  models/
    decision_log.py
    linkage_log.py
    schemas.py
  pipeline/
    config.py
    module1_linker.py
    module2_endpoint_matcher.py
    module3_bayesian.py
    module4_power_audit.py
    scorecard.py

data/
  gold_standard/
  logs/

tests/
```

## Governance principles

- Every AI decision is logged.
- Human review is structural, not optional, for ambiguous rows.
- Gold-standard and inter-rater templates can be generated from the dashboard.
- Reviewer actions are written back to the audit log with initials and timestamps.
- Export produces a reproducible package of the current audit artefacts.
