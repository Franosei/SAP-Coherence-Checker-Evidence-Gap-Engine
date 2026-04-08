# SAP Coherence Checker and Evidence Gap Engine v3.0

This repository runs a breast cancer trial audit pipeline. It:

- fetches completed Phase 2 and Phase 3 interventional breast cancer trials from ClinicalTrials.gov
- links each trial to a PubMed publication
- compares registered primary endpoints against published primary endpoints
- routes ambiguous endpoint pairs to OpenAI for adjudication
- records human review decisions in append-only audit logs
- generates calibration, scorecard, and power audit outputs
- supports sequential Bayesian evidence synthesis on confirmed poolable pairs

## Scope

The current pipeline is configured for breast cancer trials across:

- `her2_positive`
- `hr_positive`
- `tnbc`

Treatment setting is classified as:

- `neoadjuvant`
- `adjuvant`
- `metastatic`

Trials with unclear subtype or setting are retained and flagged for review.

## Pipeline

### Module 1: Trial fetch and publication linkage

File: `src/pipeline/module1_linker.py`

This module:

- queries the ClinicalTrials.gov v2 API
- extracts registered primary and secondary outcomes from the protocol section
- classifies subtype and treatment setting from title, conditions, and eligibility text
- links trials to PubMed using a staged NCT-to-PMID cascade
- writes each linkage decision to `data/logs/linkage_audit_log.csv`

Linkage stages:

| Stage | Method | Output |
|---|---|---|
| 0 | CT.gov submitted PMID | High confidence link |
| 1 | Direct NCT ID PubMed search | High confidence link |
| 2 | Title similarity using Jaccard score | High or medium confidence link |
| 3 | Author and year check | Medium or low confidence link |

### Article-type gate

File: `src/pipeline/article_classifier.py`

This gate filters PubMed records before endpoint matching. It is used to exclude records that are not primary results papers, such as:

- protocol papers
- design papers
- subgroup analyses
- editorials
- reviews
- pharmacokinetic reports
- safety-only reports

The gate uses:

- a heuristic classifier
- an OpenAI arbiter for uncertain cases

Gate decisions are written into the linkage notes so they remain visible in the audit trail.

### Module 2: Endpoint matching

File: `src/pipeline/module2_endpoint_matcher.py`

This module compares the registered endpoint from ClinicalTrials.gov against the published endpoint extracted from the PubMed abstract.

Layer 1 uses OpenAI embeddings. Layer 2 uses an OpenAI chat model for adjudication when the similarity score is ambiguous.

Routing logic:

| Similarity score | Routing |
|---|---|
| `>= 0.90` | Auto concordant |
| `0.50 to 0.89` | LLM adjudication |
| `< 0.50` | Auto major switch |

The module writes decisions to `data/logs/decision_log.csv`.

Switch directions:

| Direction | Meaning |
|---|---|
| `none` | Endpoints are concordant |
| `promotion_of_secondary` | A secondary endpoint was promoted |
| `composite_modified` | Composite components were changed |
| `timeframe_changed` | Timepoint or follow-up window changed |
| `endpoint_replaced` | A different endpoint was used |
| `surrogate_substituted` | A surrogate and a long-term endpoint were substituted for one another |

### Module 3: Bayesian evidence accumulation

File: `src/pipeline/module3_bayesian.py`

This module runs sequential Bayesian random-effects meta-analysis on trial pairs that have been confirmed as poolable after review.

### Module 4: Power audit

File: `src/pipeline/module4_power_audit.py`

This module:

- extracts hazard ratios and confidence intervals from PubMed abstracts
- back-calculates implied effect size assumptions
- compares assumptions with prior evidence where available
- writes outputs to `data/logs/power_audit_log.csv`

## Dashboard

File: `src/dashboard/app.py`

The dashboard is built with Shiny for Python. It reads the pipeline outputs and displays:

- overview metrics
- linkage review items
- endpoint review queue
- decision log
- AI calibration outputs
- scorecard outputs
- power audit outputs
- provenance records
- export tools

The dashboard is file-driven. It needs pipeline outputs in `data/outputs` and `data/logs` before it can show populated content.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows:

```powershell
.venv\Scripts\activate
Copy-Item .env.example .env
```

Required `.env` values:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
LLM_PROVIDER=openai
```

Optional:

```env
NCBI_API_KEY=...
```

`.env` is loaded automatically from `src/pipeline/config.py`.

## Run the pipeline

Full run:

```bash
python run_pipeline.py --fresh-run
```

Smoke test:

```bash
python run_pipeline.py --fresh-run --max-trials 10 --skip-bayesian
```

Resume from existing outputs:

```bash
python run_pipeline.py
```

Skip selected steps:

```bash
python run_pipeline.py --skip-linkage
python run_pipeline.py --skip-matching
python run_pipeline.py --skip-bayesian
```

## Run the dashboard

Recommended on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

Direct command:

```bash
python -m shiny run --port 8030 src/dashboard/app.py
```

Open:

```text
http://127.0.0.1:8030
```

## Tests

```bash
python -m pytest
python -m ruff check src tests run_pipeline.py
```

## Main outputs

Files written by the pipeline:

- `data/outputs/trials.csv`
- `data/outputs/linked_trials.csv`
- `data/outputs/matched_trials.csv`
- `data/outputs/effect_measures.json`
- `data/logs/linkage_audit_log.csv`
- `data/logs/decision_log.csv`
- `data/logs/power_audit_log.csv`

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
    article_classifier.py
    config.py
    hr_extractor.py
    module1_linker.py
    module2_endpoint_matcher.py
    module3_bayesian.py
    module4_power_audit.py
    pubmed_client.py
    scorecard.py
    validation.py

data/
  gold_standard/
  logs/
  outputs/

tests/
run_pipeline.py
run_dashboard.ps1
```
