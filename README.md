# SAP Coherence Checker and Evidence Gap Engine v3.0

This repository runs a breast cancer trial audit pipeline. It:

- fetches completed Phase 2 and Phase 3 interventional breast cancer trials from ClinicalTrials.gov
- builds the complete PubMed publication family for each trial
- identifies primary, interim, updated, final, secondary, subgroup, safety,
  quality-of-life, biomarker, follow-up, extension, and protocol/SAP papers
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

### Module 1: Trial fetch and publication-family construction

File: `src/pipeline/module1_linker.py`

This module:

- queries the ClinicalTrials.gov v2 API
- extracts registered primary and secondary outcomes from the protocol section
- classifies subtype and treatment setting from title, conditions, and eligibility text
- discovers candidates through ClinicalTrials.gov references, exact NCT searches,
  trial-identity searches, and forward/backward PubMed citation chaining (from a
  RESULT / exact-NCT seed)
- deduplicates by PMID, then DOI, then normalized title
- screens PubMed publication types: protocols, meta-analyses, systematic
  reviews, editorials, comments and case reports are excluded from the
  results question outright ("Clinical Trial Protocol" is a stronger negative
  signal than "Clinical Trial" is a positive one)
- runs a cheap identity prefilter to trim citation-chain noise, then makes
  **one LLM call per trial** over the shortlist. The LLM is not shown any
  registered endpoint. For every candidate it returns same_trial,
  reports_randomized_arms, reports_outcome_data, role, analysis_stage and
  population_scope; overall it names the single `primary_results_pmid` that
  reports the pre-specified primary analysis, or null
- linkage is **binary**: `SELECTED` (a committed primary-results PMID) or
  `NOT_FOUND`. Low-confidence picks stay SELECTED but carry
  `PRIMARY_RESULTS_NEEDS_REVIEW`. There is no "ambiguous" outcome
- without an API key, a conservative no-LLM fallback picks a single strong
  CT.gov-RESULT / exact-NCT results paper or returns NOT_FOUND
- writes each trial's row to `data/outputs/linked_trials.partial.csv` as it
  completes, so an interrupted linkage run resumes from the first unfinished NCT
- stores one auditable row per candidate (with the LLM's per-candidate verdict
  and reasoning) in `data/outputs/publication_family.csv`
- writes each linkage decision to `data/logs/linkage_audit_log.csv`

The trial output preserves aligned `ctgov_publication_pmids`,
`ctgov_publication_types`, and `ctgov_publication_urls` columns. Primary,
interim, updated, final, and secondary-result PMID lists are stored separately.

### Registry history gate

File: `src/pipeline/registry_history.py`

The pipeline attempts to preserve original, pre-recruitment,
pre-primary-completion, pre-publication, and current registry snapshots in
`data/outputs/registry_history.csv`.

This is an **optional enrichment**, not a requirement. ClinicalTrials.gov's
public API v2 exposes only the current record, and its internal version-history
service is not reachable programmatically (HTTP 403). When history is available,
endpoint comparison uses the latest pre-recruitment snapshot (falling back to the
original version) and `endpoint_switch_assessable = True`. When it is not, the
comparison falls back to the **current** registered endpoint (always available
from the Module 1 fetch) and `endpoint_switch_assessable = False` a concordant
result then cannot rule out a quiet registry edit toward the published endpoint,
and that limitation is recorded per row and carried into the decision log.

### Publication family & primary-paper selection

File: `src/pipeline/publication_family.py`

One LLM call per trial (endpoint withheld) both assigns every candidate a
scientific role and names the primary-results paper. Roles distinguished:

- primary, interim, updated, and final results
- protocol and SAP papers
- subgroup analyses
- safety, QoL/PRO, biomarker, long-term follow-up, and secondary-endpoint analyses

The registered primary endpoint is never in that prompt. A generic PubMed
`Randomized Controlled Trial` tag is not sufficient evidence that an article is
the primary analysis. An interim analysis is never selected as the primary
paper, even when it reports the primary endpoint on the full randomized cohort.

### Module 2: Endpoint matching

File: `src/pipeline/module2_endpoint_matcher.py`

This module runs only after publication-role classification, same-trial
verification, primary-paper selection, and registry-history resolution. It
compares the prespecified registered endpoint against the published endpoint
(PubMed Results section when available, full abstract as fallback). Only
`SELECTED` trials enter this comparison.

**v4.1: every pair is adjudicated by the LLM.** Endpoint-embedding cosine
similarity is still computed and stored (`similarity_score`) as an
informational reference and for the calibration view, but it no longer routes
anything — on short endpoint phrases cosine diverged too often from the
clinical switch-type judgement. Set `ENDPOINT_MATCHING_LLM_ONLY=false` to
restore the old three-band routing (`>=0.90` auto-concordant, `0.50-0.89` LLM,
`<0.50` auto-major-switch).

**Adjudicator prompt (v4.5).** The model acts as a conservative
*outcome-reporting adjudicator*. It is given the registered **primary** outcome
(original pre-recruitment snapshot when `registry_history` retrieved one, plus
the current registry wording), the registered **secondary** outcomes, a note on
whether registry history was available, and the publication's **Results +
Conclusion** text — and it must identify the *published primary* itself rather
than trust a noisy pre-extracted snippet. It is explicitly told **not** to call
a switch merely because the paper reports many hazard ratios, subgroups,
adjusted/unadjusted models, additional secondaries, or a changed treatment
comparison, or because the primary result was non-significant. When evidence is
ambiguous it must pick the less severe class and set `human_review_required`.
An abstract's failure to restate a qualifier, assessment schedule, or registered
timeframe is not evidence that the endpoint changed. A change classification
requires affirmative evidence that the actual published endpoint differs.
`major_switch` requires strong evidence that the registered primary was
omitted / replaced / demoted **and** the change was undisclosed. Its JSON
(`registered_primary_original`, `published_primary`, `primary_endpoint_match`,
`final_classification`, `confidence` 0–1, `disclosure_status`, evidence quotes,
`reasoning_summary`, …) is mapped deterministically onto the internal
`switch_type` / `direction` / `confidence_score` / `comparability_for_pooling` /
`flag_for_human_review` fields no re-judging in code.

**Five permitted classifications (v4.5):**

| value | meaning |
|---|---|
| `concordant` | **Concordant (no change)**: the same endpoint; wording, effect measure, or omitted abstract detail alone is not a change |
| `additional_outcome` | **Additional outcome (disclosed exploratory)**: an explicitly additional/exploratory outcome that does not replace the registered primary |
| `minor_modification` | **Outcome modification (timeframe/definition/population)**: the same underlying endpoint with an explicitly different substantive feature |
| `moderate_switch` | **Outcome switch — moderate / partly disclosed**: replacement or material alteration that is partly acknowledged or substantially overlaps the prespecified endpoint |
| `major_switch` | **Outcome switch — major / undisclosed**: an omitted/replaced materially different endpoint without disclosure or explanation |

The switch **rate** counts only `moderate_switch + major_switch`.

For trials with multiple registered primary endpoints, v4.5 creates one
structured comparison per endpoint before assigning the overall trial-level
classification. `same_construct`, `same_timeframe`, `same_definition`,
`same_population`, and `same_measurement_method` are tri-state: true, false, or
not stated/unclear. Only an affirmative false supported by change evidence can
trigger modification. The overall label is the most severe supported
endpoint-level result, while the full per-endpoint breakdown remains in the
decision log and review dashboard.

**Human-in-the-loop (simplified, Sept 2026).** The task is simple: does the
publication report the registered primary endpoint? A pair is routed to a
human ONLY when the model's numeric `confidence_score` is below
`ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD` (0.5) or there is no verdict at all
(a malformed/failed response) or the comparison couldn't be made
(no publication text, or no registered primary endpoint on the trial side).
A verdict at or above the threshold is accepted directly
(`human_reviewed = "auto_accepted"`) **regardless of switch severity** (a
confident `major_switch` is accepted like a confident `concordant`) **and
regardless of the model's advisory `flag_for_human_review` boolean** a
confident model that also ticks "you might want to look at this" is still
trusted. Deterministic guardrails already fold their residual uncertainty
into `confidence_score` (capped at 0.65), so a genuinely borderline guardrail
case still falls below a threshold set above 0.65.
`ENDPOINT_AUTO_ACCEPT` defaults to `true`; set it to `false` to send every
verdict to review during a recalibration.

A deterministic `SPOT_CHECK_RATE` (default 15%) sample of the auto-accepted
verdicts is still drawn into the queue so an AI-human agreement rate can be
reported even though those pairs weren't individually necessary to review.
Existing human decisions remain final and override the model in every
downstream metric. The review queue (`validation.pairs_needing_human_review`)
is: not-confident/no-verdict pairs + missing-publication-text pairs +
missing-registered-endpoint pairs + the spot-check sample, minus anything a
human has already resolved. Decisions are written to
`data/logs/decision_log.csv`; the human verdict (`human_final_class`)
overrides the LLM's for every downstream metric.

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

Random-effects meta-analysis on the trial pairs a human confirmed as poolable
(`human_poolable = True`). The `(mu, tau)` posterior is computed **exactly** by
1-D numerical marginalisation on a `tau` grid (mu conjugate-Normal given tau,
the trial effects integrated out analytically) no MCMC, no compiled backend,
the whole sequential run in ~2 s. `run_sequential_analysis` re-fits one trial at
a time in registration-date order to show when the evidence crossed clinical
significance.

**Per-endpoint pooling.** `run_pipeline.step6` fits the model *separately* for
each endpoint cluster on that cluster's own poolable effect measures. A cluster
with no poolable effect measure gets no pooled HR (an explicit evidence gap)
never a copy of the global posterior.

### HR / effect-measure extraction

File: `src/pipeline/hr_extractor.py`

Regex cascade first (explicit `HR 0.65 (95% CI ...)` forms); on failure an LLM
reads the full abstract + Results + Conclusion and returns the primary-endpoint
HR / OR / RR, or reports that no poolable ratio exists (pCR / ORR trials). The
audit log `data/logs/effect_measure_log.csv` holds **exactly one row per
trial-publication pair** and is rewritten each run re-runs cannot duplicate
rows, and pairs already attempted are not re-sent to the LLM.

### Module 4: Power audit

File: `src/pipeline/module4_power_audit.py`

- back-calculates the implied effect-size assumption from each trial's
  registered enrollment
- compares it against the Bayesian posterior available at the trial's
  registration date (optimism bias)
- writes `data/logs/power_audit_log.csv` one row per trial, rewritten each run

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

### C compiler for the Bayesian model (Module 3)

PyMC/PyTensor needs a C++ compiler to compile the sampler; without one it falls
back to slow pure-Python evaluation. On a standalone (non-conda) Windows Python:

```powershell
winget install --id BrechtSanders.WinLibs.POSIX.UCRT --scope user
```

This adds `g++` to your user `PATH`; open a new terminal afterwards. PyTensor
auto-detects it (`python -c "import pytensor; print(pytensor.config.cxx)"` should
print the g++ path). With conda instead, use `conda install gxx`.

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

Linkage (Step 2) checkpoints after every trial. If it is interrupted, re-running
resumes from the first unfinished NCT via `data/outputs/linked_trials.partial.csv`
(delete that file to force a clean rebuild; `--fresh-run` also clears it).

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
- `data/outputs/publication_family.csv`
- `data/outputs/registry_history.csv`
- `data/outputs/matched_trials.csv`
- `data/outputs/effect_measures.json`
- `data/outputs/scorecard.csv` one row per endpoint cluster (per-cluster pooled HR)
- `data/outputs/switching_summary.csv` outcome-switching characterisation:
  overall + per-cluster + per-switch-form rates, results-driven fraction,
  disclosed-exploratory count, human-confirmed vs AI-only split, AI–human
  agreement rate
- `data/logs/linkage_audit_log.csv`
- `data/logs/decision_log.csv`
- `data/logs/effect_measure_log.csv` one row per pair, rewritten each run
- `data/logs/power_audit_log.csv` one row per trial, rewritten each run

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
    hr_extractor.py
    module1_linker.py
    module2_endpoint_matcher.py
    module3_bayesian.py
    module4_power_audit.py
    publication_family.py
    pubmed_client.py
    registry_history.py
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
