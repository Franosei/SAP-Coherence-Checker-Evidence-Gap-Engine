# SAP Coherence Checker and Evidence Gap Engine (v4.5)

An auditable pipeline that detects **outcome switching** - a discrepancy between
the pre-registered primary endpoint and the primary endpoint actually reported -
in completed Phase 2/3 breast-cancer randomised trials, and then synthesises the
posterior treatment effect that survives once endpoint compatibility has been
established.

It:

- fetches completed Phase 2/3 interventional breast-cancer trials with posted
  results from ClinicalTrials.gov
- builds each trial's PubMed publication family and commits to a single
  primary-results paper (one LLM call per trial, registered endpoint withheld)
- adjudicates the registered vs published primary endpoint into a five-category
  schema, with a confidence-gated human-review queue
- extracts the primary-endpoint effect estimate (HR / OR / RR)
- fits a per-endpoint-family Bayesian random-effects meta-analysis on the
  human-confirmed poolable pairs (exact 1-D marginalisation, no MCMC)
- back-calculates each trial's optimism bias against the evidence available at
  its registration date
- writes a per-cluster scorecard and an outcome-switching characterisation

**LLM:** `OPENAI_MODEL` (`.env`), default `gpt-5.6-luna` at
`LLM_REASONING_EFFORT=medium`. Endpoint-embedding similarity uses
`text-embedding-3-small`.

## Scope

Breast-cancer subtypes: `her2_positive` | `hr_positive` | `tnbc`.
Treatment settings: `neoadjuvant` | `adjuvant` | `metastatic`.
Trials the classifier cannot place on both axes (`bc_flagged`) are hard-excluded
at fetch.

## Pipeline (`run_pipeline.py`)

### Module 1 - trial fetch and publication-family construction

`src/pipeline/module1_linker.py` + `publication_family.py` + `registry_history.py`

- queries the ClinicalTrials.gov v2 API; extracts registered primary and
  secondary outcomes; classifies subtype x setting from title / conditions /
  eligibility text
- discovers candidate publications via CT.gov references, exact-NCT searches,
  trial-identity searches, and forward/backward PubMed citation chaining from a
  results / exact-NCT seed; deduplicates by PMID -> DOI -> normalised title
- screens PubMed publication types (protocol, meta-analysis, systematic review,
  editorial, comment, case report -> excluded from the results question)
- runs a cheap identity prefilter, then **one LLM call per trial** over the
  shortlist - endpoint withheld. Per candidate it returns `same_trial`,
  `reports_randomized_arms`, `reports_outcome_data`, `role`, `analysis_stage`,
  `population_scope`; overall it names the single `primary_results_pmid` (or
  null). An interim analysis is never selected, even on the full cohort.
- linkage is **binary**: `SELECTED` or `NOT_FOUND` (no "ambiguous"). Low-
  confidence picks stay `SELECTED` but carry `PRIMARY_RESULTS_NEEDS_REVIEW`.
- without an API key, a conservative no-LLM fallback picks one strong
  CT.gov-results / exact-NCT paper or returns `NOT_FOUND`
- resumable: each trial's row is appended to
  `data/outputs/linked_trials.partial.csv` as it completes
- one auditable row per candidate -> `data/outputs/publication_family.csv`;
  linkage decisions -> `data/logs/linkage_audit_log.csv`

**Registry history** is an optional enrichment. CT.gov's public API exposes only
the current record and its version-history service returns HTTP 403, so when a
pre-recruitment snapshot cannot be retrieved the comparison falls back to the
**current** registered endpoint with `endpoint_switch_assessable = False` - a
concordant result then cannot rule out a quiet registry edit, and that caveat is
carried into the decision log.

### Module 2 - endpoint adjudication

`src/pipeline/module2_endpoint_matcher.py`

Runs on `SELECTED` trials only. Endpoint-embedding cosine similarity is computed
and stored (`similarity_score`) but **does not route anything** - every pair is
adjudicated by the LLM (`ENDPOINT_MATCHING_LLM_ONLY=true`; set to `false` to
restore the legacy three-band cosine routing).

The model acts as a conservative **outcome-reporting adjudicator**. It is given
the registered **primary** outcome (original snapshot + current wording), the
registered **secondary** outcomes, a registry-history-availability note, and the
publication's **Results + Conclusion** text, and must identify the *published
primary* itself. It is told **not** to call a switch for many hazard ratios,
subgroups, adjusted/unadjusted models, extra secondaries, a changed treatment
comparison, or a non-significant primary; and that an abstract failing to restate
a qualifier or timeframe is **not** evidence of a change. A change classification
requires affirmative textual evidence. For a trial with multiple registered
primaries the model returns one structured `endpoint_comparisons` item per
endpoint (tri-state `same_construct` / `same_timeframe` / ... : true / false /
not-stated); a fixed decision tree then derives the overall label and
`_adjudication_to_schema()` maps everything onto the internal
`LLMEndpointClassification` - no re-judging in code.

**Five categories** (switch **rate** = `moderate_switch + major_switch` only):

| value | meaning |
|---|---|
| `concordant` | same endpoint; wording / effect measure / omitted abstract detail alone is not a change |
| `additional_outcome` | an explicitly additional / exploratory outcome that does not replace the registered primary |
| `minor_modification` | same underlying endpoint, an explicitly different substantive feature (timeframe / definition / population) |
| `moderate_switch` | replacement or material alteration, partly acknowledged or substantially overlapping the prespecified endpoint |
| `major_switch` | an omitted / replaced materially different endpoint, undisclosed |

**Human review (confidence-gated).** A pair is routed to a human **only** when
the model's numeric `confidence_score` is below
`ENDPOINT_REVIEW_CONFIDENCE_THRESHOLD` (0.5), or there is no verdict at all
(malformed response), or the comparison could not be made (no publication text,
no registered endpoint). A verdict at/above the threshold is accepted directly
(`human_reviewed = "auto_accepted"`) **regardless of switch severity** and
regardless of the model's advisory `flag_for_human_review` flag - deterministic
guardrails already fold their residual uncertainty into `confidence_score` (cap
0.65). `ENDPOINT_AUTO_ACCEPT` defaults `true`; set `false` to review every
verdict during a recalibration. A deterministic `SPOT_CHECK_RATE` (15%) sample of
the accepted verdicts is still queued so an AI/human agreement rate can be
reported. The queue lives in `validation.pairs_needing_human_review`; the human
verdict (`human_final_class`) overrides the model for every downstream metric.
Decisions are written to `data/logs/decision_log.csv`.

### HR / effect-measure extraction

`src/pipeline/hr_extractor.py`

Regex cascade first (explicit `HR 0.65 (95% CI ...)` forms); on failure an LLM
reads the full abstract + Results + Conclusion and returns the primary-endpoint
HR / OR / RR, or reports that no poolable ratio exists (pCR / ORR trials).
`data/logs/effect_measure_log.csv` holds exactly one row per pair and is
**rewritten** each run - re-runs cannot duplicate rows, and pairs already
attempted are not re-sent to the LLM.

### Module 3 - Bayesian evidence accumulation

`src/pipeline/module3_bayesian.py`

Random-effects meta-analysis on the pairs a human confirmed poolable
(`human_poolable = True`). The `(mu, tau)` posterior is computed **exactly** by
1-D numerical marginalisation on a `tau` grid - no MCMC, no compiled backend
(the full run is ~2 s). `run_sequential_analysis` re-fits one trial at a time in
registration-date order. `run_pipeline` step 6 fits the model **separately per
endpoint family**; a cluster with no poolable effect measure gets no pooled HR
(an explicit gap), never a copy of the global posterior.

### Module 4 - power audit

`src/pipeline/module4_power_audit.py`

Back-calculates the implied effect-size assumption from each trial's registered
enrolment and compares it against the Bayesian posterior available at that
trial's registration date (optimism bias). `data/logs/power_audit_log.csv` - one
row per trial, rewritten each run.

## Dashboard

`src/dashboard/app.py` - Shiny for Python, file-driven (needs pipeline outputs in
`data/outputs` and `data/logs`). Overview metrics, linkage review, endpoint
review queue, decision log, AI calibration, scorecard, power audit, provenance,
export tools.

```bash
python -m shiny run --port 8030 src/dashboard/app.py        # -> http://127.0.0.1:8030
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # Windows: Copy-Item .env.example .env
```

Required `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.6-luna
LLM_REASONING_EFFORT=medium      # ignored for classic gpt-4*/gpt-3* models
LLM_PROVIDER=openai
```

Optional: `NCBI_API_KEY=...` (raises the PubMed rate limit 3->10 req/s).
`.env` is loaded automatically by `src/pipeline/config.py`.

> Module 3 uses exact numerical marginalisation, **not** PyMC/PyTensor - no C++
> compiler is required.

## Run the pipeline

```bash
python run_pipeline.py --fresh-run                       # full run from scratch
python run_pipeline.py --fresh-run --max-trials 10 --skip-bayesian   # smoke test
python run_pipeline.py                                   # resume from existing outputs
python run_pipeline.py --skip-linkage                    # reuse linked_trials.csv
python run_pipeline.py --skip-matching                   # stop after linkage
python run_pipeline.py --skip-bayesian                   # stop after HR extraction
python run_pipeline.py --resume-analysis                 # Module 2 + review done:
                                                         #   run only HR + Bayesian + scorecard
```

Step 2 (linkage) checkpoints after every trial to
`data/outputs/linked_trials.partial.csv` and resumes from the first unfinished
NCT. Step 3 resumes from `data/logs/decision_log.csv` - if that file is missing
but `matched_trials.csv` exists, the run **aborts** rather than silently
re-spending every LLM call (restore the log or pass `--fresh-run`).

**The decision log is the only home of the human review and is not tracked by
git.** Back it up after any review session; recover a lost one from OneDrive /
filesystem version history.

## Gold-standard calibration

```bash
python create_gold_standard.py                # blinded 20-pair template
# a human fills gold_switch_type for every row (endpoints only, no AI answer)
```

`validation.compute_ai_calibration()` (and the dashboard's AI Calibration tab)
then reports AUC / precision / recall of the pipeline's classifications against
the completed `data/gold_standard/gold_standard.csv`.

## Figures

```bash
python make_poster_figures.py                 # -> data/outputs/poster_figures/*.{png,pdf}
```

Greyscale, journal-technical. See `data/outputs/poster_figures/README.md`.

## Tests

```bash
python -m pytest
python -m ruff check .
```

## Main outputs

| Path | Contents |
|---|---|
| `data/outputs/trials.csv` | fetched CT.gov trials |
| `data/outputs/linked_trials.csv` | trials + committed primary-results paper |
| `data/outputs/publication_family.csv` | one row per candidate publication |
| `data/outputs/registry_history.csv` | registry snapshots (where retrievable) |
| `data/outputs/matched_trials.csv` | linked trials + `pair_id` / `similarity_score` |
| `data/outputs/effect_measures.json` | extracted poolable effect measures |
| `data/outputs/scorecard.csv` | one row per endpoint cluster (per-cluster pooled HR, I², optimism bias, evidence strength) |
| `data/outputs/switching_summary.csv` | overall + per-cluster + per-switch-form rates, results-driven fraction, human-confirmed vs AI-only, AI/human agreement |
| `data/logs/linkage_audit_log.csv` | one row per linkage decision |
| `data/logs/decision_log.csv` | one row per adjudicated pair + human review |
| `data/logs/effect_measure_log.csv` | one row per pair, rewritten each run |
| `data/logs/power_audit_log.csv` | one row per trial, rewritten each run |
| `data/logs/bayes_traces/*.nc` | posterior draws (per cluster + sequential) |

## Project structure

```text
run_pipeline.py            pipeline runner
create_gold_standard.py    blinded gold-standard template
make_poster_figures.py     figures
run_dashboard.ps1          dashboard launcher (Windows)

src/
  dashboard/     app.py, helpers.py, www/style.css
  models/        schemas.py, decision_log.py, linkage_log.py
  pipeline/      config.py, module1_linker.py, publication_family.py,
                 registry_history.py, pubmed_client.py,
                 module2_endpoint_matcher.py, hr_extractor.py,
                 module3_bayesian.py, module4_power_audit.py,
                 scorecard.py, validation.py
                 article_classifier.py  (legacy - superseded by publication_family)
tests/
data/            gold_standard/  logs/  outputs/
```
