# SAP Coherence Checker & Evidence Gap Engine v2.0

Human-in-the-loop pipeline for linking HFrEF Phase 2/3 RCTs to their peer-reviewed publications, detecting registered-versus-published endpoint switching, and surfacing evidence gaps — with a fully auditable review trail designed for publication in BMJ Open / Trials / Statistics in Medicine.

---

## Architecture

### Module 1 — Trial fetch & publication linkage (`module1_linker.py`)

Queries ClinicalTrials.gov REST API v2 for HFrEF Phase 2/3 interventional RCTs, then links each trial to its peer-reviewed publication via a four-stage NCT-to-PMID cascade.

**HFrEF population filter** — A regex classifier reads each trial's eligibility criteria text and classifies it as `hfref_confirmed`, `ambiguous`, `mixed_flag`, or `hfpef_excluded`. HFpEF and mixed trials are removed before any downstream analysis. Ambiguous trials are flagged for human review.

**NCT-to-PMID linkage cascade:**

| Stage | Method | Confidence |
|-------|--------|------------|
| 0 | CT.gov investigator-submitted RESULT PMID | High |
| 1 | Direct NCT ID search in PubMed (`[si]` field tag) | High |
| 2 | Title fuzzy matching — Jaccard token similarity ≥ 0.70 | High |
| 2b | Jaccard ≥ 0.50 (medium band) → Stage 3 | Medium → Stage 3 |
| 3 | First-author surname + publication year disambiguation | Medium / Low |

Every decision at every stage is written to the linkage audit log with the stage, method, confidence level, and a free-text resolution note.

---

### Article-type gate (`article_classifier.py`)

Before any candidate PubMed record enters the endpoint matching pipeline, it passes through a **two-tier agentic article-type gate**. This gate determines whether the linked paper is a *primary results paper* — the kind of publication that reports the trial's main efficacy outcome — and rejects protocol papers, design papers, subgroup analyses, systematic reviews, editorials, pharmacokinetic reports, and safety reports.

#### Why this matters

Without the gate, the linkage cascade can return a PubMed record for a paper that happens to mention the NCT identifier (e.g., a protocol paper published before the trial completed, or a secondary analysis). Comparing a registered endpoint against a protocol paper's methods section is scientifically meaningless and creates unnecessary manual review burden.

#### Tier 1 — Heuristic classifier

The heuristic classifier assigns an integer `reject_score` and `accept_score` to each PubMed record using five independent signal categories. A record is accepted or rejected with high confidence when the two scores differ by ≥ 3 points.

| Signal category | How it is checked | Direction |
|---|---|---|
| PubMed `PublicationType` tags | Exact match against accept/reject frozensets | Both |
| Title keywords | Regex against `_REJECT_TITLE_PATTERNS` (e.g., "protocol", "rationale", "design", "systematic review") | Reject |
| MeSH headings | Match against `_REJECT_MESH_TERMS` | Reject |
| Abstract section labels | Look for "Results" / "Findings" structured headings in `abstract_sections` | Accept |
| Abstract text patterns | Detect outcome reporting language ("reduced the risk", "hazard ratio", "primary endpoint was met") or protocol language ("will be randomized", "aims to evaluate") | Both |

Accept signals include PubMed types such as `Clinical Trial`, `Randomized Controlled Trial`, `Multicenter Study`, and the presence of a structured Results section or outcome-reporting language in the abstract.

Reject signals include PubMed types such as `Clinical Trial Protocol`, `Systematic Review`, `Meta-Analysis`, `Editorial`, `Letter`, title patterns like *"a randomised protocol"* or *"study design"*, and abstract language like *"will recruit"* or *"aims to assess"*.

#### Tier 2 — LLM arbiter

When the heuristic scores are too close to call (UNCERTAIN verdict, i.e., score difference < 3), the record is escalated to the LLM arbiter. The arbiter is called via the OpenAI API in JSON mode using `LLM_MODEL_PRIMARY`. It receives the PMID, title, journal, publication year, PubMed types, and a 600-character abstract snippet, then returns a structured verdict with `article_type`, `verdict`, `confidence`, and `reason`.

The LLM arbiter is called for roughly 15–25% of records. High-confidence heuristic verdicts (≥ 80% of records) never incur an LLM API call.

#### Gate integration in the linkage cascade

The gate is integrated at a single point — `_fetch_and_gate(pmid, nct_id, client)` — called instead of `client.fetch_record()` at every stage of the cascade:

- **High-confidence REJECT** → returns `(None, gate_note)`. The cascade skips this PMID and continues to the next stage or candidate.
- **Medium-confidence REJECT or UNCERTAIN** → returns `(record, gate_note)`. The record is accepted but the linkage confidence is capped at Low and the pair is flagged for human review in the dashboard.
- **ACCEPT** → returns `(record, gate_note)`. Normal cascade flow continues.

Every gate verdict is appended to the `linkage_notes` field of the linkage audit log entry, so reviewers can see exactly which gate tier fired and why.

---

### Module 2 — Endpoint matching (`module2_endpoint_matcher.py`)

Compares each trial's pre-specified registered endpoint (from CT.gov protocol section) against its published endpoint (extracted from the PubMed abstract Results section). Uses OpenAI `text-embedding-3-small` for Layer 1 cosine similarity and an OpenAI LLM 5-step reasoning chain (JSON mode) for Layer 2 adjudication of ambiguous pairs.

**Embeddings** — All unique endpoint texts are collected in a single pass, deduplicated, and sent to the OpenAI Embeddings API in batches of 64. OpenAI returns unit vectors, so cosine similarity reduces to a dot product.

**Routing thresholds:**

| Similarity score | Routing |
|---|---|
| ≥ 0.90 | Auto concordant — no LLM call |
| 0.50 – 0.89 | LLM adjudication |
| < 0.50 | Auto major switch — no LLM call |

---

### Module 3 — Bayesian evidence accumulation (`module3_bayesian.py`)

Sequential Bayesian random-effects meta-analysis (PyMC) operating exclusively on human-confirmed poolable trial pairs. Trials enter the model in registration-date order, simulating prospective evidence accumulation.

---

### Module 4 — Power audit (`module4_power_audit.py`)

Extracts hazard ratios and 95% confidence intervals from PubMed abstracts using six named regex patterns covering NEJM, JAMA, Lancet, and ESC reporting styles. Back-calculates assumed versus evidence-informed effect sizes and flags implausibly optimistic assumptions.

---

### Dashboard (`src/dashboard/app.py`)

Shiny for Python (v1.6) audit dashboard with nine tabs:

| Tab | Purpose |
|---|---|
| Overview | KPIs, routing breakdown, AI calibration overview, runtime settings |
| Linkage Review | Manual linkage queue for Low/Unlinked trials |
| Review Queue | Human-in-the-loop review of LLM-adjudicated pairs |
| Decision Log | Full audit log with filter, sort, and export |
| AI Calibration | Override rate, confusion matrix, agreement statistics |
| Evidence Scorecard | Endpoint concordance rates by trial and drug class |
| Power Audit | Effect-size back-calculation visualisation |
| Provenance | Per-pair decision trail with full linkage and gate notes |
| Export | Download full audit package as a zip file |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
NCBI_API_KEY=...          # optional — raises PubMed rate limit from 3 to 10 req/s
LLM_PROVIDER=openai
```

The application loads `.env` automatically from `src/pipeline/config.py`. All components (pipeline, tests, dashboard) share the same settings.

---

## Running the pipeline

Full run (fetches trials, links to PubMed, matches endpoints, extracts HRs, runs Bayesian module):

```bash
python run_pipeline.py --fresh-run --max-trials 20
```

Resume from a checkpoint (skips already-completed steps):

```bash
python run_pipeline.py                        # resumes from last checkpoint
python run_pipeline.py --skip-linkage         # skip Module 1 linkage
python run_pipeline.py --skip-matching        # skip Module 2 matching
python run_pipeline.py --skip-bayesian        # skip Module 3 Bayesian
```

Direct module use:

```python
from src.pipeline.module1_linker import fetch_hfref_trials, link_to_pubmed
from src.pipeline.module2_endpoint_matcher import run_endpoint_matching

trials = fetch_hfref_trials(max_records=20)
linked = link_to_pubmed(trials)
matched = run_endpoint_matching(linked)
```

---

## Running the dashboard

```bash
python -m shiny run --reload --port 8080 src/dashboard/app.py
```

Windows convenience launcher:

```powershell
.\run_dashboard.ps1
```

---

## Running tests

```bash
python -m pytest
python -m pytest --cov=src --cov-report=term-missing
```

---

## Project structure

```
src/
  dashboard/
    app.py               Shiny dashboard (9 tabs)
    helpers.py           Shared UI helpers and reactive utilities
    www/style.css        Dashboard stylesheet
  models/
    decision_log.py      Append-only decision log with portalocker
    linkage_log.py       Linkage audit log with confidence summary
    schemas.py           Pydantic v2 schemas for all log entries
  pipeline/
    article_classifier.py  Two-tier agentic article-type gate
    config.py            Single source of truth for all settings
    hr_extractor.py      HR/CI regex extraction from abstracts
    module1_linker.py    CT.gov fetch + NCT-to-PMID linkage cascade
    module2_endpoint_matcher.py  Embedding + LLM endpoint matching
    module3_bayesian.py  Sequential Bayesian random-effects meta-analysis
    module4_power_audit.py  Power/effect-size audit
    pubmed_client.py     PubMed E-utilities client with rate limiting
    scorecard.py         Evidence scorecard builder
    validation.py        Spot-check sampling and schema validation

data/
  gold_standard/
  logs/

tests/
run_pipeline.py          CLI entry point with checkpoint resumption
run_dashboard.ps1        Windows dashboard launcher
```

---

## Governance principles

- Every AI decision is logged with the routing, classification, confidence, and full reasoning chain.
- Human review is structural, not optional, for Low-confidence links and LLM-adjudicated endpoint pairs.
- The article-type gate prevents non-results papers from reaching the endpoint matching step, reducing spurious manual review.
- The linkage audit log records every cascade stage outcome — including gate verdicts — with a timestamp and pipeline version.
- Gold-standard and inter-rater reliability templates are generated from the dashboard Export tab.
- Reviewer actions are written back to the audit log with initials, timestamps, and override reasons.
- The Bayesian module operates only on human-confirmed poolable pairs — AI verdicts alone never enter the evidence pool.
