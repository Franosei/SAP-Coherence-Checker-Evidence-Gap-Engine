"""
SAP Coherence Checker — pipeline runner.

Executes the full four-module pipeline in sequence, with checkpoint resumption
at each stage so that a partial run can be continued without re-fetching data
or re-spending LLM API credits.

Pipeline flow
-------------
Step 1 — ClinicalTrials.gov fetch  (Module 1, Step A)
    Query CT.gov for all HFrEF Phase 2/3 RCTs with posted results.
    Output: ``data/outputs/trials.csv``

Step 2 — NCT-to-PMID linkage  (Module 1, Step B)
    Link each registered trial to its published journal paper via the
    3-stage PubMed linkage cascade.  Low-confidence links are flagged for
    human review in the dashboard before downstream analysis proceeds.
    Output: ``data/outputs/linked_trials.csv``
             ``data/logs/linkage_audit_log.csv``

Step 3 — Endpoint coherence analysis  (Module 2)
    Compare each registered endpoint (CT.gov protocol section) to its
    published counterpart (PubMed abstract) using embedding similarity +
    LLM 5-step reasoning.  Flagged pairs land in the dashboard review queue.
    Output: ``data/outputs/matched_trials.csv``
             ``data/logs/decision_log.csv``

Step 4 — HR extraction  (hr_extractor)
    Extract Hazard Ratio and 95% CI from PubMed abstracts for all High/Medium-
    confidence linked trials.  Failed extractions are flagged for manual entry.
    Output: ``data/logs/power_audit_log.csv``

Step 5 — Bayesian sequential meta-analysis  (Module 3)
    Fit a random-effects Bayesian model on human-confirmed poolable trial pairs
    from the decision log.  Model runs sequentially (one trial at a time) to
    show when the evidence crossed clinical significance.
    Output: ``data/logs/bayes_traces/trace_*.nc``

Usage
-----
    python run_pipeline.py                       # full run
    python run_pipeline.py --max-trials 20       # smoke test (20 trials)
    python run_pipeline.py --fresh-run           # wipe previous outputs, restart
    python run_pipeline.py --skip-linkage        # re-use existing linked_trials.csv
    python run_pipeline.py --skip-matching       # stop after linkage (no LLM)
    python run_pipeline.py --skip-bayesian       # stop before Bayesian model
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from src.pipeline.config import (
    DECISION_LOG_PATH,
    LINKAGE_LOG_PATH,
    OUTPUTS_DIR,
    POWER_AUDIT_LOG_PATH,
)
from src.pipeline.hr_extractor import extract_effect_measures
from src.pipeline.module1_linker import fetch_hfref_trials, link_to_pubmed
from src.pipeline.module2_endpoint_matcher import run_endpoint_matching
from src.pipeline.module4_power_audit import run_power_audit
from src.pipeline.scorecard import build_scorecard, cluster_endpoints

OUTPUT_DIR           = OUTPUTS_DIR
TRIALS_PATH          = OUTPUT_DIR / "trials.csv"
LINKED_TRIALS_PATH   = OUTPUT_DIR / "linked_trials.csv"
MATCHED_TRIALS_PATH  = OUTPUT_DIR / "matched_trials.csv"
EFFECT_MEASURES_PATH = OUTPUT_DIR / "effect_measures.json"
SCORECARD_PATH       = OUTPUT_DIR / "scorecard.csv"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAP Coherence Checker — full pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-trials", type=int, default=0,
        help="Limit number of trials fetched from CT.gov (0 = no limit).",
    )
    parser.add_argument(
        "--fresh-run", action="store_true",
        help="Delete all previous outputs and restart from scratch.",
    )
    parser.add_argument(
        "--skip-linkage", action="store_true",
        help="Skip Step 2 — reuse existing linked_trials.csv.",
    )
    parser.add_argument(
        "--skip-matching", action="store_true",
        help="Stop after Step 2 linkage (no LLM calls).",
    )
    parser.add_argument(
        "--skip-bayesian", action="store_true",
        help="Stop after Step 4 HR extraction (no Bayesian model).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("data/pipeline_run.log", mode="a", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------

def reset_outputs() -> None:
    """Delete all pipeline outputs so the run starts from a clean state."""
    targets = [
        TRIALS_PATH,
        LINKED_TRIALS_PATH,
        MATCHED_TRIALS_PATH,
        EFFECT_MEASURES_PATH,
        DECISION_LOG_PATH,
        DECISION_LOG_PATH.with_suffix(f"{DECISION_LOG_PATH.suffix}.lock"),
        LINKAGE_LOG_PATH,
        LINKAGE_LOG_PATH.with_suffix(f"{LINKAGE_LOG_PATH.suffix}.lock"),
        POWER_AUDIT_LOG_PATH,
    ]
    for path in targets:
        if path.exists():
            path.unlink()
            logging.info("Removed: %s", path)


def _already_matched() -> set[str]:
    """
    Return NCT IDs already present in the decision log.

    Used to resume a partially completed Step 3 without re-processing trials
    that have already been through the endpoint matching module.
    """
    if not DECISION_LOG_PATH.exists():
        return set()
    try:
        df = pd.read_csv(DECISION_LOG_PATH, usecols=["pair_id"], dtype=str)
        return {row.split("_")[0] for row in df["pair_id"].dropna()}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_fetch_trials(max_records: int | None, fresh_run: bool) -> pd.DataFrame:
    """
    Step 1 — Fetch HFrEF trials from ClinicalTrials.gov.

    Reuses ``trials.csv`` if it exists and ``--fresh-run`` is not set,
    allowing the pipeline to resume without an additional CT.gov query.
    """
    if TRIALS_PATH.exists() and not fresh_run:
        logging.info("Step 1 — Reusing existing %s", TRIALS_PATH)
        trials = pd.read_csv(TRIALS_PATH, dtype=str, keep_default_na=False)
        logging.info("  Loaded %d trials from file.", len(trials))
        return trials

    logging.info("Step 1 — Fetching trials from ClinicalTrials.gov...")
    trials = fetch_hfref_trials(max_records=max_records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(TRIALS_PATH, index=False)
    logging.info("  Saved %d trials to %s", len(trials), TRIALS_PATH)
    return trials


def step2_link_to_pubmed(trials: pd.DataFrame, skip: bool, fresh_run: bool) -> pd.DataFrame:
    """
    Step 2 — NCT-to-PMID linkage cascade.

    Reuses ``linked_trials.csv`` if it exists, ``--skip-linkage`` is set,
    and ``--fresh-run`` is not set.  This preserves previously completed
    linkage work and avoids redundant PubMed API calls.

    Low-confidence and unlinked trials are flagged in the linkage audit log;
    the human reviewer must resolve them in the dashboard before those trials
    enter the endpoint matching step.
    """
    if skip and LINKED_TRIALS_PATH.exists() and not fresh_run:
        logging.info("Step 2 — Reusing existing %s (--skip-linkage)", LINKED_TRIALS_PATH)
        linked = pd.read_csv(LINKED_TRIALS_PATH, dtype=str, keep_default_na=False)
        logging.info("  Loaded %d linked trials from file.", len(linked))
        return linked

    logging.info("Step 2 — Running NCT-to-PMID linkage cascade...")
    linked = link_to_pubmed(trials)
    linked.to_csv(LINKED_TRIALS_PATH, index=False)
    logging.info(
        "  Linkage complete. Saved %d rows to %s", len(linked), LINKED_TRIALS_PATH
    )

    # Governance summary
    n_high     = (linked["linkage_confidence"] == "High").sum()
    n_medium   = (linked["linkage_confidence"] == "Medium").sum()
    n_low      = (linked["linkage_confidence"] == "Low").sum()
    n_unlinked = (linked["linkage_confidence"] == "Unlinked").sum()
    n_flagged  = (linked.get("linkage_flag", pd.Series()) != "").sum()

    logging.info(
        "  Linkage distribution — High: %d | Medium: %d | Low: %d | "
        "Unlinked: %d | Flagged for human review: %d",
        n_high, n_medium, n_low, n_unlinked, n_flagged,
    )
    if n_flagged:
        logging.warning(
            "  %d trials flagged for human review. Open the dashboard "
            "(run_dashboard.ps1) and clear the Linkage Review queue before "
            "running endpoint matching on those trials.",
            n_flagged,
        )
    return linked


def step3_endpoint_matching(linked: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 — Endpoint coherence analysis (Module 2).

    Compares each trial's registered primary endpoint (from CT.gov protocol)
    to its published endpoint (from PubMed abstract) using:
      Layer 1 — embedding similarity (fast screen)
      Layer 2 — LLM 5-step reasoning for the ambiguous zone
      Layer 3 — human review gate (flagged pairs held in dashboard queue)

    Supports checkpoint resumption: trials already present in the decision log
    are skipped so that an interrupted run can continue.
    """
    already_done = _already_matched()
    if already_done:
        todo = linked[~linked["nct_id"].isin(already_done)].copy()
        logging.info(
            "Step 3 — Checkpoint resume: %d already matched, %d remaining.",
            len(already_done), len(todo),
        )
    else:
        todo = linked.copy()
        logging.info("Step 3 — Running endpoint coherence analysis on %d trials...", len(todo))

    if todo.empty:
        logging.info("  All trials already processed — loading existing results.")
        if MATCHED_TRIALS_PATH.exists():
            return pd.read_csv(MATCHED_TRIALS_PATH, dtype=str, keep_default_na=False)
        return linked

    # module2 now reads `published_endpoint` (from PubMed) rather than
    # `ctgov_reported_outcome` — the correct cross-source comparison.
    matched_subset = run_endpoint_matching(todo)
    update_cols = [col for col in ["nct_id", "pmid", "pair_id", "similarity_score", "routing"] if col in matched_subset.columns]
    updates = matched_subset[update_cols].drop_duplicates(subset=["nct_id", "pmid"])

    if MATCHED_TRIALS_PATH.exists():
        existing = pd.read_csv(MATCHED_TRIALS_PATH, dtype=str, keep_default_na=False)
        existing_updates = existing[[col for col in update_cols if col in existing.columns]]
        existing_updates = existing_updates.drop_duplicates(subset=["nct_id", "pmid"])
        updates = pd.concat([existing_updates, updates], ignore_index=True)
        updates = updates.drop_duplicates(subset=["nct_id", "pmid"], keep="last")

    matched = linked.drop(
        columns=[col for col in ["pair_id", "similarity_score", "routing"] if col in linked.columns],
        errors="ignore",
    ).merge(
        updates,
        on=["nct_id", "pmid"],
        how="left",
    )
    matched.to_csv(MATCHED_TRIALS_PATH, index=False)
    logging.info("  Matched output saved to %s", MATCHED_TRIALS_PATH)
    return matched


def step4_extract_hr(linked: pd.DataFrame) -> None:
    """
    Step 4 — HR/CI extraction from PubMed abstracts.

    Attempts regex-based extraction for all High/Medium-confidence linked
    trials.  Successful extractions are saved as JSON for Module 3.  Failed
    extractions are written to the power audit log and flagged for manual entry.

    Skips extraction if ``effect_measures.json`` already exists and
    ``--fresh-run`` is not set.
    """
    if EFFECT_MEASURES_PATH.exists():
        logging.info("Step 4 — Reusing existing %s", EFFECT_MEASURES_PATH)
        return

    logging.info("Step 4 — Extracting HR/CI from PubMed abstracts...")
    effect_measures = extract_effect_measures(linked)

    serialised = [em.model_dump() for em in effect_measures]
    EFFECT_MEASURES_PATH.write_text(
        json.dumps(serialised, indent=2, default=str), encoding="utf-8"
    )
    logging.info(
        "  Extracted %d effect measures. Saved to %s", len(effect_measures), EFFECT_MEASURES_PATH
    )

    n_manual = sum(1 for em in effect_measures if "rr_or_fallback" in em.extraction_method)
    if n_manual:
        logging.warning(
            "  %d trials used RR/OR fallback — these require manual verification "
            "before entering the Bayesian model.",
            n_manual,
        )


def step5_bayesian(skip: bool, trials: pd.DataFrame) -> list[dict]:
    """
    Step 5 — Bayesian sequential meta-analysis (Module 3) + power audit (Module 4).

    Loads EffectMeasure objects from ``effect_measures.json``, filters to
    human-confirmed poolable pairs, fits the random-effects Bayesian model
    sequentially (one trial at a time in registration-date order), then
    immediately runs the power audit so that optimism bias can be computed
    against the posterior available at each trial's registration date.

    Returns
    -------
    list[dict]
        Sequential analysis results (one dict per trial added). Empty list
        if skipped or no poolable pairs are available.
    """
    if skip:
        logging.info("Step 5 — Bayesian model skipped (--skip-bayesian).")
        return []

    if not EFFECT_MEASURES_PATH.exists():
        logging.warning("Step 5 — %s not found. Run Step 4 first.", EFFECT_MEASURES_PATH)
        return []

    if not DECISION_LOG_PATH.exists():
        logging.warning("Step 5 — Decision log not found. Complete Steps 3-4 first.")
        return []

    from src.models.schemas import EffectMeasure
    from src.pipeline.module3_bayesian import load_poolable_effects, run_sequential_analysis

    raw_data     = json.loads(EFFECT_MEASURES_PATH.read_text(encoding="utf-8"))
    all_measures = [EffectMeasure(**item) for item in raw_data]

    poolable_df = load_poolable_effects(all_measures)
    if poolable_df.empty:
        logging.info(
            "Step 5 — No human-confirmed poolable pairs yet. "
            "Clear the Review Queue in the dashboard, then re-run."
        )
        return []

    logging.info(
        "Step 5 — Fitting Bayesian sequential meta-analysis on %d poolable pairs...",
        len(poolable_df),
    )
    sequential_results = run_sequential_analysis(poolable_df)
    final = sequential_results[-1]
    logging.info(
        "  Final pooled HR = %.3f (95%% CrI %.3f-%.3f), tau = %.3f  [n=%d trials]",
        final["mu_mean"],
        final["mu_hdi_lower"],
        final["mu_hdi_upper"],
        final["tau_mean"],
        final["n_trials"],
    )

    # ---- Power audit (Module 4) ----------------------------------------
    # Runs immediately after Bayesian so sequential_results are in scope.
    # The power audit needs registration_date from the trials DataFrame and
    # the posterior at each trial's registration date from sequential_results.
    logging.info("Step 5b — Running power audit (Module 4)...")
    if not trials.empty:
        power_entries = run_power_audit(
            trials_df          = trials,
            sequential_results = sequential_results,
        )
        included = sum(1 for e in power_entries if e.excluded_reason is None)
        biased   = sum(
            1 for e in power_entries
            if e.optimism_bias is not None and e.optimism_bias < -0.05
        )
        logging.info(
            "  Power audit: %d included | %d excluded | %d with optimism bias > 0.05 HR units",
            included, len(power_entries) - included, biased,
        )
    else:
        logging.warning("  Power audit skipped — trials DataFrame is empty.")

    return sequential_results


def step6_scorecard(matched: pd.DataFrame, sequential_results: list[dict]) -> None:
    """
    Step 6 — Evidence gap scorecard (Module scorecard.py).

    Clusters the matched trial pairs by endpoint type, then builds one
    ScorecardEntry per cluster combining:
      - Human review decisions (from decision log)
      - Bayesian pooled HR and credible interval (from sequential analysis)
      - Correct I² using within-trial variance σ²_i = se_log_hr² (from effect measures)
      - Optimism bias (from power audit log)

    Output: ``data/outputs/scorecard.csv``
    """
    if matched.empty:
        logging.warning("Step 6 — No matched trials; scorecard not generated.")
        return

    if not DECISION_LOG_PATH.exists():
        logging.warning("Step 6 — Decision log missing; scorecard not generated.")
        return

    logging.info("Step 6 — Building evidence gap scorecard...")

    # Cluster pairs by endpoint keyword matching
    endpoint_clusters = cluster_endpoints(matched)
    if not endpoint_clusters:
        logging.warning("Step 6 — No endpoint clusters produced; scorecard not generated.")
        return

    # Build within-trial variance map {pair_id: se_log_hr^2} from effect measures
    within_trial_variances: dict[str, float] = {}
    if EFFECT_MEASURES_PATH.exists():
        from src.models.schemas import EffectMeasure
        raw = json.loads(EFFECT_MEASURES_PATH.read_text(encoding="utf-8"))
        for item in raw:
            em = EffectMeasure(**item)
            if em.se_log_hr and em.se_log_hr > 0:
                within_trial_variances[em.pair_id] = round(em.se_log_hr ** 2, 8)

    # Build Bayesian summaries per cluster
    # sequential_results is ordered by trial; associate the final result with
    # whichever cluster the last trial belongs to. For a richer mapping we
    # use the full-dataset final posterior for every cluster that has poolable
    # trials (conservative — per-cluster Bayesian fitting is out of scope here
    # but architecturally supported via bayesian_summaries parameter).
    bayesian_summaries: dict[str, dict] = {}
    if sequential_results:
        from src.pipeline.module3_bayesian import summarise_posterior
        final_idata = sequential_results[-1]["idata"]
        final_summary = summarise_posterior(final_idata)
        # Assign the full-dataset posterior to every cluster that has poolable pairs
        dl = pd.read_csv(DECISION_LOG_PATH, dtype=str, keep_default_na=False)
        poolable_ids = set(dl.loc[dl["human_poolable"].str.lower() == "true", "pair_id"])
        for cluster_name, pair_ids in endpoint_clusters.items():
            if any(pid in poolable_ids for pid in pair_ids):
                bayesian_summaries[cluster_name] = final_summary

    # Build power audit optimism bias map {nct_id: optimism_bias}
    power_audit_summary: dict[str, float] = {}
    if POWER_AUDIT_LOG_PATH.exists():
        pa = pd.read_csv(POWER_AUDIT_LOG_PATH, dtype=str, keep_default_na=False)
        for _, row in pa.iterrows():
            bias_raw = row.get("optimism_bias", "")
            try:
                power_audit_summary[row["nct_id"]] = float(bias_raw)
            except (ValueError, TypeError):
                pass

    scorecard_entries = build_scorecard(
        endpoint_clusters      = endpoint_clusters,
        bayesian_summaries     = bayesian_summaries if bayesian_summaries else None,
        power_audit_summary    = power_audit_summary if power_audit_summary else None,
        within_trial_variances = within_trial_variances if within_trial_variances else None,
    )

    if not scorecard_entries:
        logging.warning("Step 6 — Scorecard is empty.")
        return

    scorecard_df = pd.DataFrame([e.model_dump() for e in scorecard_entries])
    SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    scorecard_df.to_csv(SCORECARD_PATH, index=False)
    logging.info(
        "  Scorecard saved: %d clusters to %s", len(scorecard_entries), SCORECARD_PATH
    )
    for entry in scorecard_entries:
        logging.info(
            "  %-30s  included=%d  switch_rate=%.0f%%  strength=%s  HR=%s",
            entry.endpoint_cluster,
            entry.trials_included,
            entry.human_confirmed_switch_rate_pct,
            entry.evidence_strength.value,
            f"{entry.pooled_hr:.3f}" if entry.pooled_hr else "N/A",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()

    if args.fresh_run:
        reset_outputs()

    max_records: int | None = args.max_trials if args.max_trials > 0 else None

    # ------------------------------------------------------------------ #
    # Step 1 — Fetch trials from ClinicalTrials.gov                      #
    # ------------------------------------------------------------------ #
    trials = step1_fetch_trials(max_records, fresh_run=args.fresh_run)

    # ------------------------------------------------------------------ #
    # Step 2 — Link each trial to its PubMed publication                 #
    # ------------------------------------------------------------------ #
    linked = step2_link_to_pubmed(
        trials,
        skip      = args.skip_linkage,
        fresh_run = args.fresh_run,
    )

    if args.skip_matching:
        logging.info("--skip-matching set. Stopping after Step 2 (linkage).")
        return

    # ------------------------------------------------------------------ #
    # Step 3 — Endpoint coherence analysis                               #
    # ------------------------------------------------------------------ #
    step3_endpoint_matching(linked)

    # ------------------------------------------------------------------ #
    # Step 4 — HR / CI extraction from PubMed abstracts                 #
    # ------------------------------------------------------------------ #
    step4_extract_hr(linked)

    if args.skip_bayesian:
        logging.info("--skip-bayesian set. Stopping after Step 4 (HR extraction).")
        return

    # ------------------------------------------------------------------ #
    # Step 5 — Bayesian sequential meta-analysis + power audit           #
    # ------------------------------------------------------------------ #
    sequential_results = step5_bayesian(skip=False, trials=trials)

    # ------------------------------------------------------------------ #
    # Step 6 — Evidence gap scorecard                                    #
    # ------------------------------------------------------------------ #
    matched = (
        pd.read_csv(MATCHED_TRIALS_PATH, dtype=str, keep_default_na=False)
        if MATCHED_TRIALS_PATH.exists()
        else pd.DataFrame()
    )
    step6_scorecard(matched=matched, sequential_results=sequential_results)

    logging.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
