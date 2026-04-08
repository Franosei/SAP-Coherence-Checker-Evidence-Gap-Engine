"""
Create the 20-pair gold-standard review template.

This script draws a deterministic, routing-stratified sample from the
decision log and writes it to data/gold_standard/gold_standard.csv.

The output CSV contains ONLY the raw endpoint strings and pair identifiers
needed for blinded manual review.  AI fields (llm_switch_type, llm_reasoning,
etc.) are intentionally excluded so that the reviewer cannot see the model's
answer before giving their own classification.

Columns in the output
---------------------
pair_id               — unique identifier: {nct_id}_{pmid}
nct_id                — ClinicalTrials.gov identifier
pmid                  — PubMed identifier of the linked paper
registered_endpoint   — pre-specified endpoint text from CT.gov protocol section
published_endpoint    — extracted endpoint text from the PubMed abstract
gold_switch_type      — FILL IN: concordant | minor_modification |
                                  moderate_switch | major_switch
gold_direction        — FILL IN: none | promotion_of_secondary |
                                  composite_modified | timeframe_changed |
                                  endpoint_replaced
notes                 — FILL IN: free-text reviewer notes (optional)

Usage
-----
    python create_gold_standard.py               # default 20 pairs
    python create_gold_standard.py --sample 30   # custom sample size
    python create_gold_standard.py --overwrite   # replace existing file

After filling in gold_switch_type and gold_direction for every row,
the completed file feeds directly into the AI Calibration tab in the
dashboard and into compute_ai_calibration() for AUC / precision / recall
metrics (Section 5.2 of the proposal).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when run from any working directory
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.decision_log import DecisionLog
from src.pipeline.validation import build_gold_standard_template

GOLD_STANDARD_DIR  = ROOT / "data" / "gold_standard"
GOLD_STANDARD_PATH = GOLD_STANDARD_DIR / "gold_standard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 20-pair gold-standard review template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample", type=int, default=20,
        help="Number of pairs to include in the gold standard (default: 20).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing gold_standard.csv without prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Guard against overwriting a completed gold standard                 #
    # ------------------------------------------------------------------ #
    if GOLD_STANDARD_PATH.exists() and not args.overwrite:
        existing = __import__("pandas").read_csv(
            GOLD_STANDARD_PATH, dtype=str, keep_default_na=False
        )
        completed = existing["gold_switch_type"].ne("").sum()
        if completed > 0:
            print(
                f"\nGold standard already exists at:\n  {GOLD_STANDARD_PATH}\n"
                f"  {completed}/{len(existing)} rows already have gold_switch_type filled in.\n"
                "\nTo regenerate and LOSE those annotations, re-run with --overwrite.\n"
                "To keep the existing file, do nothing.\n"
            )
            sys.exit(0)

    # ------------------------------------------------------------------ #
    # Load decision log                                                   #
    # ------------------------------------------------------------------ #
    dl = DecisionLog().read()

    if dl.empty:
        print(
            "\nDecision log is empty.\n"
            "Run the pipeline first:\n"
            "  python run_pipeline.py --fresh-run --max-trials 20\n"
            "Then clear the Review Queue in the dashboard, and re-run this script.\n"
        )
        sys.exit(1)

    if dl["registered_endpoint"].eq("").all():
        print(
            "\nDecision log has no registered endpoint text.\n"
            "Ensure Module 2 (endpoint matching) completed successfully before "
            "generating the gold standard.\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Build template                                                      #
    # ------------------------------------------------------------------ #
    template = build_gold_standard_template(dl, sample_size=args.sample)

    if template.empty:
        print(
            "\nNo pairs available to sample.\n"
            "Check that the decision log contains rows with non-empty registered endpoints.\n"
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write output                                                        #
    # ------------------------------------------------------------------ #
    GOLD_STANDARD_DIR.mkdir(parents=True, exist_ok=True)
    template.to_csv(GOLD_STANDARD_PATH, index=False)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    routing_counts = dl.loc[
        dl["pair_id"].isin(template["pair_id"]), "routing"
    ].value_counts().to_dict()

    print(f"\nGold standard template written to:\n  {GOLD_STANDARD_PATH}")
    print(f"\n{len(template)} pairs sampled:")
    for routing, count in routing_counts.items():
        label = {
            "auto_concordant":    "Auto concordant",
            "llm":                "Model adjudicated",
            "auto_major_switch":  "Auto major switch",
        }.get(routing, routing)
        print(f"  {label:<25} {count} pair(s)")

    print(
        "\nNext steps:\n"
        "  1. Open the CSV in Excel or any spreadsheet editor.\n"
        "  2. For each row, read both endpoint columns carefully.\n"
        "  3. Fill in gold_switch_type from:\n"
        "       concordant | minor_modification | moderate_switch | major_switch\n"
        "  4. Fill in gold_direction from:\n"
        "       none | promotion_of_secondary | composite_modified |\n"
        "       timeframe_changed | endpoint_replaced\n"
        "  5. Add any reviewer notes in the notes column.\n"
        "  6. Save the file (keep it as gold_standard.csv in the same location).\n"
        "  7. The AI Calibration tab in the dashboard will pick it up automatically.\n"
    )


if __name__ == "__main__":
    main()
