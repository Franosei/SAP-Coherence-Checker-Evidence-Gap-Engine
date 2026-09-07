"""
Create the blinded gold-standard review template.

Draws a deterministic, endpoint-stratified sample from the decision log and
writes it to ``data/gold_standard/gold_standard.csv``. The output contains
ONLY the raw endpoint strings and pair identifiers needed for blinded manual
review — the model's ``llm_switch_type`` / ``llm_reasoning`` fields are
intentionally excluded so the reviewer cannot see the answer first.

Output columns
--------------
pair_id / nct_id / pmid   identifiers
registered_endpoint       pre-specified endpoint text (CT.gov protocol section)
published_endpoint        endpoint text extracted from the PubMed abstract
gold_switch_type          FILL IN: concordant | additional_outcome |
                                   minor_modification | moderate_switch |
                                   major_switch
gold_direction            FILL IN (optional): none | promotion_of_secondary |
                                   composite_modified | timeframe_changed |
                                   endpoint_replaced | ...
notes                     FILL IN (optional): free-text reviewer notes

Once every row has a ``gold_switch_type``, the completed file feeds
``validation.compute_ai_calibration()`` (and the dashboard's AI Calibration
tab) for AUC / precision / recall against the pipeline's classifications.

Usage
-----
    python create_gold_standard.py                # default 20 pairs
    python create_gold_standard.py --sample 30    # custom sample size
    python create_gold_standard.py --overwrite    # replace a completed file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.decision_log import DecisionLog  # noqa: E402  (after sys.path bootstrap)
from src.pipeline.validation import build_gold_standard_template  # noqa: E402

GOLD_STANDARD_PATH = ROOT / "data" / "gold_standard" / "gold_standard.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the blinded gold-standard review template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of pairs to include (default: 20).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing gold_standard.csv without prompting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if GOLD_STANDARD_PATH.exists() and not args.overwrite:
        existing = pd.read_csv(GOLD_STANDARD_PATH, dtype=str, keep_default_na=False)
        completed = existing.get("gold_switch_type", pd.Series(dtype=str)).ne("").sum()
        if completed:
            print(
                f"\n{GOLD_STANDARD_PATH} already has {completed}/{len(existing)} rows filled in.\n"
                "Re-run with --overwrite to regenerate and lose those annotations.\n"
            )
            sys.exit(0)

    dl = DecisionLog().read()
    if dl.empty:
        sys.exit("Decision log is empty — run the pipeline through Module 2 first.")
    if dl["registered_endpoint"].eq("").all():
        sys.exit("Decision log has no registered endpoint text — Module 2 did not complete.")

    template = build_gold_standard_template(dl, sample_size=args.sample)
    if template.empty:
        sys.exit("No pairs available to sample.")

    GOLD_STANDARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(GOLD_STANDARD_PATH, index=False)

    print(f"\nWrote {len(template)} blinded pairs to {GOLD_STANDARD_PATH}\n")
    print(
        "Next: fill gold_switch_type for every row from\n"
        "  concordant | additional_outcome | minor_modification | "
        "moderate_switch | major_switch\n"
        "then re-run the AI Calibration tab (or compute_ai_calibration())."
    )


if __name__ == "__main__":
    main()
