from __future__ import annotations

import pandas as pd

from src.pipeline.validation import (
    build_gold_standard_template,
    build_inter_rater_template,
    compute_ai_calibration,
    compute_inter_rater_reliability,
    select_spot_check_pairs,
)


def _decision_log_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "NCT001_111",
                "routing": "auto_concordant",
                "llm_switch_type": "",
                "llm_confidence": "",
                "llm_flag": "",
                "human_reviewed": "yes",
                "human_final_class": "concordant",
                "human_poolable": "True",
                "registered_endpoint": "All-cause mortality at 12 months",
                "published_endpoint": "All-cause mortality at 12 months",
            },
            {
                "pair_id": "NCT002_222",
                "routing": "llm",
                "llm_switch_type": "moderate_switch",
                "llm_confidence": "low",
                "llm_flag": "True",
                "human_reviewed": "yes",
                "human_final_class": "moderate_switch",
                "human_poolable": "False",
                "registered_endpoint": "Composite CV death or HF admission",
                "published_endpoint": "CV death only at 12 months",
            },
            {
                "pair_id": "NCT003_333",
                "routing": "auto_major_switch",
                "llm_switch_type": "",
                "llm_confidence": "",
                "llm_flag": "",
                "human_reviewed": "spot_check",
                "human_final_class": "major_switch",
                "human_poolable": "False",
                "registered_endpoint": "Renal composite",
                "published_endpoint": "Quality of life score",
            },
        ]
    )


def test_compute_ai_calibration_returns_expected_metrics() -> None:
    decision_log = _decision_log_frame()
    gold = pd.DataFrame(
        {
            "pair_id": ["NCT001_111", "NCT002_222", "NCT003_333"],
            "gold_switch_type": ["concordant", "moderate_switch", "major_switch"],
        }
    )

    metrics = compute_ai_calibration(decision_log, gold)

    assert metrics["n_gold_standard_pairs"] == 3
    assert metrics["auc"] == 1.0
    assert metrics["precision_moderate_or_above"] == 1.0
    assert metrics["recall_moderate_or_above"] == 1.0
    assert metrics["low_confidence_flag_rate"] == 1.0
    assert metrics["recommended_actions"] == []


def test_compute_inter_rater_reliability_uses_completed_second_reviews() -> None:
    decision_log = _decision_log_frame()
    second_review = pd.DataFrame(
        {
            "pair_id": ["NCT001_111", "NCT002_222", "NCT003_333"],
            "second_switch_type": ["concordant", "moderate_switch", "major_switch"],
            "second_poolable": ["True", "False", "False"],
        }
    )

    metrics = compute_inter_rater_reliability(decision_log, second_review)

    assert metrics["n_inter_rater_pairs"] == 3
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["classification_agreement_rate_pct"] == 100.0
    assert metrics["poolable_agreement_rate_pct"] == 100.0


def test_templates_are_blinded_and_include_expected_columns() -> None:
    decision_log = _decision_log_frame()

    gold_template = build_gold_standard_template(decision_log, sample_size=2)
    inter_rater_template = build_inter_rater_template(decision_log, sample_rate=0.5)

    assert "gold_switch_type" in gold_template.columns
    assert "llm_switch_type" not in gold_template.columns
    assert len(gold_template) == 2

    assert "second_switch_type" in inter_rater_template.columns
    assert "human_final_class" not in inter_rater_template.columns
    assert len(inter_rater_template) >= 1


def test_select_spot_check_pairs_samples_only_auto_routed_rows() -> None:
    decision_log = _decision_log_frame()
    pair_ids = select_spot_check_pairs(decision_log, rate=1.0)

    assert pair_ids == {"NCT001_111", "NCT003_333"}
