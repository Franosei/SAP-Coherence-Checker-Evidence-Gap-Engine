from __future__ import annotations

import pandas as pd

from src.pipeline.scorecard import build_switching_summary, cluster_endpoints


def _decision_log(tmp_path) -> str:
    frame = pd.DataFrame(
        [
            {
                "pair_id": "NCT001_1",
                "registered_endpoint": "Overall survival at 5 years",
                "published_endpoint": "Overall survival",
                "llm_switch_type": "concordant",
                "llm_results_driven": "False",
                "llm_disclosed_exploratory": "False",
                "llm_switch_forms": "",
                "human_reviewed": "auto_accepted",
                "human_final_class": "",
                "human_poolable": "True",
            },
            {
                "pair_id": "NCT002_2",
                "registered_endpoint": "Progression-free survival",
                "published_endpoint": "Objective response rate",
                "llm_switch_type": "major_switch",
                "llm_results_driven": "True",
                "llm_disclosed_exploratory": "False",
                "llm_switch_forms": "endpoint_replaced | timeframe_changed",
                "human_reviewed": "yes",
                "human_final_class": "major_switch",
                "human_poolable": "False",
            },
            {
                "pair_id": "NCT003_3",
                "registered_endpoint": "Progression-free survival at 12 months",
                "published_endpoint": "PFS at 18 months",
                "llm_switch_type": "moderate_switch",
                "llm_results_driven": "False",
                "llm_disclosed_exploratory": "False",
                "llm_switch_forms": "timeframe_changed",
                "human_reviewed": "yes",
                "human_final_class": "minor_modification",
                "human_poolable": "True",
            },
        ]
    )
    path = tmp_path / "decision_log.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_switching_summary_overall_and_per_cluster(tmp_path) -> None:
    path = _decision_log(tmp_path)
    dl = pd.read_csv(path, dtype=str, keep_default_na=False)
    clusters = cluster_endpoints(dl)

    summary = build_switching_summary(clusters, decision_log_path=path)
    overall = summary[summary["scope"] == "overall"].iloc[0]

    # NCT002 is a confirmed switch; NCT003 was downgraded by the human to a
    # modification, so it is NOT counted.
    assert overall["n_outcome_switch"] == 1
    assert overall["n_human_confirmed_switch"] == 1
    assert overall["n_results_driven"] == 1

    # AI said moderate_switch, human said minor_modification, and AI said
    # major_switch, human agreed → 1/2 agreement.
    assert overall["ai_human_agreement_pct"] == 50.0

    # Per-form breakdown present.
    assert any(summary["scope"] == "form:timeframe_changed")


def test_switching_summary_empty_log(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["pair_id"]).to_csv(path, index=False)
    assert build_switching_summary({}, decision_log_path=str(path)).empty
