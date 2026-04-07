"""
Validation workflow helpers for gold-standard calibration and inter-rater review.

This module centralises the proposal's Section 5 workflow:

- build a 20-pair gold-standard template from raw endpoint strings
- compare AI classifications to that gold standard
- build a blinded 10% inter-rater review sample
- compute Cohen's kappa and agreement rates once a second reviewer completes it
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import pandas as pd
from sklearn.metrics import (  # type: ignore
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.pipeline.config import (
    SPOT_CHECK_RATE,
    VALIDATION_LLM_LOW_CONF_FLAG_RATE,
    VALIDATION_TARGET_AUC,
    VALIDATION_TARGET_PRECISION,
    VALIDATION_TARGET_RECALL,
)

_VALID_SWITCH_TYPES = {
    "concordant",
    "minor_modification",
    "moderate_switch",
    "major_switch",
}
_MODERATE_OR_ABOVE = {"moderate_switch", "major_switch"}
_REVIEWED_STATUSES = {"yes", "spot_check"}


def _stable_rank(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_sample(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if frame.empty or n_rows <= 0:
        return frame.iloc[0:0].copy()
    ranked = frame.copy()
    ranked["_stable_rank"] = ranked["pair_id"].map(_stable_rank)
    sampled = ranked.sort_values(["_stable_rank", "pair_id"]).head(n_rows)
    return sampled.drop(columns=["_stable_rank"]).reset_index(drop=True)


def _parse_pair_id(pair_id: str) -> tuple[str, str]:
    if "_" not in pair_id:
        return pair_id, ""
    nct_id, pmid = pair_id.split("_", 1)
    return nct_id, pmid


def derive_ai_switch_type(row: pd.Series) -> str:
    llm_switch = str(row.get("llm_switch_type", "")).strip()
    if llm_switch:
        return llm_switch

    routing = str(row.get("routing", "")).strip()
    if routing == "auto_concordant":
        return "concordant"
    if routing == "auto_major_switch":
        return "major_switch"
    return ""


def build_gold_standard_template(
    decision_log: pd.DataFrame,
    sample_size: int = 20,
) -> pd.DataFrame:
    """
    Build a deterministic, routing-stratified gold-standard review template.

    The output intentionally contains only raw endpoint strings and identifiers
    needed for blinded manual review. AI fields are omitted on purpose.
    """
    if decision_log.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "nct_id",
                "pmid",
                "registered_endpoint",
                "published_endpoint",
                "gold_switch_type",
                "gold_direction",
                "notes",
            ]
        )

    candidates = decision_log[
        decision_log["registered_endpoint"].fillna("").ne("")
    ][["pair_id", "registered_endpoint", "published_endpoint", "routing"]].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "nct_id",
                "pmid",
                "registered_endpoint",
                "published_endpoint",
                "gold_switch_type",
                "gold_direction",
                "notes",
            ]
        )

    groups = ["auto_concordant", "llm", "auto_major_switch"]
    quota = max(sample_size // len(groups), 1)
    selected_parts: list[pd.DataFrame] = []

    for routing in groups:
        group = candidates[candidates["routing"] == routing]
        if group.empty:
            continue
        selected_parts.append(_stable_sample(group, min(quota, len(group))))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.iloc[0:0].copy()
    if len(selected) > sample_size:
        selected = _stable_sample(selected, sample_size)

    if len(selected) < min(sample_size, len(candidates)):
        remaining = candidates[~candidates["pair_id"].isin(selected["pair_id"])]
        needed = min(sample_size, len(candidates)) - len(selected)
        selected = pd.concat([selected, _stable_sample(remaining, needed)], ignore_index=True)

    selected[["nct_id", "pmid"]] = selected["pair_id"].apply(_parse_pair_id).apply(pd.Series)
    selected["gold_switch_type"] = ""
    selected["gold_direction"] = ""
    selected["notes"] = ""
    return selected[
        [
            "pair_id",
            "nct_id",
            "pmid",
            "registered_endpoint",
            "published_endpoint",
            "gold_switch_type",
            "gold_direction",
            "notes",
        ]
    ].reset_index(drop=True)


def select_spot_check_pairs(
    decision_log: pd.DataFrame,
    rate: float = SPOT_CHECK_RATE,
) -> set[str]:
    """
    Deterministically select the proposal's 10% spot-check sample from auto-routed pairs.
    """
    if decision_log.empty or rate <= 0:
        return set()

    auto_rows = decision_log[
        decision_log["routing"].isin(["auto_concordant", "auto_major_switch"])
    ]["pair_id"].dropna()
    if auto_rows.empty:
        return set()

    ordered = sorted(auto_rows, key=_stable_rank)
    sample_size = min(len(ordered), int(len(ordered) * rate))
    if sample_size <= 0:
        return set()
    return set(ordered[:sample_size])


def build_inter_rater_template(
    decision_log: pd.DataFrame,
    sample_rate: float = SPOT_CHECK_RATE,
) -> pd.DataFrame:
    """
    Build a blinded second-reviewer template from human-reviewed pairs.
    """
    if decision_log.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "nct_id",
                "pmid",
                "registered_endpoint",
                "published_endpoint",
                "second_reviewer_initials",
                "second_switch_type",
                "second_poolable",
                "notes",
            ]
        )

    reviewed = decision_log[
        decision_log["human_reviewed"].isin(_REVIEWED_STATUSES)
    ][["pair_id", "registered_endpoint", "published_endpoint"]].copy()
    if reviewed.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "nct_id",
                "pmid",
                "registered_endpoint",
                "published_endpoint",
                "second_reviewer_initials",
                "second_switch_type",
                "second_poolable",
                "notes",
            ]
        )

    sample_size = min(len(reviewed), max(math.ceil(len(reviewed) * sample_rate), 1))
    selected = _stable_sample(reviewed, sample_size)
    selected[["nct_id", "pmid"]] = selected["pair_id"].apply(_parse_pair_id).apply(pd.Series)
    selected["second_reviewer_initials"] = ""
    selected["second_switch_type"] = ""
    selected["second_poolable"] = ""
    selected["notes"] = ""
    return selected[
        [
            "pair_id",
            "nct_id",
            "pmid",
            "registered_endpoint",
            "published_endpoint",
            "second_reviewer_initials",
            "second_switch_type",
            "second_poolable",
            "notes",
        ]
    ].reset_index(drop=True)


def compute_ai_calibration(
    decision_log: pd.DataFrame,
    gold_standard: pd.DataFrame,
) -> dict:
    """
    Compare AI classifications to the 20-pair gold standard from Section 5.2.
    """
    if decision_log.empty or gold_standard.empty:
        return {"error": "Decision log or gold standard is empty"}

    required_cols = {"pair_id", "gold_switch_type"}
    if not required_cols.issubset(gold_standard.columns):
        return {"error": "Gold standard is missing required columns"}

    gold = gold_standard.copy()
    gold["gold_switch_type"] = gold["gold_switch_type"].astype(str).str.strip()
    gold = gold[gold["gold_switch_type"].isin(_VALID_SWITCH_TYPES)].copy()

    merged = decision_log.merge(
        gold[["pair_id", "gold_switch_type"]],
        on="pair_id",
        how="inner",
    )
    if merged.empty:
        return {"error": "No matching pair_ids between decision log and gold standard"}

    merged["ai_switch_type"] = merged.apply(derive_ai_switch_type, axis=1)
    merged = merged[merged["ai_switch_type"].isin(_VALID_SWITCH_TYPES)].copy()
    if merged.empty:
        return {"error": "No gold-standard pairs have a usable AI classification"}

    merged["gold_binary"] = merged["gold_switch_type"].ne("concordant").astype(int)
    merged["ai_binary"] = merged["ai_switch_type"].ne("concordant").astype(int)
    merged["gold_moderate_or_above"] = merged["gold_switch_type"].isin(_MODERATE_OR_ABOVE).astype(int)
    merged["ai_moderate_or_above"] = merged["ai_switch_type"].isin(_MODERATE_OR_ABOVE).astype(int)

    auc = None
    if merged["gold_binary"].nunique() > 1:
        auc = roc_auc_score(merged["gold_binary"], merged["ai_binary"])

    precision = precision_score(
        merged["gold_moderate_or_above"],
        merged["ai_moderate_or_above"],
        zero_division=0,
    )
    recall = recall_score(
        merged["gold_moderate_or_above"],
        merged["ai_moderate_or_above"],
        zero_division=0,
    )

    low_confidence_rows = merged[merged["llm_confidence"].astype(str).str.lower() == "low"]
    low_confidence_flag_rate = None
    if not low_confidence_rows.empty:
        low_confidence_flag_rate = (
            (
                low_confidence_rows["llm_flag"].astype(str).str.lower().isin({"true", "yes", "1"})
                | low_confidence_rows["human_reviewed"].astype(str).str.lower().isin(_REVIEWED_STATUSES)
            ).mean()
        )

    agreement = (merged["ai_switch_type"] == merged["gold_switch_type"]).mean()

    recommended_actions: list[str] = []
    if auc is not None and auc < VALIDATION_TARGET_AUC:
        recommended_actions.append("Adjust similarity thresholds and re-evaluate the LLM prompt.")
    if precision < VALIDATION_TARGET_PRECISION:
        recommended_actions.append("Tighten the lower routing threshold from 0.50 toward 0.60.")
    if recall < VALIDATION_TARGET_RECALL:
        recommended_actions.append("Loosen the upper routing threshold from 0.90 toward 0.85.")
    if (
        low_confidence_flag_rate is not None
        and low_confidence_flag_rate < VALIDATION_LLM_LOW_CONF_FLAG_RATE
    ):
        recommended_actions.append("Escalate medium-confidence LLM cases to human review as well.")

    return {
        "n_gold_standard_pairs": len(merged),
        "auc": round(auc, 3) if auc is not None else None,
        "precision_moderate_or_above": round(float(precision), 3),
        "recall_moderate_or_above": round(float(recall), 3),
        "low_confidence_flag_rate": (
            round(float(low_confidence_flag_rate), 3)
            if low_confidence_flag_rate is not None
            else None
        ),
        "ai_human_agreement_rate_pct": round(float(agreement) * 100, 1),
        "auc_target_met": auc is None or auc >= VALIDATION_TARGET_AUC,
        "precision_target_met": precision >= VALIDATION_TARGET_PRECISION,
        "recall_target_met": recall >= VALIDATION_TARGET_RECALL,
        "low_confidence_target_met": (
            low_confidence_flag_rate is None
            or low_confidence_flag_rate >= VALIDATION_LLM_LOW_CONF_FLAG_RATE
        ),
        "recommended_actions": recommended_actions,
    }


def compute_inter_rater_reliability(
    decision_log: pd.DataFrame,
    second_review: pd.DataFrame,
) -> dict:
    """
    Compute Cohen's kappa for the blinded second-reviewer sample from Section 5.4.
    """
    if decision_log.empty or second_review.empty:
        return {"error": "Decision log or inter-rater review file is empty"}

    required_cols = {"pair_id", "second_switch_type"}
    if not required_cols.issubset(second_review.columns):
        return {"error": "Inter-rater review file is missing required columns"}

    second = second_review.copy()
    second["second_switch_type"] = second["second_switch_type"].astype(str).str.strip()
    second = second[second["second_switch_type"].isin(_VALID_SWITCH_TYPES)].copy()
    if second.empty:
        return {"error": "No completed second-reviewer classifications found"}

    merged = decision_log.merge(
        second[["pair_id", "second_switch_type", "second_poolable"]],
        on="pair_id",
        how="inner",
    )
    merged = merged[merged["human_final_class"].isin(_VALID_SWITCH_TYPES)].copy()
    if merged.empty:
        return {"error": "No overlapping reviewed pairs found for inter-rater analysis"}

    kappa = cohen_kappa_score(merged["human_final_class"], merged["second_switch_type"])
    class_agreement = (merged["human_final_class"] == merged["second_switch_type"]).mean()

    poolable_agreement = None
    if "second_poolable" in merged.columns:
        second_poolable = merged["second_poolable"].astype(str).str.lower().isin({"true", "yes", "1"})
        first_poolable = merged["human_poolable"].astype(str).str.lower().isin({"true", "yes", "1"})
        poolable_agreement = (first_poolable == second_poolable).mean()

    return {
        "n_inter_rater_pairs": len(merged),
        "cohen_kappa": round(float(kappa), 3),
        "classification_agreement_rate_pct": round(float(class_agreement) * 100, 1),
        "poolable_agreement_rate_pct": (
            round(float(poolable_agreement) * 100, 1)
            if poolable_agreement is not None
            else None
        ),
    }


def iter_review_actions(metrics: dict) -> Iterable[str]:
    """Yield non-empty follow-up actions from a validation metrics payload."""
    for action in metrics.get("recommended_actions", []):
        if action:
            yield action
