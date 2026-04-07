"""
Module 2 - Hybrid endpoint matching with human oversight.

This module implements the proposal's three-layer architecture for comparing
the registered primary endpoint on ClinicalTrials.gov against the published
primary endpoint extracted from the linked PubMed/PMC record.

Layer 1  Embedding similarity
    Runs on every High/Medium-confidence linked trial-publication pair.
    Routing follows the proposal exactly:
      0.90-1.00  -> auto_concordant
      0.50-0.89  -> llm
      0.00-0.49  -> auto_major_switch

Layer 2  LLM clinical judge
    Runs only for the ambiguous zone (0.50-0.89). Responses are validated with
    Pydantic and malformed outputs are flagged for human review.

Layer 3  Human review gate
    Structural gate implemented via the decision log and dashboard workflow.
    Low-confidence or unlinked publication links are excluded upstream and
    therefore never enter this module until resolved.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import pandas as pd

from src.models.decision_log import DecisionLog
from src.models.schemas import (
    DecisionLogEntry,
    EndpointRouting,
    LLMEndpointClassification,
)
from src.pipeline.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_COST_PER_1K_USD,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_COST_CEILING_USD,
    LLM_COST_PER_1K_INPUT_USD,
    LLM_COST_PER_1K_OUTPUT_USD,
    LLM_MAX_TOKENS,
    LLM_MODEL_PRIMARY,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    SIMILARITY_AUTO_CONCORDANT,
    SIMILARITY_LLM_LOWER,
)

logger = logging.getLogger(__name__)


_LLM_SYSTEM_PROMPT = """\
You are a clinical biostatistician expert in randomised controlled trial design \
and endpoint reporting. Your task is to classify whether the primary endpoint \
registered in ClinicalTrials.gov matches the primary endpoint reported in the \
published paper.

Reason step by step before giving your final classification. \
Return ONLY a valid JSON object matching the schema below — no surrounding text, \
no markdown fences.

Schema:
{
  "switch_type": "concordant" | "minor_modification" | "moderate_switch" | "major_switch",
  "direction":   "none" | "promotion_of_secondary" | "composite_modified" |
                 "timeframe_changed" | "endpoint_replaced",
  "step_by_step_reasoning": "<mandatory — minimum 2-3 sentences of clinical reasoning>",
  "confidence":             "high" | "medium" | "low",
  "comparability_for_pooling": true | false,
  "flag_for_human_review":     true | false,
  "key_differences":           ["<specific difference 1>", "<specific difference 2>"]
}

Rules:
- step_by_step_reasoning is mandatory. Never omit it.
- Set flag_for_human_review=true whenever you are uncertain about any aspect.
- Set confidence="low" when endpoint text is ambiguous or composite components are unclear.
- comparability_for_pooling answers: are these two endpoints clinically equivalent enough to \
  combine in a meta-analysis of the same outcome?
"""

_LLM_USER_TEMPLATE = """\
Registered endpoint (ClinicalTrials.gov):
{registered}

Published endpoint (PubMed / PMC):
{published}

Classify the relationship between these two endpoints.
"""


def _normalise(text: object) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return "" if cleaned.lower() in {"", "none", "nan"} else cleaned


def _split_registered_candidates(text: str) -> list[str]:
    norm = _normalise(text)
    if not norm:
        return []
    parts = [_normalise(part) for part in re.split(r"\s*\|\s*|\n+", norm)]
    return [part for part in parts if part] or [norm]


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two pre-normalised embedding vectors.

    The OpenAI Embeddings API returns unit vectors (L2-normalised), so the
    cosine similarity is equal to the dot product — no magnitude division
    required.  Clamped to [0, 1] to guard against floating-point drift.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return max(0.0, min(1.0, dot))


def _embed_batch(texts: list[str], client: "openai.OpenAI") -> list[list[float]]:
    """
    Embed a batch of texts via the OpenAI Embeddings API.

    Sends texts in chunks of ``EMBEDDING_BATCH_SIZE`` to stay within the
    API's per-request limit.  Returns one embedding vector per input text,
    preserving order.

    Parameters
    ----------
    texts:
        Non-empty strings to embed.
    client:
        An authenticated ``openai.OpenAI`` instance.

    Returns
    -------
    list[list[float]]
        One 1 536-dimensional unit vector per input text.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        chunk = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        # Response items are ordered to match the input order.
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

        tokens_used = response.usage.total_tokens
        cost = tokens_used / 1_000 * EMBEDDING_COST_PER_1K_USD
        logger.debug(
            "Embedding batch [%d–%d]: %d tokens, $%.6f",
            start, start + len(chunk) - 1, tokens_used, cost,
        )
    return vectors


def _compute_similarity_scores(
    registered_endpoints: list[str],
    published_endpoints: list[str],
) -> list[float]:
    """
    Compute cosine similarity scores between registered and published endpoints.

    Uses the OpenAI Embeddings API (``text-embedding-3-small``).  All unique
    texts are embedded in a single batched call to minimise API
    round-trips.  Registered endpoints may contain multiple pipe-separated
    candidate measures from CT.gov; the best cosine score against the
    published endpoint is used for routing.

    Parameters
    ----------
    registered_endpoints:
        List of registered primary endpoint strings (one per trial).
    published_endpoints:
        Corresponding list of published primary endpoint strings from PubMed.

    Returns
    -------
    list[float]
        Cosine similarity score in [0.0, 1.0] for each trial pair.
        Returns 0.0 for pairs where either endpoint string is empty.
    """
    import openai  # type: ignore

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    client = openai.OpenAI(api_key=api_key, base_url=LLM_BASE_URL)

    # Collect every unique text that needs embedding in one pass so we can
    # batch all requests together rather than calling the API per-pair.
    pair_candidates: list[tuple[list[str], str]] = []
    all_texts: list[str] = []
    text_index: dict[str, int] = {}  # text → index in all_texts (deduplication)

    for registered, published in zip(registered_endpoints, published_endpoints):
        candidates   = _split_registered_candidates(registered)
        pub_text     = _normalise(published)
        pair_candidates.append((candidates, pub_text))

        for text in candidates + ([pub_text] if pub_text else []):
            if text and text not in text_index:
                text_index[text] = len(all_texts)
                all_texts.append(text)

    if not all_texts:
        return [0.0] * len(registered_endpoints)

    logger.info(
        "Layer 1 — embedding %d unique endpoint strings via %s...",
        len(all_texts), EMBEDDING_MODEL,
    )
    all_vectors = _embed_batch(all_texts, client)

    scores: list[float] = []
    for candidates, pub_text in pair_candidates:
        if not candidates or not pub_text:
            scores.append(0.0)
            continue

        pub_vec = all_vectors[text_index[pub_text]]
        best    = max(
            _cosine(all_vectors[text_index[c]], pub_vec)
            for c in candidates
            if c in text_index
        )
        scores.append(round(best, 4))

    return scores


def _route_from_score(score: float) -> EndpointRouting:
    if score >= SIMILARITY_AUTO_CONCORDANT:
        return EndpointRouting.AUTO_CONCORDANT
    if score >= SIMILARITY_LLM_LOWER:
        return EndpointRouting.LLM
    return EndpointRouting.AUTO_MAJOR_SWITCH


class _LLMCostTracker:
    """In-process LLM cost accumulator with a hard ceiling."""

    def __init__(self, ceiling: float = LLM_COST_CEILING_USD) -> None:
        self._ceiling = ceiling
        self._total_usd = 0.0
        self._total_calls = 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        cost = (
            prompt_tokens / 1_000 * LLM_COST_PER_1K_INPUT_USD
            + completion_tokens / 1_000 * LLM_COST_PER_1K_OUTPUT_USD
        )
        self._total_usd += cost
        self._total_calls += 1
        logger.debug(
            "LLM call #%d - %d input / %d output tokens - $%.6f this call - $%.4f total",
            self._total_calls,
            prompt_tokens,
            completion_tokens,
            cost,
            self._total_usd,
        )
        if self._total_usd > self._ceiling:
            raise RuntimeError(
                f"LLM cost ceiling of ${self._ceiling:.2f} exceeded "
                f"(running total: ${self._total_usd:.4f} after {self._total_calls} calls)."
            )

    @property
    def total_usd(self) -> float:
        return round(self._total_usd, 6)

    @property
    def total_calls(self) -> int:
        return self._total_calls


_cost_tracker = _LLMCostTracker()


def _call_llm(registered: str, published: str) -> Optional[LLMEndpointClassification]:
    """
    Submit one ambiguous endpoint pair to the configured OpenAI-compatible model.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    if api_key.lower().startswith("your_") or "api_key_here" in api_key.lower():
        logger.error(
            "OPENAI_API_KEY appears to still be a placeholder value. "
            "Replace it in .env to enable LLM adjudication."
        )
        return None

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    user_content = _LLM_USER_TEMPLATE.format(
        registered=_normalise(registered),
        published=_normalise(published),
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_PRIMARY,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
    except Exception as exc:
        logger.error("%s API error: %s", LLM_PROVIDER, exc)
        return None

    if response.usage:
        _cost_tracker.record(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    raw_text = (response.choices[0].message.content or "").strip()
    raw_text = re.sub(r"^```[a-z]*\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        parsed = json.loads(raw_text)
        return LLMEndpointClassification(**parsed)
    except Exception as exc:
        logger.warning(
            "Malformed LLM response for endpoint pair; flagging for human review. "
            "Error: %s | Raw response (first 400 chars): %s",
            exc,
            raw_text[:400],
        )
        return None


def run_endpoint_matching(linked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Layers 1-2 for every High/Medium-confidence linked trial-publication pair.

    Required columns
    ----------------
    nct_id, pmid, linkage_confidence, primary_outcomes, published_endpoint

    Notes
    -----
    Low-confidence and unlinked publication pairs are intentionally excluded
    here. They must be resolved in the linkage review workflow before they can
    enter endpoint matching or any downstream analysis.
    """
    decision_log = DecisionLog()

    processable = linked_df[
        linked_df["pmid"].notna()
        & ~linked_df["pmid"].isin(["", "None"])
        & linked_df["linkage_confidence"].isin(["High", "Medium"])
        & linked_df["primary_outcomes"].apply(_normalise).astype(bool)
    ].copy()

    skipped_due_to_linkage = len(linked_df) - len(processable)
    if skipped_due_to_linkage:
        logger.info(
            "Endpoint matching skipped %d row(s) because linkage was unresolved, "
            "low-confidence, or missing a registered endpoint.",
            skipped_due_to_linkage,
        )

    if processable.empty:
        logger.warning(
            "run_endpoint_matching: no processable linked rows after linkage gate. "
            "Resolve Low/Unlinked publication links before running Module 2."
        )
        return linked_df

    registered_endpoints = processable["primary_outcomes"].fillna("").tolist()
    published_endpoints = processable["published_endpoint"].fillna("").tolist()

    logger.info(
        "Layer 1 - computing embedding similarity for %d linked trial-publication pairs...",
        len(processable),
    )
    scores = _compute_similarity_scores(registered_endpoints, published_endpoints)

    pair_ids: list[str] = []
    routings: list[str] = []

    for idx, (_, row) in enumerate(processable.iterrows()):
        nct_id = str(row["nct_id"]).strip()
        pmid = str(row["pmid"]).strip()
        pair_id = f"{nct_id}_{pmid}"
        pair_ids.append(pair_id)

        registered = registered_endpoints[idx]
        published = published_endpoints[idx]
        score = scores[idx]
        routing = _route_from_score(score)
        routings.append(routing.value)

        entry = DecisionLogEntry.from_layer1(
            pair_id=pair_id,
            registered_endpoint=registered,
            published_endpoint=published,
            similarity_score=score,
            routing=routing,
        )

        if routing == EndpointRouting.LLM:
            logger.debug("Layer 2 - LLM adjudication for %s (score=%.4f)", pair_id, score)
            llm_result = _call_llm(registered, published)
            entry.llm_model = LLM_MODEL_PRIMARY

            if llm_result is not None:
                entry.llm_switch_type = llm_result.switch_type
                entry.llm_reasoning = llm_result.step_by_step_reasoning
                entry.llm_confidence = llm_result.confidence
                entry.llm_comparability = llm_result.comparability_for_pooling
                entry.llm_flag = llm_result.flag_for_human_review
            else:
                entry.llm_flag = True
                logger.warning(
                    "Layer 2 - invalid LLM response for %s. Pair flagged for human review.",
                    pair_id,
                )

        decision_log.append(entry)

    processable["pair_id"] = pair_ids
    processable["similarity_score"] = scores
    processable["routing"] = routings

    logger.info(
        "Endpoint matching complete - %d processed | %d LLM calls | $%.6f total LLM cost",
        len(processable),
        sum(1 for routing in routings if routing == EndpointRouting.LLM.value),
        _cost_tracker.total_usd,
    )
    logger.info("Governance snapshot: %s", decision_log.governance_summary())

    return linked_df.merge(
        processable[["nct_id", "pmid", "pair_id", "similarity_score", "routing"]],
        on=["nct_id", "pmid"],
        how="left",
    )
