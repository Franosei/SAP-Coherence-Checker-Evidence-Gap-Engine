"""
SAP Coherence Checker - Audit Dashboard (v3.0).

Run:
    python -m shiny run --port 8030 src/dashboard/app.py
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from htmltools import head_content
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_plotly

from src.dashboard.helpers import (
    ROUTING_LABELS,
    SWITCH_LABELS,
    card,
    default_poolable,
    default_switch,
    empty_figure,
    fmt_num,
    fmt_pct,
    is_spot_check_row,
    kpi_box,
    page_header,
    pending_review_rows,
    read_csv_or_empty,
    safe_text,
    truthy,
)
from src.models.decision_log import DecisionLog
from src.models.linkage_log import LinkageLog
from src.models.schemas import (
    HumanDecision,
    HumanReviewStatus,
    LinkageAuditEntry,
    LinkageConfidence,
    LinkageMethod,
    SwitchType,
)
from src.pipeline.config import (
    DECISION_LOG_PATH,
    GOLD_STANDARD_PATH,
    INTER_RATER_REVIEW_PATH,
    LINKAGE_LOG_PATH,
    LLM_BASE_URL,
    LLM_MODEL_PRIMARY,
    LLM_PROVIDER,
    OVERRIDE_RATE_HIGH_THRESHOLD,
    OVERRIDE_RATE_LOW_THRESHOLD,
    POWER_AUDIT_LOG_PATH,
)
from src.pipeline.module1_linker import _extract_published_endpoint
from src.pipeline.module2_endpoint_matcher import run_endpoint_matching
from src.pipeline.pubmed_client import PubMedClient
from src.pipeline.validation import (
    build_gold_standard_template,
    build_inter_rater_template,
    compute_ai_calibration,
    compute_inter_rater_reliability,
    iter_review_actions,
)

APP_DIR        = Path(__file__).resolve().parent
CSS_PATH       = APP_DIR / "www" / "style.css"
TRIALS_PATH    = Path("data") / "outputs" / "trials.csv"
LINKED_TRIALS_PATH = Path("data") / "outputs" / "linked_trials.csv"
MATCHED_TRIALS_PATH = Path("data") / "outputs" / "matched_trials.csv"

TRIALS_COLUMNS = [
    "nct_id", "official_title", "brief_title", "phase",
    "start_date", "completion_date", "enrollment",
    "primary_outcomes", "secondary_outcomes",
    "results_first_posted_date", "ctgov_pmid",
]

DECISION_COLUMNS = [
    "pair_id", "registered_endpoint", "published_endpoint",
    "similarity_score", "ssi", "routing",
    "llm_model", "llm_switch_type", "llm_reasoning", "llm_confidence",
    "llm_comparability", "llm_flag",
    "human_reviewed", "human_decision", "human_final_class", "human_poolable",
    "override_reason", "reviewer_initials", "review_timestamp",
    "pipeline_version", "created_at",
]
LINKAGE_COLUMNS = [
    "nct_id", "pmid", "linkage_method", "linkage_confidence",
    "timestamp", "linked_by", "notes", "pipeline_version",
]
POWER_COLUMNS = [
    "nct_id", "enrollment", "assumed_hr", "assumed_event_rate",
    "alpha", "power", "posterior_hr_at_registration",
    "optimism_bias", "excluded_reason", "inputs_source", "audit_timestamp",
]
INTER_RATER_COLUMNS = [
    "pair_id", "nct_id", "pmid", "registered_endpoint", "published_endpoint",
    "second_reviewer_initials", "second_switch_type", "second_poolable", "notes",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def safe_input(getter: Any, default: Any = None) -> Any:
    try:
        value = getter()
    except Exception:
        return default
    return default if value is None else value


def kv(*pairs: str) -> ui.Tag:
    """Build a styled key/value grid from flat key, value, key, value… args."""
    items: list[ui.Tag] = []
    it = iter(pairs)
    for key in it:
        val = next(it, "")
        items += [
            ui.div(str(key), class_="kv-term"),
            ui.div(str(val) if val else "Not available", class_="kv-def"),
        ]
    return ui.div(*items, class_="kv-list")


def info_box(message: str, tone: str = "info") -> ui.Tag:
    return ui.div(message, class_=f"alert alert-{tone}")


def _reasoning_block(row: "pd.Series") -> list[ui.Tag]:
    """
    Build the AI reasoning disclosure widget for a decision log row.

    Returns a list so it can be unpacked with ``*_reasoning_block(row)``
    inside a parent ``ui.div()``.

    - auto_concordant / auto_major_switch: explains that no LLM was needed.
    - llm routing + reasoning present: shows the full reasoning in a <details>.
    - llm routing + reasoning blank: explains the LLM response was invalid.
    """
    routing = str(row.get("routing", "")).strip()
    reasoning = str(row.get("llm_reasoning", "")).strip()

    if routing == "auto_concordant":
        return [ui.div(
            "No model reasoning recorded. This pair was auto-classified as concordant because the "
            f"embedding similarity score ({fmt_num(row.get('similarity_score'), 3)}) was ≥ 0.90. "
            "The AI model was not called.",
            class_="alert alert-info",
        )]

    if routing == "auto_major_switch":
        return [ui.div(
            "No model reasoning recorded. This pair was auto-classified as a major switch because the "
            f"embedding similarity score ({fmt_num(row.get('similarity_score'), 3)}) was < 0.50. "
            "The AI model was not called.",
            class_="alert alert-warning",
        )]

    # routing == "llm"
    if not reasoning:
        return [ui.div(
            "The AI model was called for this pair but returned an invalid or empty response. "
            "The pair has been flagged for human review.",
            class_="alert alert-danger",
        )]

    return [ui.tags.details(
        ui.tags.summary("View full AI reasoning (expand)"),
        ui.tags.p(reasoning, class_="details-copy"),
        class_="details-panel",
    )]


def brand_title() -> ui.Tag:
    return ui.div(
        ui.div("SC", class_="brand-icon"),
        ui.div(
            ui.div("SAP Coherence Checker", class_="brand-title"),
            ui.div("v3.0", class_="brand-version"),
            class_="brand-copy",
        ),
        class_="brand-wrap",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_navbar(

    # ── Overview ──────────────────────────────────────────────────────────
    ui.nav_panel(
        "Overview",
        ui.div(
            page_header("Overview"),
            ui.output_ui("overview_kpis"),
            # Explain what the numbers mean
            ui.output_ui("overview_guidance"),
            ui.div(
                card("Routing breakdown",
                     ui.p("How the AI classified each trial pair.",
                          class_="text-muted text-small"),
                     output_widget("routing_overview_plot")),
                card("Review progress",
                     ui.p("How many pairs have been reviewed vs are still pending.",
                          class_="text-muted text-small"),
                     output_widget("review_overview_plot")),
                class_="grid-2",
            ),
            card("Runtime settings", ui.output_ui("runtime_configuration")),
            class_="page-shell",
        ),
    ),

    # ── Linkage Audit ──────────────────────────────────────────────────────
    ui.nav_panel(
        "Linkage Audit",
        ui.div(
            page_header("Linkage Audit"),
            ui.output_ui("linkage_intro"),
            ui.output_ui("linkage_kpis"),
            ui.div(
                card(
                    "Flagged trial-publication links",
                    ui.output_ui("linkage_selector"),
                    ui.tags.hr(),
                    ui.p(
                        "Manual linkage decisions are appended to the linkage audit log with your initials and notes.",
                        class_="text-muted text-small",
                    ),
                    ui.input_text("linkage_reviewer_initials", "Your initials"),
                    ui.output_ui("linkage_feedback"),
                    ui.tags.hr(),
                    ui.output_data_frame("linkage_queue_table"),
                ),
                card("Selected trial: Manual Linkage Review", ui.output_ui("linkage_case_panel")),
                class_="grid-2",
            ),
            card(
                "Full Linkage Audit Log: All NCTs and Linked PMIDs",
                ui.p(
                    "Every NCT ID and every PMID the cascade considered, with the stage, "
                    "confidence, and article-gate verdict. Use this to verify the cascade "
                    "decisions and spot any NCT linked to multiple PMIDs.",
                    class_="text-muted text-small",
                ),
                ui.output_data_frame("full_linkage_log_table"),
            ),
            class_="page-shell",
        ),
    ),

    # ── Review Queue ──────────────────────────────────────────────────────
    ui.nav_panel(
        "Review Queue",
        ui.div(
            page_header("Review Queue"),
            ui.output_ui("review_queue_intro"),
            ui.output_ui("review_queue_kpis"),
            ui.div(
                card(
                    "Pairs to review",
                    ui.output_ui("review_selector"),
                    ui.tags.hr(),
                    ui.p("Your initials are added to the audit log with every decision.",
                         class_="text-muted text-small"),
                    ui.input_text("reviewer_initials", "Your initials"),
                    ui.output_ui("review_feedback"),
                    ui.tags.hr(),
                    ui.p("Recent decisions", style="font-weight:600; margin-bottom:8px;"),
                    ui.output_data_frame("recent_reviews_table"),
                ),
                card("Selected pair: Review Form", ui.output_ui("review_case_panel")),
                class_="grid-2",
            ),
            class_="page-shell",
        ),
    ),

    # ── Decision Log ──────────────────────────────────────────────────────
    ui.nav_panel(
        "Decision Log",
        ui.div(
            page_header("Decision Log"),
            ui.output_ui("decision_log_intro"),
            card(
                "Filters",
                ui.div(
                    ui.input_selectize("filter_routing", "Routing method",
                                       ROUTING_LABELS, multiple=True),
                    ui.input_selectize("filter_confidence", "Model confidence",
                                       {"high": "High", "medium": "Medium", "low": "Low"},
                                       multiple=True),
                    ui.input_selectize("filter_decision", "Human decision",
                                       {"confirm": "Confirm", "override": "Override",
                                        "exclude": "Exclude", "defer": "Defer"},
                                       multiple=True),
                    ui.output_ui("reviewer_filter"),
                    ui.input_text("filter_text", "Search endpoints or pair ID"),
                    class_="filters-grid",
                ),
            ),
            card(
                "All pairs",
                ui.div(
                    ui.output_ui("decision_log_summary"),
                    ui.download_button("download_decision_log_filtered",
                                       "Download filtered CSV",
                                       class_="btn btn-secondary btn-sm"),
                    class_="panel-toolbar",
                ),
                ui.output_data_frame("decision_log_table"),
            ),
            class_="page-shell",
        ),
    ),

    # ── AI Calibration ────────────────────────────────────────────────────
    ui.nav_panel(
        "AI Calibration",
        ui.div(
            page_header("AI Calibration"),
            ui.output_ui("calibration_intro"),
            ui.output_ui("calibration_kpis"),
            card(
                "Validation templates",
                ui.p(
                    "Generate the proposal's manual-review templates directly from the current decision log. "
                    "The gold-standard file contains 20 raw endpoint pairs. "
                    "The inter-rater file contains a blinded 10% sample of already reviewed pairs.",
                    class_="text-muted text-small",
                ),
                ui.div(
                    ui.download_button(
                        "download_gold_standard_template",
                        "Gold standard template (.csv)",
                        class_="btn btn-secondary btn-sm",
                    ),
                    ui.download_button(
                        "download_inter_rater_template",
                        "Inter-rater template (.csv)",
                        class_="btn btn-secondary btn-sm",
                    ),
                    class_="button-strip",
                ),
            ),
            ui.div(
                card("Similarity score distribution",
                     ui.p("Each bar is one pair. Green dashed line = auto-concordant threshold (0.90). "
                          "Red dashed line = auto-major-switch threshold (0.50). "
                          "Pairs between the lines were sent to the AI model.",
                          class_="text-muted text-small"),
                     output_widget("similarity_plot")),
                card("Routing split",
                     ui.p("Share of pairs handled by each classification method.",
                          class_="text-muted text-small"),
                     output_widget("routing_pie_plot")),
                class_="grid-2",
            ),
            card("Gold standard comparison", ui.output_ui("gold_note"),
                 ui.output_data_frame("gold_matrix")),
            class_="page-shell",
        ),
    ),

    # ── Evidence Scorecard ────────────────────────────────────────────────
    ui.nav_panel(
        "Scorecard",
        ui.div(
            page_header("Evidence Scorecard"),
            ui.output_ui("scorecard_intro"),
            ui.output_ui("scorecard_kpis"),
            card("Routing vs final classification",
                 ui.p("Each cell shows how many pairs were routed a certain way and "
                      "then classified by the human reviewer. "
                      "Populated only after at least one review is saved.",
                      class_="text-muted text-small"),
                 output_widget("scorecard_heatmap")),
            class_="page-shell",
        ),
    ),

    # ── Power Audit ───────────────────────────────────────────────────────
    ui.nav_panel(
        "Power Audit",
        ui.div(
            page_header("Power Audit"),
            ui.output_ui("power_intro"),
            ui.output_ui("power_kpis"),
            ui.div(
                card("Assumed vs evidence-based HR",
                     ui.p("Each dot is one trial. X axis = what the evidence said at the "
                          "time of registration. Y axis = what the trial assumed. "
                          "Points above the diagonal were overly optimistic.",
                          class_="text-muted text-small"),
                     output_widget("power_scatter_plot")),
                card("Optimism bias distribution",
                     ui.p("Optimism bias = assumed HR minus posterior HR. "
                          "Positive values mean the trial assumed a larger effect than evidence supported.",
                          class_="text-muted text-small"),
                     output_widget("power_bias_plot")),
                class_="grid-2",
            ),
            card("Excluded trials", ui.output_data_frame("power_excluded_table")),
            class_="page-shell",
        ),
    ),

    # ── Provenance ────────────────────────────────────────────────────────
    ui.nav_panel(
        "Provenance",
        ui.div(
            page_header("Provenance"),
            ui.output_ui("provenance_intro"),
            card("Select pair", ui.output_ui("provenance_selector")),
            card("Full decision trail", ui.output_ui("provenance_panel")),
            class_="page-shell",
        ),
    ),

    # ── Export ────────────────────────────────────────────────────────────
    ui.nav_panel(
        "Export",
        ui.div(
            page_header("Export"),
            ui.output_ui("export_intro"),
            card(
                "Download files",
                ui.div(
                    ui.download_button("download_audit_package",
                                       "Full audit package (.zip)",
                                       class_="btn btn-primary"),
                    ui.download_button("download_decision_log_raw",
                                       "Decision log (.csv)",
                                       class_="btn btn-secondary"),
                    ui.download_button("download_linkage_log_raw",
                                       "Linkage log (.csv)",
                                       class_="btn btn-secondary"),
                    ui.download_button("download_power_audit_raw",
                                       "Power audit log (.csv)",
                                       class_="btn btn-secondary"),
                    class_="button-strip",
                ),
            ),
            card("Governance summary", ui.output_ui("export_governance")),
            class_="page-shell",
        ),
    ),

    title=brand_title(),
    window_title="SAP Coherence Checker",
    header=ui.div(
        ui.div(
            ui.input_action_button("refresh_data", "Reload data",
                                   class_="btn btn-secondary btn-sm"),
            class_="app-toolbar",
        ),
        head_content(ui.include_css(CSS_PATH, method="link_files")),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session) -> None:
    refresh_counter        = reactive.value(0)
    review_feedback_state  = reactive.value[tuple[str, str] | None](None)
    linkage_feedback_state = reactive.value[tuple[str, str] | None](None)

    # ── Refresh ──────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.refresh_data)
    def _refresh() -> None:
        refresh_counter.set(refresh_counter.get() + 1)
        review_feedback_state.set(("info", "Data reloaded."))
        linkage_feedback_state.set(("info", "Data reloaded."))

    # ── Data sources ──────────────────────────────────────────────────────

    @reactive.calc
    def decision_log_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(DECISION_LOG_PATH, DECISION_COLUMNS)

    @reactive.calc
    def linkage_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(LINKAGE_LOG_PATH, LINKAGE_COLUMNS)

    @reactive.calc
    def power_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(POWER_AUDIT_LOG_PATH, POWER_COLUMNS)

    @reactive.calc
    def gold_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(GOLD_STANDARD_PATH, ["pair_id", "gold_switch_type"])

    @reactive.calc
    def inter_rater_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(INTER_RATER_REVIEW_PATH, INTER_RATER_COLUMNS)

    @reactive.calc
    def trials_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(TRIALS_PATH, TRIALS_COLUMNS)

    @reactive.calc
    def linked_trials_frame() -> pd.DataFrame:
        refresh_counter.get()
        return read_csv_or_empty(LINKED_TRIALS_PATH, TRIALS_COLUMNS + [
            "pmid", "linkage_method", "linkage_confidence", "published_endpoint",
            "first_author", "pub_year", "journal", "linkage_notes", "linkage_flag",
        ])

    def _trial_meta(nct_id: str, pair_pmid: str = "") -> dict:
        """Return title, pmid, phase, enrollment, and linkage metadata for an NCT ID."""
        tf = trials_frame()
        lf = linked_trials_frame()
        if not lf.empty:
            linked = lf.loc[
                (lf["nct_id"] == nct_id)
                & ((lf["pmid"] == pair_pmid) if pair_pmid else True)
            ]
            if not linked.empty:
                row = linked.iloc[0]
                return {
                    "title":      str(row.get("official_title") or row.get("brief_title") or "").strip(),
                    "pmid":       str(row.get("pmid", "")).strip(),
                    "phase":      str(row.get("phase", "")).strip(),
                    "enrollment": str(row.get("enrollment", "")).strip(),
                    "completion": str(row.get("completion_date", "")).strip(),
                    "results_posted": str(row.get("results_first_posted_date", "")).strip(),
                    "linkage_confidence": str(row.get("linkage_confidence", "")).strip(),
                    "linkage_method": str(row.get("linkage_method", "")).strip(),
                    "journal": str(row.get("journal", "")).strip(),
                }

        if tf.empty or nct_id not in tf["nct_id"].values:
            return {"pmid": pair_pmid}
        row = tf.loc[tf["nct_id"] == nct_id].iloc[0]
        return {
            "title":      str(row.get("official_title") or row.get("brief_title") or "").strip(),
            "pmid":       pair_pmid or str(row.get("ctgov_pmid", "")).strip(),
            "phase":      str(row.get("phase", "")).strip(),
            "enrollment": str(row.get("enrollment", "")).strip(),
            "completion": str(row.get("completion_date", "")).strip(),
            "results_posted": str(row.get("results_first_posted_date", "")).strip(),
        }

    @reactive.calc
    def queue_frame() -> pd.DataFrame:
        return pending_review_rows(decision_log_frame())

    @reactive.calc
    def linkage_queue_frame() -> pd.DataFrame:
        linked = linked_trials_frame()
        if linked.empty:
            return linked
        flagged = linked[
            linked["linkage_confidence"].isin(["Low", "Unlinked"])
            | linked.get("linkage_flag", pd.Series(index=linked.index, dtype=str)).ne("")
        ].copy()
        if flagged.empty:
            return flagged
        return flagged.sort_values(["linkage_confidence", "nct_id"]).reset_index(drop=True)

    @reactive.calc
    def reviewed_frame() -> pd.DataFrame:
        frame = decision_log_frame()
        frame = frame[frame["human_reviewed"].isin(["yes", "spot_check"])].copy()
        if frame.empty:
            return frame
        frame["_ts"] = pd.to_datetime(frame["review_timestamp"], errors="coerce", utc=True)
        return (frame.sort_values("_ts", ascending=False, na_position="last")
                .drop(columns=["_ts"]))

    @reactive.calc
    def selected_queue_row() -> pd.Series | None:
        queue = queue_frame()
        if queue.empty:
            return None
        pair_id = safe_input(input.selected_review_pair, "")
        if pair_id in queue["pair_id"].values:
            return queue.loc[queue["pair_id"] == pair_id].iloc[0]
        return queue.iloc[0]

    @reactive.calc
    def selected_linkage_row() -> pd.Series | None:
        queue = linkage_queue_frame()
        if queue.empty:
            return None
        nct_id = safe_input(input.selected_linkage_nct, "")
        if nct_id in queue["nct_id"].values:
            return queue.loc[queue["nct_id"] == nct_id].iloc[0]
        return queue.iloc[0]

    @reactive.calc
    def filtered_log() -> pd.DataFrame:
        frame = decision_log_frame().copy()
        if frame.empty:
            return frame
        for col, values in [
            ("routing",           safe_input(input.filter_routing, []) or []),
            ("llm_confidence",    safe_input(input.filter_confidence, []) or []),
            ("human_decision",    safe_input(input.filter_decision, []) or []),
            ("reviewer_initials", safe_input(input.filter_reviewer, []) or []),
        ]:
            if values:
                frame = frame[frame[col].isin(set(values))]
        text = str(safe_input(input.filter_text, "")).strip().lower()
        if text:
            cols = ["pair_id", "registered_endpoint", "published_endpoint",
                    "llm_reasoning", "reviewer_initials"]
            mask = frame[cols].apply(lambda c: c.str.contains(text, case=False, na=False))
            frame = frame[mask.any(axis=1)]
        return frame.reset_index(drop=True)

    def _write_linked_trials(frame: pd.DataFrame) -> None:
        LINKED_TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(LINKED_TRIALS_PATH, index=False)

    def _refresh_matched_outputs(linked_frame: pd.DataFrame, nct_id: str, pmid: str) -> None:
        if not pmid:
            return
        candidate = linked_frame[
            (linked_frame["nct_id"] == nct_id)
            & (linked_frame["pmid"] == pmid)
        ].copy()
        if candidate.empty:
            return

        matched_subset = run_endpoint_matching(candidate)
        update_cols = [
            col for col in ["nct_id", "pmid", "pair_id", "similarity_score", "routing"]
            if col in matched_subset.columns
        ]
        updates = matched_subset[update_cols].drop_duplicates(subset=["nct_id", "pmid"])

        if MATCHED_TRIALS_PATH.exists():
            existing = pd.read_csv(MATCHED_TRIALS_PATH, dtype=str, keep_default_na=False)
            existing_updates = existing[[col for col in update_cols if col in existing.columns]]
            existing_updates = existing_updates.drop_duplicates(subset=["nct_id", "pmid"])
            updates = pd.concat([existing_updates, updates], ignore_index=True)
            updates = updates.drop_duplicates(subset=["nct_id", "pmid"], keep="last")

        refreshed = linked_frame.drop(
            columns=[col for col in ["pair_id", "similarity_score", "routing"] if col in linked_frame.columns],
            errors="ignore",
        ).merge(
            updates,
            on=["nct_id", "pmid"],
            how="left",
        )
        MATCHED_TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        refreshed.to_csv(MATCHED_TRIALS_PATH, index=False)

    @reactive.effect
    @reactive.event(input.submit_linkage_review)
    def _submit_linkage_review() -> None:
        row = selected_linkage_row()
        if row is None:
            linkage_feedback_state.set(("warning", "No linkage case selected."))
            return

        reviewer = str(safe_input(input.linkage_reviewer_initials, "")).strip().upper()
        chosen_conf = str(safe_input(input.linkage_confidence_choice, "")).strip()
        pmid = str(safe_input(input.linkage_pmid, "")).strip()
        notes = str(safe_input(input.linkage_notes, "")).strip()
        nct_id = str(row.get("nct_id", "")).strip()

        if not reviewer:
            linkage_feedback_state.set(("danger", "Enter your initials first."))
            return
        if chosen_conf not in {"High", "Medium", "Low", "Unlinked"}:
            linkage_feedback_state.set(("danger", "Choose a final linkage status."))
            return
        if chosen_conf != "Unlinked" and not pmid:
            linkage_feedback_state.set(("danger", "Enter the resolved PMID for linked cases."))
            return
        if not notes:
            linkage_feedback_state.set(("danger", "Add a short audit note for the manual linkage decision."))
            return

        linked = linked_trials_frame().copy()
        if linked.empty or nct_id not in linked["nct_id"].values:
            linkage_feedback_state.set(("danger", "Could not load linked_trials.csv for this case."))
            return

        fetch_warning = ""
        abstract_text = ""
        published_endpoint = ""
        first_author = ""
        pub_year = ""
        journal = ""

        if pmid and chosen_conf != "Unlinked":
            try:
                record = PubMedClient().fetch_record(pmid)
                abstract_text = record.abstract_text
                published_endpoint = _extract_published_endpoint(record)
                first_author = record.authors[0] if record.authors else ""
                pub_year = record.pub_year
                journal = record.journal
            except Exception as exc:
                fetch_warning = f" PMID fetch failed while saving manual linkage: {exc}"

        idx = linked.index[linked["nct_id"] == nct_id]
        linked.loc[idx, "pmid"] = "" if chosen_conf == "Unlinked" else pmid
        linked.loc[idx, "linkage_method"] = LinkageMethod.MANUAL.value
        linked.loc[idx, "linkage_confidence"] = chosen_conf
        linked.loc[idx, "abstract_text"] = abstract_text
        linked.loc[idx, "published_endpoint"] = published_endpoint
        linked.loc[idx, "first_author"] = first_author
        linked.loc[idx, "pub_year"] = pub_year
        linked.loc[idx, "journal"] = journal
        linked.loc[idx, "linkage_notes"] = notes + fetch_warning
        linked.loc[idx, "linkage_flag"] = "" if chosen_conf in {"High", "Medium"} else "FLAGGED_FOR_REVIEW"
        _write_linked_trials(linked)

        LinkageLog().append(
            LinkageAuditEntry(
                nct_id=nct_id,
                pmid=None if chosen_conf == "Unlinked" else pmid,
                linkage_method=LinkageMethod.MANUAL,
                linkage_confidence=LinkageConfidence(chosen_conf),
                linked_by=f"reviewer:{reviewer}",
                notes=notes + fetch_warning,
            )
        )

        if chosen_conf in {"High", "Medium"} and pmid:
            try:
                _refresh_matched_outputs(linked, nct_id, pmid)
            except Exception as exc:
                linkage_feedback_state.set((
                    "warning",
                    f"Manual linkage saved for {nct_id}, but endpoint matching did not refresh automatically: {exc}",
                ))
                refresh_counter.set(refresh_counter.get() + 1)
                return

        refresh_counter.set(refresh_counter.get() + 1)
        message = f"Saved manual linkage review for {nct_id} ({chosen_conf})."
        if fetch_warning:
            message += " PubMed endpoint extraction still needs manual follow-up."
        linkage_feedback_state.set(("success", message))

    @render.ui
    def linkage_intro() -> ui.Tag:
        queue = linkage_queue_frame()
        if queue.empty:
            return info_box("All trial-publication links are currently resolved to High or Medium confidence.", "success")
        return info_box(
            f"{len(queue)} trial(s) still need publication-link review. "
            "Confirm the PMID manually, choose the final linkage confidence, and save an audit note. "
            "High and Medium manual resolutions are pushed straight into endpoint matching.",
            "warning",
        )

    @render.ui
    def linkage_kpis() -> ui.Tag:
        frame = linked_trials_frame()
        if frame.empty:
            return ui.div(
                kpi_box("Linked trials", "0", "Run Module 1 to populate"),
                kpi_box("High confidence", "0", "Direct or confirmed links", "success"),
                kpi_box("Medium confidence", "0", "Usable after disambiguation", "accent"),
                kpi_box("Flagged", "0", "Low or unlinked", "warning"),
                class_="kpi-grid",
            )
        return ui.div(
            kpi_box("Linked trials", str(len(frame)), "Rows in linked_trials.csv"),
            kpi_box("High confidence", str(len(frame[frame["linkage_confidence"] == "High"])),
                    "Direct or confirmed links", "success"),
            kpi_box("Medium confidence", str(len(frame[frame["linkage_confidence"] == "Medium"])),
                    "Usable after disambiguation", "accent"),
            kpi_box("Flagged", str(len(linkage_queue_frame())),
                    "Low or unlinked cases needing review", "warning"),
            class_="kpi-grid",
        )

    @render.ui
    def linkage_selector() -> ui.Tag:
        queue = linkage_queue_frame()
        if queue.empty:
            return info_box("Linkage queue is clear.", "success")
        choices = {
            str(row["nct_id"]): f"{row['nct_id']}: {safe_text(row.get('official_title') or row.get('brief_title'), 'Untitled trial')[:80]}"
            for _, row in queue.iterrows()
        }
        return ui.input_selectize(
            "selected_linkage_nct",
            "Select a trial to review",
            choices,
            selected=queue.iloc[0]["nct_id"],
        )

    @render.ui
    def linkage_feedback() -> ui.Tag:
        state = linkage_feedback_state.get()
        if not state:
            return ui.div()
        tone, message = state
        return info_box(message, tone if tone in {"info", "success", "warning", "danger"} else "info")

    @render.data_frame
    def linkage_queue_table() -> pd.DataFrame:
        queue = linkage_queue_frame()
        if queue.empty:
            return pd.DataFrame(columns=["nct_id", "pmid", "confidence", "method", "notes"])
        out = queue[["nct_id", "pmid", "linkage_confidence", "linkage_method", "linkage_notes"]].copy()
        out.columns = ["nct_id", "pmid", "confidence", "method", "notes"]
        return out.reset_index(drop=True)

    @render.data_frame
    def full_linkage_log_table() -> pd.DataFrame:
        """Show every entry in the linkage audit log: all NCTs and all linked PMIDs."""
        log = linkage_frame()
        if log.empty:
            return pd.DataFrame(columns=[
                "nct_id", "pmid", "confidence", "method", "linked_by", "timestamp", "notes"
            ])
        cols = {
            "nct_id":              "nct_id",
            "pmid":                "pmid",
            "linkage_confidence":  "confidence",
            "linkage_method":      "method",
            "linked_by":           "linked_by",
            "timestamp":           "timestamp",
            "notes":               "notes",
        }
        # Keep only columns that actually exist in the log CSV
        available = {k: v for k, v in cols.items() if k in log.columns}
        out = log[list(available.keys())].copy()
        out.columns = list(available.values())
        # Sort: unlinked first, then low, medium, high; problems surface at top
        conf_order = {"Unlinked": 0, "Low": 1, "Medium": 2, "High": 3}
        out["_sort"] = out["confidence"].map(conf_order).fillna(99)
        out = out.sort_values(["_sort", "nct_id"]).drop(columns=["_sort"]).reset_index(drop=True)
        return out

    @render.ui
    def linkage_case_panel() -> ui.Tag:
        row = selected_linkage_row()
        if row is None:
            return info_box("No linkage issues pending.", "success")

        nct_id = str(row.get("nct_id", ""))
        title = safe_text(row.get("official_title") or row.get("brief_title"), "Title not available")
        ctgov_url = f"https://clinicaltrials.gov/study/{nct_id}"
        current_pmid = str(row.get("pmid", "")).strip()
        current_pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{current_pmid}/" if current_pmid else ""

        header_links = [ui.tags.a(nct_id, href=ctgov_url, target="_blank", class_="trial-link ctgov-link")]
        if current_pmid:
            header_links.append(
                ui.tags.a(f"PMID {current_pmid}", href=current_pubmed_url, target="_blank", class_="trial-link pubmed-link")
            )

        return ui.div(
            ui.div(
                ui.div(*header_links, class_="trial-id-row"),
                ui.div(title, class_="trial-title-text"),
                ui.div(
                    ui.span(f"Current confidence: {safe_text(row.get('linkage_confidence'), 'Not set')}", class_="trial-badge"),
                    ui.span(f"Current method: {safe_text(row.get('linkage_method'), 'Not set')}", class_="trial-badge"),
                    ui.span(f"CT.gov result PMID: {safe_text(row.get('ctgov_pmid'), 'Not available')}", class_="trial-badge"),
                    class_="trial-badges",
                ),
                class_="trial-header",
            ),
            info_box(
                safe_text(row.get("linkage_notes"), "No audit notes recorded yet."),
                "info",
            ),
            ui.input_text(
                "linkage_pmid",
                "Resolved PMID",
                value=current_pmid,
                placeholder="e.g. 29130810",
            ),
            ui.input_select(
                "linkage_confidence_choice",
                "Final linkage status",
                {
                    "High": "High",
                    "Medium": "Medium",
                    "Low": "Low",
                    "Unlinked": "Unlinked",
                },
                selected=str(row.get("linkage_confidence", "Unlinked")) or "Unlinked",
            ),
            ui.input_text_area(
                "linkage_notes",
                "Audit note",
                rows=4,
                placeholder="e.g. Title mismatch resolved manually after checking the abstract and publication date.",
            ),
            ui.input_action_button("submit_linkage_review", "Save linkage review", class_="btn btn-primary"),
            class_="detail-stack",
        )

    # ── Submit review ─────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.submit_review)
    def _submit_review() -> None:
        row         = selected_queue_row()
        if row is None:
            review_feedback_state.set(("warning", "No pair selected."))
            return
        reviewer    = str(safe_input(input.reviewer_initials, "")).strip().upper()
        decision    = str(safe_input(input.review_decision, "")).strip()
        final_class = str(safe_input(input.review_final_class, default_switch(row))).strip()
        reason      = str(safe_input(input.override_reason, "")).strip()
        poolable    = bool(safe_input(input.review_poolable, default_poolable(row)))

        if not reviewer:
            review_feedback_state.set(("danger", "Enter your initials first."))
            return
        if decision not in {"confirm", "override", "exclude", "defer"}:
            review_feedback_state.set(("danger", "Choose an action."))
            return
        if decision == "override" and not reason:
            review_feedback_state.set(("danger", "Enter a reason for the override."))
            return

        if decision == "defer":
            status = HumanReviewStatus.NO
        elif is_spot_check_row(row, decision_log_frame()):
            status = HumanReviewStatus.SPOT_CHECK
        else:
            status = HumanReviewStatus.YES
        try:
            DecisionLog().record_human_review(
                pair_id=str(row["pair_id"]),
                human_decision=HumanDecision(decision),
                human_final_class=SwitchType(final_class),
                human_poolable=poolable,
                reviewer_initials=reviewer,
                override_reason=reason or None,
                review_status=status,
            )
        except Exception as exc:
            review_feedback_state.set(("danger", f"Save failed: {exc}"))
            return

        refresh_counter.set(refresh_counter.get() + 1)
        msg = (f"{row['pair_id']} deferred."
               if decision == "defer"
               else f"Saved: {row['pair_id']} → {decision}.")
        review_feedback_state.set(("success", msg))

    # ── Overview ──────────────────────────────────────────────────────────

    @render.ui
    def overview_kpis() -> ui.Tag:
        frame    = decision_log_frame()
        pending  = len(queue_frame())
        reviewed = len(reviewed_frame())
        poolable = len(frame[frame["human_poolable"].map(truthy)])
        total    = len(frame)
        return ui.div(
            kpi_box("Trial pairs",    str(total),    "Linked trial-publication pairs in this run"),
            kpi_box("Review queue",   str(pending),  "Pairs awaiting reviewer decision",                           "warning"),
            kpi_box("Reviewed",       str(reviewed), "Pairs reviewed",                                             "success"),
            kpi_box("Poolable",       str(poolable), "Pairs confirmed eligible for meta-analysis",                 "accent"),
            class_="kpi-grid",
        )

    @render.ui
    def overview_guidance() -> ui.Tag:
        frame   = decision_log_frame()
        pending = len(queue_frame())
        if frame.empty:
            return info_box(
                "No pipeline data found. Run the pipeline first: "
                "python run_pipeline.py --fresh-run --max-trials 20",
                "info",
            )
        auto_major = len(frame[frame["routing"] == "auto_major_switch"])
        auto_conc  = len(frame[frame["routing"] == "auto_concordant"])
        no_pub_ep  = len(frame[frame["published_endpoint"].eq("")])

        lines = [
            f"The pipeline linked {len(frame)} breast cancer trial-publication pairs using the PubMed linkage cascade.",
            f"{auto_conc} pairs were routed as concordant.",
            f"{auto_major} pairs were routed as major switch.",
            f"{len(frame[frame['routing']=='llm'])} pairs were sent to the model for adjudication.",
        ]
        if no_pub_ep:
            lines.append(
                f"{no_pub_ep} pairs are missing a published endpoint. "
                "PubMed/PMC did not return usable endpoint text. "
                "These still need manual review."
            )
        if pending > 0:
            lines.append(
                f"{pending} pair(s) remain in the review queue."
            )
        else:
            lines.append("All pairs have been reviewed.")

        return ui.div(
            ui.tags.ul(*[ui.tags.li(line) for line in lines]),
            class_="alert alert-info",
        )

    @render.ui
    def runtime_configuration() -> ui.Tag:
        reviewed = reviewed_frame()
        rate     = (len(reviewed[reviewed["human_decision"] == "override"])
                    / max(len(reviewed), 1))
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            key_status = "missing"
        elif api_key.lower().startswith("your_") or "api_key_here" in api_key.lower():
            key_status = "placeholder / invalid"
        else:
            key_status = "present"

        if rate > OVERRIDE_RATE_HIGH_THRESHOLD:
            note = f"{rate:.0%} (high: consider recalibrating similarity thresholds)"
        elif rate < OVERRIDE_RATE_LOW_THRESHOLD and len(reviewed) > 0:
            note = f"{rate:.0%} (very low: verify you are not over-trusting the model)"
        else:
            note = f"{rate:.0%} (within expected range)"

        return kv(
            "AI provider",    LLM_PROVIDER,
            "Model",          LLM_MODEL_PRIMARY,
            "API endpoint",   LLM_BASE_URL,
            "API key status", key_status,
            "Override rate",  note,
        )

    @render_plotly
    def routing_overview_plot():
        frame = decision_log_frame()
        if frame.empty:
            return empty_figure("Routing breakdown", "No data yet.")
        counts = (frame["routing"].replace(ROUTING_LABELS)
                  .value_counts().reset_index())
        counts.columns = ["routing", "count"]
        return (px.bar(counts, x="routing", y="count", text="count", color="routing",
                       color_discrete_sequence=["#1a4a6b", "#15896b", "#e67e22"])
                .update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10)))

    @render_plotly
    def review_overview_plot():
        frame = decision_log_frame()
        if frame.empty:
            return empty_figure("Review progress", "No data yet.")
        data = pd.DataFrame({
            "status": ["Pending review", "Reviewed", "Auto-classified"],
            "count":  [
                len(queue_frame()),
                len(reviewed_frame()),
                max(len(frame) - len(queue_frame()) - len(reviewed_frame()), 0),
            ],
        })
        return (px.bar(data, x="status", y="count", text="count", color="status",
                       color_discrete_sequence=["#e67e22", "#27ae60", "#1a4a6b"])
                .update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10)))

    # ── Review Queue ──────────────────────────────────────────────────────

    @render.ui
    def review_queue_intro() -> ui.Tag:
        pending = len(queue_frame())
        if pending == 0:
            return info_box("All pairs have been reviewed. Nothing left to do here.", "success")
        queue = queue_frame()
        missing_pub = len(queue[queue["published_endpoint"].eq("")])
        spot_checks = sum(is_spot_check_row(row, decision_log_frame()) for _, row in queue.iterrows())
        return info_box(
            f"{pending} pair(s) need your review. "
            "For each pair, read the registered CT.gov endpoint and the published PubMed endpoint, then choose "
            "Confirm (model was right), Override (model is wrong: pick the correct class), "
            "Exclude (pair should not be used), or Defer (come back later). "
            + (
                f"{missing_pub} pair(s) are waiting on manual review because no published endpoint text was extracted. "
                if missing_pub
                else ""
            )
            + (
                f"{spot_checks} pair(s) were selected for the required spot-check sample."
                if spot_checks
                else ""
            ),
            "warning",
        )

    @render.ui
    def review_queue_kpis() -> ui.Tag:
        frame = decision_log_frame()
        return ui.div(
            kpi_box("Pending",         str(len(queue_frame())),
                    "Pairs still to review",                            "warning"),
            kpi_box("Reviewed",        str(len(reviewed_frame())),
                    "Completed",                                        "success"),
            kpi_box("Sent to AI model",str(len(frame[frame["routing"] == "llm"])),
                    "Pairs in the ambiguous similarity zone"),
            kpi_box("Missing endpoint",str(len(frame[frame["published_endpoint"].eq("")])),
                    "No published endpoint found. These pairs require manual endpoint extraction.","danger"),
            class_="kpi-grid",
        )

    @render.ui
    def review_selector() -> ui.Tag:
        queue = queue_frame()
        if queue.empty:
            return info_box("Queue is clear.", "success")
        choices = {pid: pid for pid in queue["pair_id"].tolist()}
        return ui.input_selectize(
            "selected_review_pair", "Select a pair to review",
            choices, selected=queue.iloc[0]["pair_id"],
        )

    @render.ui
    def review_feedback() -> ui.Tag:
        state = review_feedback_state.get()
        if not state:
            return ui.div()
        tone, message = state
        return info_box(message, tone if tone in {"info","success","warning","danger"} else "info")

    @render.data_frame
    def recent_reviews_table() -> pd.DataFrame:
        reviewed = reviewed_frame()
        if reviewed.empty:
            return pd.DataFrame(columns=["pair_id", "decision", "classification",
                                         "reviewer", "timestamp"])
        out = reviewed[["pair_id", "human_decision", "human_final_class",
                        "reviewer_initials", "review_timestamp"]].head(8).copy()
        out.columns = ["pair_id", "decision", "classification", "reviewer", "timestamp"]
        return out.reset_index(drop=True)

    @render.ui
    def review_case_panel() -> ui.Tag:
        row = selected_queue_row()
        if row is None:
            return info_box("Queue is empty.", "success")

        pair_id = str(row.get("pair_id", ""))
        nct_id, pair_pmid = pair_id.split("_", 1) if "_" in pair_id else (pair_id, "")
        pair_pmid = "" if pair_pmid == "ctgov" else pair_pmid
        meta = _trial_meta(nct_id, pair_pmid)

        reg_ep  = safe_text(row.get("registered_endpoint"), "Not available.")
        pub_ep  = safe_text(row.get("published_endpoint"), "")
        score   = fmt_num(row.get("similarity_score"), 3)
        routing = str(row.get("routing", ""))

        title      = meta.get("title", "")
        pmid       = pair_pmid or meta.get("pmid", "")
        phase      = meta.get("phase", "")
        enrollment = meta.get("enrollment", "")
        completion = meta.get("completion", "")
        linkage_confidence = meta.get("linkage_confidence", "")
        linkage_method = meta.get("linkage_method", "")
        journal = meta.get("journal", "")

        # --- Trial header ---
        ctgov_url  = f"https://clinicaltrials.gov/study/{nct_id}"
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        header_children: list = [
            ui.tags.a(nct_id, href=ctgov_url, target="_blank",
                      class_="trial-link ctgov-link"),
        ]
        if pmid:
            header_children.append(
                ui.tags.a(f"PMID {pmid}", href=pubmed_url, target="_blank",
                          class_="trial-link pubmed-link"),
            )
        else:
            header_children.append(
                ui.span("No PubMed record linked", class_="text-muted text-small"),
            )

        trial_header = ui.div(
            ui.div(*header_children, class_="trial-id-row"),
            ui.div(title or "Title not available", class_="trial-title-text"),
            ui.div(
                ui.span(f"Phase: {phase or 'Not recorded'}", class_="trial-badge"),
                ui.span(f"Enrolled: {enrollment or 'Not recorded'}", class_="trial-badge"),
                ui.span(f"Completed: {completion or 'Not recorded'}", class_="trial-badge"),
                ui.span(f"Linkage: {linkage_confidence or 'Not set'}", class_="trial-badge"),
                ui.span(f"Method: {linkage_method or 'Not set'}", class_="trial-badge"),
                ui.span(f"Journal: {journal or 'Not recorded'}", class_="trial-badge"),
                class_="trial-badges",
            ),
            class_="trial-header",
        )

        review_triggers: list[str] = []
        if routing == "llm":
            review_triggers.append("Similarity score fell in the ambiguous 0.50-0.89 range, so the AI model was called.")
        if str(row.get("llm_confidence", "")).lower() == "low":
            review_triggers.append("The AI model self-reported low confidence, so human review is mandatory.")
        if truthy(row.get("llm_flag")):
            review_triggers.append("The AI model explicitly flagged this pair for human review.")
        if str(row.get("llm_comparability", "")).lower() == "false":
            review_triggers.append("The AI model judged the endpoints not comparable for pooling.")
        if not str(row.get("published_endpoint", "")).strip():
            review_triggers.append("No published endpoint text could be extracted from the linked PubMed/PMC record.")
        if is_spot_check_row(row, decision_log_frame()):
            review_triggers.append("This pair was selected in the proposal's 10% spot-check sample for auto-routed decisions.")

        # --- Routing note ---
        if routing == "auto_major_switch":
            routing_note = (
                f"Similarity score {score}: endpoints look substantially different. "
                "Auto-classified as major switch."
            )
            routing_tone = "danger"
        elif routing == "auto_concordant":
            routing_note = f"Similarity score {score}: endpoints appear semantically equivalent. Auto-classified as concordant."
            routing_tone = "success"
        else:
            ai_class = SWITCH_LABELS.get(str(row.get("llm_switch_type")), "")
            routing_note = (
                f"Similarity score {score}: ambiguous zone (0.50 to 0.89). Sent to the model for adjudication. "
                + (f"AI classified as: {ai_class}." if ai_class
                   else "The model returned an invalid response. Review the endpoints manually.")
            )
            routing_tone = "warning"

        # --- Published endpoint display ---
        pub_ep_display = pub_ep if pub_ep else (
            "No published endpoint text was extracted from the linked PubMed/PMC record. "
            + (f"Open PMID {pmid} on PubMed and check the abstract or full text manually."
               if pmid else f"Check the trial record directly: {nct_id}.")
        )
        pub_ep_tone = "endpoint-text" if pub_ep else "endpoint-text text-muted"

        return ui.div(
            trial_header,
            info_box(routing_note, routing_tone),
            ui.tags.ul(*[ui.tags.li(trigger) for trigger in review_triggers], class_="trigger-list")
            if review_triggers else ui.div(),

            # Endpoint comparison
            ui.div(
                ui.div(
                    ui.div("Registered endpoint (ClinicalTrials.gov, pre-specified)",
                           class_="endpoint-source"),
                    ui.div(reg_ep, class_="endpoint-text"),
                    class_="endpoint-box",
                ),
                ui.div(
                    ui.div("Published endpoint (PubMed / PMC abstract or full text)",
                           class_="endpoint-source"),
                    ui.div(pub_ep_display, class_=pub_ep_tone),
                    class_="endpoint-box",
                ),
                class_="endpoint-grid",
            ),

            # AI signals
            ui.div(
                ui.div(f"Similarity score: {score}",                                                   class_="signal-item"),
                ui.div(f"Switch index (SSI): {fmt_num(row.get('ssi'), 1)}",                            class_="signal-item"),
                ui.div(f"AI classification: {SWITCH_LABELS.get(str(row.get('llm_switch_type')), 'Not classified')}", class_="signal-item"),
                ui.div(f"AI confidence: {safe_text(row.get('llm_confidence'), 'Not called')}",         class_="signal-item"),
                class_="signals-grid",
            ),

            # Full model reasoning chain -- shown only when the model was called
            *_reasoning_block(row),

            # Review form
            ui.tags.hr(),
            ui.p("Your decision", style="font-weight:700; margin-bottom:12px;"),
            ui.input_select(
                "review_decision", "Action",
                {"":        "Select an action",
                 "confirm": "Confirm: model classification is correct",
                 "override": "Override: model is wrong, I am correcting the classification",
                 "exclude": "Exclude: this pair should not be in the analysis",
                 "defer":   "Defer: I will review this pair later"},
                selected="",
            ),
            ui.input_select(
                "review_final_class", "Final classification",
                SWITCH_LABELS, selected=default_switch(row),
            ),
            ui.input_checkbox(
                "review_poolable",
                "Eligible for quantitative pooling in the meta-analysis",
                value=default_poolable(row),
            ),
            ui.input_text_area(
                "override_reason",
                "Reason for override (required if action = Override)",
                rows=3,
                placeholder="e.g. Composite components changed: the published paper removed urgent hospitalisation from the composite.",
            ),
            ui.input_action_button("submit_review", "Save decision",
                                   class_="btn btn-primary"),
            class_="detail-stack",
        )

    # ── Decision Log ──────────────────────────────────────────────────────

    @render.ui
    def decision_log_intro() -> ui.Tag:
        return info_box(
            "This is the complete audit log for every pair the pipeline processed. "
            "Each row records the registered endpoint, the published endpoint, "
            "the AI classification, and your review decision. "
            "Use the filters below to find specific rows, then download if needed.",
            "info",
        )

    @render.ui
    def reviewer_filter() -> ui.Tag:
        reviewers = sorted({r for r in decision_log_frame()["reviewer_initials"].tolist() if r})
        return ui.input_selectize(
            "filter_reviewer", "Reviewed by",
            {r: r for r in reviewers}, multiple=True,
        )

    @render.ui
    def decision_log_summary() -> ui.Tag:
        return ui.div(
            f"{len(filtered_log())} of {len(decision_log_frame())} rows",
            class_="summary-chip",
        )

    @render.data_frame
    def decision_log_table() -> pd.DataFrame:
        frame = filtered_log()
        if frame.empty:
            return frame
        # Show the most useful columns first
        show = [c for c in [
            "pair_id", "routing", "similarity_score",
            "llm_switch_type", "llm_confidence",
            "human_decision", "human_final_class", "human_poolable",
            "reviewer_initials", "review_timestamp",
        ] if c in frame.columns]
        return frame[show].reset_index(drop=True)

    @render.download(
        filename=lambda: f"decision-log-{datetime.now(UTC).date().isoformat()}.csv"
    )
    def download_decision_log_filtered() -> str:
        return filtered_log().to_csv(index=False)

    # ── AI Calibration ────────────────────────────────────────────────────

    @render.ui
    def calibration_intro() -> ui.Tag:
        frame = decision_log_frame()
        gold = gold_frame()
        inter_rater = inter_rater_frame()
        recommended_actions: list[str] = []
        if not frame.empty and not gold.empty:
            recommended_actions = list(iter_review_actions(compute_ai_calibration(frame, gold)))
        return info_box(
            "This panel shows how the AI performed against the proposal's validation workflow. "
            "Use the template downloads above to build the 20-pair gold standard and the blinded 10% inter-rater sample. "
            "The similarity score histogram shows the embedding similarity distribution across all pairs. "
            + ("No inter-rater review file is loaded yet. " if inter_rater.empty else "")
            + (" ".join(recommended_actions) if recommended_actions else ""),
            "info",
        )

    @render.ui
    def calibration_kpis() -> ui.Tag:
        frame = decision_log_frame()
        reviewed = reviewed_frame()
        gold = gold_frame()
        inter_rater = inter_rater_frame()
        calibration = compute_ai_calibration(frame, gold) if (not frame.empty and not gold.empty) else {}
        kappa_metrics = (
            compute_inter_rater_reliability(frame, inter_rater)
            if (not frame.empty and not inter_rater.empty)
            else {}
        )
        return ui.div(
            kpi_box("Model adjudication rate", fmt_pct(len(frame[frame["routing"] == "llm"]), len(frame)),
                    "Share of pairs sent to the AI model", "accent"),
            kpi_box("Gold AUC", fmt_num(calibration.get("auc"), 3, "-"),
                    "Target > 0.80", "primary"),
            kpi_box("Precision (mod+)", fmt_num(calibration.get("precision_moderate_or_above"), 3, "-"),
                    "Target > 0.75", "success"),
            kpi_box("Recall (mod+)", fmt_num(calibration.get("recall_moderate_or_above"), 3, "-"),
                    "Target > 0.80", "warning"),
            kpi_box("Inter-rater kappa", fmt_num(kappa_metrics.get("cohen_kappa"), 3, "-"),
                    "Second-reviewer agreement", "accent"),
            kpi_box("Reviewed", fmt_pct(len(reviewed), len(frame)),
                    "Share of log that has been human-reviewed", "success"),
            class_="kpi-grid",
        )

    @render_plotly
    def similarity_plot():
        frame = decision_log_frame()
        if frame.empty:
            return empty_figure("Similarity scores", "No data yet.")
        plot = frame.copy()
        plot["similarity_score"] = pd.to_numeric(plot["similarity_score"], errors="coerce")
        plot = plot[plot["similarity_score"].notna()].copy()
        if plot.empty:
            return empty_figure("Similarity scores", "No numeric scores yet.")
        plot["outcome"] = plot["human_final_class"].replace("", "not yet reviewed")
        fig = (px.histogram(plot, x="similarity_score", color="outcome", nbins=35,
                            labels={"similarity_score": "Similarity score (0 to 1)",
                                    "outcome": "Human classification"})
               .update_layout(margin=dict(l=10, r=10, t=10, b=10)))
        fig.add_vline(x=0.90, line_dash="dash", line_color="#15896b",
                      annotation_text="Auto-concordant (0.90)")
        fig.add_vline(x=0.50, line_dash="dash", line_color="#c0392b",
                      annotation_text="Auto-major-switch (0.50)")
        return fig

    @render_plotly
    def routing_pie_plot():
        frame = decision_log_frame()
        if frame.empty:
            return empty_figure("Routing split", "No data yet.")
        counts = (frame["routing"].replace(ROUTING_LABELS)
                  .value_counts().reset_index())
        counts.columns = ["routing", "count"]
        return (px.pie(counts, values="count", names="routing", hole=0.45,
                       color_discrete_sequence=["#1a4a6b", "#15896b", "#e67e22"])
                .update_layout(margin=dict(l=10, r=10, t=10, b=10)))

    @render.ui
    def gold_note() -> ui.Tag:
        if gold_frame().empty:
            return info_box(
                "No gold standard file found. "
                "Use the template download above or add data/gold_standard/gold_standard.csv "
                "with at least pair_id and gold_switch_type columns.",
                "info",
            )
        metrics = compute_ai_calibration(decision_log_frame(), gold_frame())
        message = f"{len(gold_frame())} gold-standard pairs loaded."
        actions = list(iter_review_actions(metrics))
        if actions:
            message += " " + " ".join(actions)
        return info_box(message, "success")

    @render.download(
        filename=lambda: f"gold-standard-template-{datetime.now(UTC).date().isoformat()}.csv"
    )
    def download_gold_standard_template() -> str:
        return build_gold_standard_template(decision_log_frame()).to_csv(index=False)

    @render.download(
        filename=lambda: f"inter-rater-template-{datetime.now(UTC).date().isoformat()}.csv"
    )
    def download_inter_rater_template() -> str:
        return build_inter_rater_template(decision_log_frame()).to_csv(index=False)

    @render.data_frame
    def gold_matrix() -> pd.DataFrame:
        frame = decision_log_frame()
        gold  = gold_frame()
        if frame.empty or gold.empty:
            return pd.DataFrame(columns=["(no data)"])
        merged = frame.merge(gold[["pair_id", "gold_switch_type"]], on="pair_id", how="inner")
        if merged.empty:
            return pd.DataFrame({"message": ["No matching pair IDs."]})
        merged["ai_switch"] = merged.apply(
            lambda r: (r["llm_switch_type"] or
                       ("concordant" if r["routing"] == "auto_concordant" else "major_switch")),
            axis=1,
        )
        return pd.crosstab(
            merged["ai_switch"], merged["gold_switch_type"]
        ).rename_axis("AI →  / Gold ↓").reset_index()

    # ── Scorecard ─────────────────────────────────────────────────────────

    @render.ui
    def scorecard_intro() -> ui.Tag:
        reviewed = len(reviewed_frame())
        if reviewed == 0:
            return info_box(
                "Nothing to show yet. Go to Review Queue and review at least one pair first.",
                "warning",
            )
        return info_box(
            f"Based on {reviewed} reviewed pair(s). "
            "The heatmap shows how routing method and human classification relate. "
            "Switch rate = share of reviewed pairs where the endpoint changed between "
            "registration and publication.",
            "info",
        )

    @render.ui
    def scorecard_kpis() -> ui.Tag:
        frame    = decision_log_frame()
        reviewed = reviewed_frame()
        switched = reviewed[reviewed["human_final_class"].isin(
            ["minor_modification", "moderate_switch", "major_switch"])]
        agreed   = reviewed[reviewed["llm_switch_type"].eq(reviewed["human_final_class"])]
        return ui.div(
            kpi_box("Reviewed",          fmt_pct(len(reviewed), len(frame)),
                    "Of all pairs",                              "accent"),
            kpi_box("Confirmed switches", fmt_pct(len(switched), len(reviewed)),
                    "Endpoint changed between registration and publication", "warning"),
            kpi_box("AI-human agreement", fmt_pct(len(agreed), len(reviewed)),
                    "AI and reviewer agreed on classification",  "success"),
            kpi_box("Poolable",           str(len(frame[frame["human_poolable"].map(truthy)])),
                    "Approved for meta-analysis",                "primary"),
            class_="kpi-grid",
        )

    @render_plotly
    def scorecard_heatmap():
        frame = decision_log_frame()
        plot  = frame[frame["human_final_class"].ne("")].copy()
        if plot.empty:
            return empty_figure("Routing vs classification",
                                "No reviews saved yet.")
        pivot = pd.crosstab(plot["routing"].replace(ROUTING_LABELS),
                            plot["human_final_class"].replace(SWITCH_LABELS))
        return (px.imshow(pivot, text_auto=True,
                          color_continuous_scale=["#eef4fb", "#7aa0c4", "#1a4a6b"],
                          aspect="auto",
                          labels={"x": "Human classification", "y": "Routing method"})
                .update_layout(margin=dict(l=10, r=10, t=10, b=10)))

    # ── Power Audit ───────────────────────────────────────────────────────

    @render.ui
    def power_intro() -> ui.Tag:
        if power_frame().empty:
            return info_box(
                "Power audit data not found. "
                "This panel is populated after Module 4 runs. "
                "Run the full pipeline to generate power audit results.",
                "info",
            )
        return info_box(
            "Each trial's sample size implies an assumed treatment effect. "
            "This panel compares that assumed effect against what the evidence "
            "actually showed at the time the trial was designed.",
            "info",
        )

    @render.ui
    def power_kpis() -> ui.Tag:
        frame = power_frame()
        if frame.empty:
            return ui.div(
                kpi_box("Trials audited", "0", "Run Module 4 to populate"),
                kpi_box("Included", "0", "With valid inputs",     "success"),
                kpi_box("Excluded", "0", "Missing inputs",        "warning"),
                kpi_box("Mean bias", "-", "Assumed minus posterior HR", "accent"),
                class_="kpi-grid",
            )
        excluded = frame[frame["excluded_reason"].ne("")]
        bias     = pd.to_numeric(frame["optimism_bias"], errors="coerce").dropna()
        return ui.div(
            kpi_box("Trials audited", str(len(frame)),
                    "Rows in power audit log"),
            kpi_box("Included", str(len(frame) - len(excluded)),
                    "Valid inputs",                               "success"),
            kpi_box("Excluded", str(len(excluded)),
                    "Missing inputs",                             "warning"),
            kpi_box("Mean bias", fmt_num(bias.mean(), 3, "-"),
                    "Assumed HR − posterior HR. Positive = optimistic", "accent"),
            class_="kpi-grid",
        )

    @render_plotly
    def power_scatter_plot():
        frame = power_frame()
        if frame.empty:
            return empty_figure("Assumed vs evidence-based HR", "Run Module 4 first.")
        plot = frame.copy()
        for col in ("assumed_hr", "posterior_hr_at_registration"):
            plot[col] = pd.to_numeric(plot[col], errors="coerce")
        plot = plot.dropna(subset=["assumed_hr", "posterior_hr_at_registration"])
        if plot.empty:
            return empty_figure("Assumed vs evidence-based HR", "No comparable values yet.")
        fig = px.scatter(
            plot, x="posterior_hr_at_registration", y="assumed_hr",
            hover_data=["nct_id"],
            labels={"posterior_hr_at_registration": "Evidence-based HR at registration",
                    "assumed_hr": "HR assumed in sample size calculation"},
        )
        mn = min(plot["posterior_hr_at_registration"].min(), plot["assumed_hr"].min()) * 0.95
        mx = max(plot["posterior_hr_at_registration"].max(), plot["assumed_hr"].max()) * 1.05
        fig.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx,
                      line=dict(dash="dash", color="#7f8c8d"),
                      name="No bias line")
        return fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    @render_plotly
    def power_bias_plot():
        frame = power_frame()
        if frame.empty:
            return empty_figure("Optimism bias", "No data yet.")
        plot = frame.copy()
        plot["optimism_bias"] = pd.to_numeric(plot["optimism_bias"], errors="coerce")
        plot = plot.dropna(subset=["optimism_bias"])
        if plot.empty:
            return empty_figure("Optimism bias", "Not yet computed.")
        fig = px.histogram(
            plot, x="optimism_bias", nbins=20,
            labels={"optimism_bias": "Optimism bias (assumed HR − evidence HR)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#7f8c8d",
                      annotation_text="Zero bias")
        return fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    @render.data_frame
    def power_excluded_table() -> pd.DataFrame:
        frame = power_frame()
        if frame.empty:
            return pd.DataFrame(columns=["nct_id", "reason"])
        out = (frame[frame["excluded_reason"].ne("")]
               [["nct_id", "excluded_reason"]].copy())
        out.columns = ["NCT ID", "reason"]
        return out.reset_index(drop=True)

    # ── Provenance ────────────────────────────────────────────────────────

    @render.ui
    def provenance_intro() -> ui.Tag:
        return info_box(
            "Select any pair to see every decision the pipeline made for it: "
            "how it was linked to a publication, what similarity score it got, "
            "what the AI said, and what the reviewer decided.",
            "info",
        )

    @render.ui
    def provenance_selector() -> ui.Tag:
        frame = decision_log_frame()
        if frame.empty:
            return info_box("No pairs available.", "info")
        choices = {pid: pid for pid in frame["pair_id"].tolist()}
        return ui.input_selectize(
            "provenance_pair", "Pair ID", choices,
            selected=frame.iloc[0]["pair_id"],
        )

    @render.ui
    def provenance_panel() -> ui.Tag:
        frame = decision_log_frame()
        if frame.empty:
            return info_box("No data available.", "info")
        pair_id = safe_input(input.provenance_pair, frame.iloc[0]["pair_id"])
        row = (frame.loc[frame["pair_id"] == pair_id].iloc[0]
               if pair_id in frame["pair_id"].values else frame.iloc[0])

        nct_id = str(row.get("pair_id", "")).split("_")[0]
        this_pmid = pair_id.split("_")[-1] if "_" in pair_id else "Not linked"

        # Show every linkage entry for this NCT, not just the first
        log = linkage_frame()
        nct_linkage = log[log["nct_id"] == nct_id] if not log.empty else pd.DataFrame()
        if nct_linkage.empty:
            linkage_text = "No linkage entry."
        else:
            parts = []
            for _, lr in nct_linkage.iterrows():
                pmid_str = safe_text(lr.get("pmid"), "Not available")
                conf     = safe_text(lr.get("linkage_confidence"), "Unknown")
                method   = safe_text(lr.get("linkage_method"), "Unknown")
                parts.append(f"PMID {pmid_str} ({method} / {conf})")
            linkage_text = " | ".join(parts)

        return ui.div(
            kv(
                "NCT ID",              nct_id,
                "Pair PMID",           this_pmid,
                "All linked PMIDs",    linkage_text,
                "Registered endpoint", safe_text(row.get("registered_endpoint")),
                "Published endpoint",  safe_text(row.get("published_endpoint"), "None extracted"),
                "Similarity score",    fmt_num(row.get("similarity_score"), 4),
                "Routing",             ROUTING_LABELS.get(str(row.get("routing")), safe_text(row.get("routing"))),
                "AI classification",   SWITCH_LABELS.get(str(row.get("llm_switch_type")), "Not classified"),
                "AI confidence",       safe_text(row.get("llm_confidence"), "Not called"),
                "Human decision",      safe_text(row.get("human_decision"), "Not reviewed"),
                "Human final class",   SWITCH_LABELS.get(str(row.get("human_final_class")), "Pending review"),
                "Poolable",            "Yes" if truthy(row.get("human_poolable")) else "No",
                "Reviewer",            safe_text(row.get("reviewer_initials"), "Not yet reviewed"),
                "Override reason",     safe_text(row.get("override_reason"), "No override"),
                "Review timestamp",    safe_text(row.get("review_timestamp"), "Not yet reviewed"),
            ),
            ui.tags.hr(style="margin: 16px 0;"),
            *_reasoning_block(row),
        )

    # ── Export ────────────────────────────────────────────────────────────

    @render.ui
    def export_intro() -> ui.Tag:
        return info_box(
            "Download the audit logs for this pipeline run. "
            "The full audit package bundles all three logs into a single zip file.",
            "info",
        )

    @render.ui
    def export_governance() -> ui.Tag:
        summary = DecisionLog().governance_summary()
        if not summary:
            return info_box("Decision log is empty.", "info")
        return kv(
            "Total pairs",          str(summary.get("total_pairs", 0)),
            "Model adjudication rate",        f"{summary.get('llm_call_rate_pct', 0)}%",
            "Human review rate",    f"{summary.get('human_review_rate_pct', 0)}%",
            "Human override rate",  f"{summary.get('human_override_rate_pct', 0)}%",
        )

    @render.download(
        filename=lambda: f"sap-audit-{datetime.now(UTC).date().isoformat()}.zip",
        media_type="application/zip",
    )
    def download_audit_package() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as arc:
            for name, path in {
                "decision_log.csv":      DECISION_LOG_PATH,
                "linkage_audit_log.csv": LINKAGE_LOG_PATH,
                "power_audit_log.csv":   POWER_AUDIT_LOG_PATH,
                "gold_standard.csv":     GOLD_STANDARD_PATH,
                "inter_rater_review.csv": INTER_RATER_REVIEW_PATH,
            }.items():
                if path.exists():
                    arc.write(path, name)
                else:
                    arc.writestr(name, f"# {name} not yet generated\n")
        buf.seek(0)
        return buf.getvalue()

    @render.download(filename="decision_log.csv")
    def download_decision_log_raw() -> str:
        return (DECISION_LOG_PATH.read_text(encoding="utf-8")
                if DECISION_LOG_PATH.exists()
                else "pair_id\n# Not yet generated\n")

    @render.download(filename="linkage_audit_log.csv")
    def download_linkage_log_raw() -> str:
        return (LINKAGE_LOG_PATH.read_text(encoding="utf-8")
                if LINKAGE_LOG_PATH.exists()
                else "nct_id\n# Not yet generated\n")

    @render.download(filename="power_audit_log.csv")
    def download_power_audit_raw() -> str:
        return (POWER_AUDIT_LOG_PATH.read_text(encoding="utf-8")
                if POWER_AUDIT_LOG_PATH.exists()
                else "nct_id\n# Not yet generated\n")


app = App(app_ui, server)
