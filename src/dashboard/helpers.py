"""
Shared helpers for the Shiny dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from shiny import ui

from src.pipeline.validation import select_spot_check_pairs

ROUTING_LABELS = {
    "auto_concordant": "Auto concordant",
    "llm": "LLM adjudication",
    "auto_major_switch": "Auto major switch",
}

SWITCH_LABELS = {
    "concordant": "Concordant",
    "minor_modification": "Minor modification",
    "moderate_switch": "Moderate switch",
    "major_switch": "Major switch",
}


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "1"}


def safe_text(value: Any, fallback: str = "Not available.") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def fmt_num(value: Any, digits: int = 2, fallback: str = "-") -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return fallback
    return f"{float(numeric):.{digits}f}"


def fmt_pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


def empty_figure(title: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=15, color="#4a5568"),
            )
        ],
    )
    return figure


def kpi_box(label: str, value: str, subtext: str, tone: str = "primary") -> ui.Tag:
    return ui.div(
        ui.div(label, class_="kpi-label"),
        ui.div(value, class_="kpi-value"),
        ui.div(subtext, class_="kpi-sub"),
        class_=f"kpi-box {tone}",
    )


def card(title: str, *body: Any) -> ui.Tag:
    return ui.div(
        ui.div(ui.h3(title), class_="sap-card-header"),
        ui.div(*body, class_="sap-card-body"),
        class_="sap-card",
    )


def page_header(title: str, subtitle: str = "") -> ui.Tag:
    children: list = [ui.h2(title)]
    if subtitle:
        children.append(ui.p(subtitle, class_="subtitle"))
    return ui.div(*children, class_="page-header")


def pending_review_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    spot_check_pairs = select_spot_check_pairs(frame)
    needs_review = (
        frame["routing"].eq("llm")
        | frame["llm_confidence"].eq("low")
        | frame["llm_flag"].map(truthy)
        | frame["published_endpoint"].eq("")
        | frame["pair_id"].isin(spot_check_pairs)
    )
    unresolved = ~frame["human_reviewed"].isin(["yes", "spot_check"])
    queue = frame[needs_review & unresolved].copy()
    queue["similarity_score_numeric"] = pd.to_numeric(queue["similarity_score"], errors="coerce")
    return queue.sort_values(["similarity_score_numeric", "pair_id"], na_position="last").drop(
        columns=["similarity_score_numeric"]
    )


def is_spot_check_row(row: pd.Series, frame: pd.DataFrame) -> bool:
    pair_id = str(row.get("pair_id", "")).strip()
    return bool(pair_id) and pair_id in select_spot_check_pairs(frame)


def default_switch(row: pd.Series) -> str:
    if safe_text(row.get("human_final_class"), ""):
        return str(row["human_final_class"])
    if safe_text(row.get("llm_switch_type"), ""):
        return str(row["llm_switch_type"])
    return "concordant" if row.get("routing") == "auto_concordant" else "major_switch"


def default_poolable(row: pd.Series) -> bool:
    if safe_text(row.get("human_poolable"), ""):
        return truthy(row["human_poolable"])
    return truthy(row.get("llm_comparability"))
