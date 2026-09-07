"""
Poster figures for the SAP Coherence Checker outcome-switching study.

Figure 1, study flow: 461 trial records -> primary-publication linkage ->
309 adjudicated pairs -> five endpoint-adjudication outcomes.

Run:  python make_poster_figures.py
Out:  data/outputs/poster_figures/*.png  (300 dpi)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402  (must follow mpl.use)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path("data/outputs/poster_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---- type -------------------------------------------------------------------
FONT = "Arial" if "Arial" in {f.name for f in fm.fontManager.ttflist} else "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": FONT,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# ---- greyscale, journal-technical ----------------------------------------
SURFACE = "#ffffff"
INK = "#111111"
INK_2 = "#555555"
PROC_EDGE = "#333333"
EXCL_FILL = "#ededed"
EXCL_EDGE = "#bdbdbd"
TERM_FILL = "#ffffff"
TERM_SWITCH_FILL = "#e6e6e6"   # the only tonal cue: the switch group sits on grey
CONNECT = "#555555"


def box(ax, x, y, w, h, *, fill, edge, lw=1.3, radius=0.9):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            mutation_aspect=0.62,
        )
    )


def arrow(ax, x0, y0, x1, y1, *, color=CONNECT, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=15,
            shrinkA=0,
            shrinkB=0,
            color=color,
            linewidth=lw,
            joinstyle="miter",
        )
    )


def elbow(ax, pts, *, color=CONNECT, lw=1.6, arrow_end=True):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        last = (x1, y1) == pts[-1]
        if last and arrow_end:
            arrow(ax, x0, y0, x1, y1, color=color, lw=lw)
        else:
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle="round")


def study_flow():
    fig, ax = plt.subplots(figsize=(6.6, 5.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(13, 99)
    ax.axis("off")

    lw_box, lw_line = 0.9, 1.0
    cx = 43          # central spine (offset left for the exclusion box)
    PW = 52          # process-box width

    def proc(y, h, title, sub, ts=9.5, ss=7.4):
        box(ax, cx, y, PW, h, fill="#ffffff", edge=PROC_EDGE, lw=lw_box, radius=0.6)
        ax.text(cx, y + h / 2 - 2.0, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=INK)
        ax.text(cx, y - h / 2 + 2.0, sub, ha="center", va="center",
                fontsize=ss, color=INK_2)

    proc(93, 8.4, "461 breast-cancer trial records",
         "ClinicalTrials.gov, Phase 2/3, completed, results posted")
    arrow(ax, cx, 88.8, cx, 84.4, lw=lw_line)

    proc(80, 8.0, "Primary-publication linkage",
         "one LLM pass per trial, registered endpoint withheld")

    # spine B -> C with the exclusion branch
    ax.plot([cx, cx], [76.0, 67.4], color=CONNECT, linewidth=lw_line, solid_capstyle="round")
    elbow(ax, [(cx, 72.0), (70.7, 72.0)], lw=lw_line, color=CONNECT)

    box(ax, 84.5, 72.0, 27, 9.6, fill=EXCL_FILL, edge=EXCL_EDGE, lw=lw_box, radius=0.6)
    ax.text(84.5, 74.1, "152 trials", ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(84.5, 70.6, "no confirmed\nprimary-results publication", ha="center",
            va="center", fontsize=6.9, color=INK_2, linespacing=1.2)

    proc(63, 8.0, "309 linked trial / publication pairs",
         "each with a committed primary-results paper")
    arrow(ax, cx, 59.0, cx, 54.6, lw=lw_line)

    proc(50, 8.0, "Endpoint adjudication",
         "registered vs published primary endpoint")

    # --- terminals: uniform white boxes, greyscale ----------------------
    terms = [
        ("249", "Concordant"),
        ("21", "Minor\nmodification"),
        ("5", "Additional\noutcome"),
        ("31", "Moderate\nswitch"),
        ("3", "Major\nswitch"),
    ]
    switch_idx = {3, 4}
    n = len(terms)
    bw, gap = 16.4, 2.6
    span = n * bw + (n - 1) * gap
    x0 = 50 - span / 2
    centers = [x0 + bw / 2 + i * (bw + gap) for i in range(n)]
    box_top = 39.0
    bh = 14.0
    box_ctr = box_top - bh / 2
    rail_y = 43.0

    ax.plot([cx, cx], [45.8, rail_y], color=CONNECT, linewidth=lw_line, solid_capstyle="round")
    ax.plot([centers[0], centers[-1]], [rail_y, rail_y], color=CONNECT,
            linewidth=lw_line, solid_capstyle="round")
    for c in centers:
        arrow(ax, c, rail_y, c, box_top + 0.15, lw=lw_line)

    for i, ((num, label), c) in enumerate(zip(terms, centers)):
        fill = TERM_SWITCH_FILL if i in switch_idx else TERM_FILL
        box(ax, c, box_ctr, bw, bh, fill=fill, edge=PROC_EDGE, lw=lw_box, radius=0.6)
        ax.text(c, box_ctr + 2.9, num, ha="center", va="center", fontsize=15,
                fontweight="bold", color=INK)
        ax.text(c, box_ctr - 3.9, label, ha="center", va="center", fontsize=7.6,
                color=INK_2, linespacing=1.15)

    # --- grouping brackets (all greyscale; emphasis by weight, not colour) ---
    def bracket(x_a, x_b, y, text, weight="normal"):
        ax.plot([x_a, x_a, x_b, x_b], [y + 1.1, y, y, y + 1.1], color=INK_2,
                linewidth=lw_line, solid_capstyle="round")
        ax.text((x_a + x_b) / 2, y - 2.4, text, ha="center", va="center", fontsize=7.6,
                color=INK, fontweight=weight)

    b_y = box_ctr - bh / 2 - 2.2
    bracket(centers[0] - bw / 2, centers[0] + bw / 2, b_y, "endpoint concordant")
    bracket(centers[1] - bw / 2, centers[2] + bw / 2, b_y, "modified, not switched")
    bracket(centers[3] - bw / 2, centers[4] + bw / 2, b_y,
            "outcome switch: 34 pairs (11.0%)", weight="bold")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_study_flow.{ext}", facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.10)
    plt.close(fig)
    print(f"wrote {OUT/'fig1_study_flow.png'} and .pdf")


def _beeswarm(x, y0, half, sep, seed=0):
    """Histogram-free beeswarm: pack points in y so none overlap in (x, y)."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    idx = np.argsort(x, kind="stable")
    xs = x[idx]
    ys = np.zeros(len(xs))
    rng = np.random.default_rng(seed)
    for i in range(len(xs)):
        k = 0
        while True:
            slots = [0.0] if k == 0 else [k * sep, -k * sep]
            for cy in slots:
                if abs(cy) > half:
                    continue
                if all(
                    abs(xs[j] - xs[i]) >= sep or abs(ys[j] - cy) >= sep
                    for j in range(i)
                ):
                    ys[i] = cy
                    break
            else:
                k += 1
                if k > 80:
                    ys[i] = rng.uniform(-half, half)
                    break
                continue
            break
    out = np.zeros(len(xs))
    out[idx] = ys
    return y0 + out


def similarity_beeswarm():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    dl = pd.read_csv("data/logs/decision_log.csv", dtype=str, keep_default_na=False)
    cls = dl["human_final_class"].where(
        dl["human_final_class"].str.strip().ne(""), dl["llm_switch_type"]
    )
    sim = pd.to_numeric(dl["similarity_score"], errors="coerce")
    keep = sim.notna()
    cls, sim = cls[keep].to_numpy(), sim[keep].to_numpy()

    order = [
        ("concordant", "Concordant"),
        ("minor_modification", "Minor modification"),
        ("additional_outcome", "Additional outcome"),
        ("moderate_switch", "Moderate switch"),
        ("major_switch", "Major switch"),
    ]
    rows = list(range(len(order), 0, -1))  # y = 5..1, top to bottom

    y_switch = np.isin(cls, ["moderate_switch", "major_switch"]).astype(int)
    auc = roc_auc_score(y_switch, 1 - sim)
    fpr, tpr, _ = roc_curve(y_switch, 1 - sim)

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [3.4, 1.0], "wspace": 0.30}
    )
    fig.patch.set_facecolor(SURFACE)

    # ================= main panel: beeswarm =================
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, len(order) + 0.7)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([])
    ax.set_xlabel("Cosine similarity of registered vs published primary endpoint",
                  fontsize=8.3, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=INK_2, length=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.grid(axis="x", color="#ebebeb", linewidth=0.6, zorder=0)
    ax.axhline(2.5, color="#d0d0d0", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)

    half = {5: 0.33, 4: 0.24, 3: 0.20, 2: 0.26, 1: 0.18}
    for (key, label), yr in zip(order, rows):
        v = sim[cls == key]
        n = len(v)
        yy = _beeswarm(v, yr, half.get(yr, 0.24), sep=0.0125, seed=yr)
        ax.scatter(v, yy, s=7, facecolor="#8f8f8f", edgecolor="none", alpha=0.7,
                   zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [yr, yr], color=INK, linewidth=1.3, zorder=4)
        for q in (q1, q3):
            ax.plot([q, q], [yr - 0.085, yr + 0.085], color=INK, linewidth=1.3, zorder=4)
        ax.scatter([med], [yr], s=30, facecolor="#ffffff", edgecolor=INK,
                   linewidth=1.3, zorder=5)
        ax.text(-0.02, yr + 0.15, label, ha="right", va="center", fontsize=8.4,
                fontweight="bold", color=INK)
        ax.text(-0.02, yr - 0.16, f"n = {n}", ha="right", va="center", fontsize=7.6,
                color=INK_2)

    # group brackets just outside the right edge
    def rbracket(y0, y1, text):
        xb, tick = 1.012, 0.012
        for seg in ([xb, xb], [y0, y1]), ([xb, xb + tick], [y0, y0]), ([xb, xb + tick], [y1, y1]):
            ax.plot(seg[0], seg[1], color=INK_2, linewidth=0.9, clip_on=False)
        ax.text(xb + 0.03, (y0 + y1) / 2, text, rotation=270, ha="left", va="center",
                fontsize=7.4, color=INK_2, clip_on=False)

    rbracket(2.7, 5.3, "non-switch")
    rbracket(0.72, 2.3, "switch")

    # ================= right panel: ROC =================
    axr.set_facecolor(SURFACE)
    axr.step(fpr, tpr, where="post", color=INK, linewidth=1.3)
    axr.plot([0, 1], [0, 1], color="#b3b3b3", linewidth=0.8, linestyle=(0, (3, 3)))
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)
    axr.set_aspect("equal")
    axr.set_xticks([0, 0.5, 1])
    axr.set_yticks([0, 0.5, 1])
    axr.tick_params(labelsize=6.8, colors=INK_2, length=2)
    axr.set_xlabel("1 − specificity", fontsize=7.2, color=INK_2, labelpad=2)
    axr.set_ylabel("sensitivity", fontsize=7.2, color=INK_2, labelpad=2)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        axr.spines[s].set_color("#999999")
    axr.set_title(f"AUROC {auc:.3f}", fontsize=8.6, color=INK, fontweight="bold", pad=4)
    axr.text(0.5, -0.34, "switch (moderate + major) vs rest\nscore = 1 − cosine similarity",
             ha="center", va="top", fontsize=6.6, color=INK_2, transform=axr.transAxes,
             linespacing=1.3)

    fig.subplots_adjust(left=0.165, right=0.93, top=0.9, bottom=0.19)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_similarity_beeswarm.{ext}", facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.10, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT/'fig2_similarity_beeswarm.png'} and .pdf  (AUROC {auc:.3f})")


def posterior_forest():
    import arviz as az
    import numpy as np
    import pandas as pd

    sc = pd.read_csv("data/outputs/scorecard.csv")
    NAME = {
        "pathological_complete_response": "pCR",
        "event_free_disease_free_survival": "EFS / DFS",
        "overall_survival": "OS",
        "progression_free_survival": "PFS",
        "objective_response_rate": "ORR",
        "other_endpoints": "Other",
    }
    row_order = ["pathological_complete_response", "event_free_disease_free_survival",
                 "overall_survival", "progression_free_survival",
                 "objective_response_rate", "other_endpoints"]
    sc = sc.set_index("endpoint_cluster")

    data = []
    for key in row_order:
        r = sc.loc[key]
        idata = az.from_netcdf(f"data/logs/bayes_traces/trace_cluster_{key}.nc")
        hr = np.exp(idata.posterior["mu"].values.flatten())
        lo50, hi50 = az.hdi(hr, hdi_prob=0.50)
        data.append(
            dict(
                label=NAME[key],
                n=int(r["trials_included"]),
                med=float(r["pooled_hr"]),
                lo=float(r["pooled_hr_cri_lower"]),
                hi=float(r["pooled_hr_cri_upper"]),
                lo50=float(lo50),
                hi50=float(hi50),
                i2=float(r["i_squared_pct"]),
            )
        )

    ys = list(range(len(data), 0, -1))  # top to bottom
    ytop = len(data) + 0.55

    fig, (ax, axt) = plt.subplots(
        1, 2, figsize=(7.7, 3.2), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.04}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---------------- forest ----------------
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xlim(0.38, 1.62)
    ax.set_ylim(0.5, ytop + 0.15)
    ticks = [0.4, 0.5, 0.75, 1.0, 1.5]
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3, colors=INK_2, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.axvline(1.0, color="#9a9a9a", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

    for d, y in zip(data, ys):
        ax.plot([d["lo"], d["hi"]], [y, y], color=INK, linewidth=1.0, zorder=3,
                solid_capstyle="round")
        ax.plot([d["lo50"], d["hi50"]], [y, y], color=INK, linewidth=3.6, zorder=4,
                solid_capstyle="butt")
        ax.scatter([d["med"]], [y], s=22, facecolor="#ffffff", edgecolor=INK,
                   linewidth=1.15, zorder=5)
        ax.text(-0.035, y + 0.13, d["label"], ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=INK, transform=ax.get_yaxis_transform(),
                clip_on=False)
        ax.text(-0.035, y - 0.16, f"n = {d['n']}", ha="right", va="center", fontsize=7.1,
                color=INK_2, transform=ax.get_yaxis_transform(), clip_on=False)

    ax.set_title("Posterior treatment effects by endpoint family",
                 fontsize=9.6, fontweight="bold", color=INK, pad=8, loc="left")
    ax.text(0.0, -0.17, "lower HR", ha="left", va="top", fontsize=7.0, color=INK_2,
            style="italic", transform=ax.transAxes)
    ax.text(1.0, -0.17, "higher HR", ha="right", va="top", fontsize=7.0, color=INK_2,
            style="italic", transform=ax.transAxes)

    # ---------------- right text + I2 bars ----------------
    axt.set_xlim(0, 1)
    axt.set_ylim(ax.get_ylim())
    axt.axis("off")
    axt.text(0.0, ytop, "Posterior HR (95% CrI)", fontsize=7.6, fontweight="bold",
             color=INK_2, va="center")
    axt.text(0.585, ytop, "Heterogeneity  I²", fontsize=7.6, fontweight="bold",
             color=INK_2, va="center")
    for d, y in zip(data, ys):
        axt.text(0.0, y, f"{d['med']:.2f}  ({d['lo']:.2f}–{d['hi']:.2f})",
                 fontsize=8.2, color=INK, va="center")
        frac = d["i2"] / 100.0
        axt.add_patch(Rectangle((0.585, y - 0.15), 0.30, 0.30, facecolor="#e4e4e4",
                                edgecolor="none"))
        axt.add_patch(Rectangle((0.585, y - 0.15), 0.30 * frac, 0.30, facecolor="#8f8f8f",
                                edgecolor="none"))
        axt.text(0.905, y, f"{d['i2']:.1f}%", fontsize=7.6, color=INK, va="center",
                 ha="left")

    fig.subplots_adjust(left=0.135, right=0.985, top=0.83, bottom=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_posterior_forest.{ext}", facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT/'fig3_posterior_forest.png'} and .pdf")


def switching_by_stratum():
    import pandas as pd
    from statsmodels.stats.proportion import proportion_confint

    dl = pd.read_csv("data/logs/decision_log.csv", dtype=str, keep_default_na=False)
    lt = pd.read_csv("data/outputs/linked_trials.csv", dtype=str, keep_default_na=False)
    dl["nct_id"] = dl["pair_id"].str.split("_").str[0]
    cls = dl["human_final_class"].where(
        dl["human_final_class"].str.strip().ne(""), dl["llm_switch_type"]
    )
    dl["switch"] = cls.isin(["moderate_switch", "major_switch"])
    m = dl.merge(
        lt[["nct_id", "bc_subtype", "bc_setting"]].drop_duplicates("nct_id"),
        on="nct_id", how="left",
    )

    groups = [
        ("Disease subtype", "bc_subtype", [
            ("tnbc", "TNBC"), ("hr_positive", "HR+ / HER2−"), ("her2_positive", "HER2+")]),
        ("Treatment setting", "bc_setting", [
            ("metastatic", "Metastatic"), ("neoadjuvant", "Neoadjuvant"), ("adjuvant", "Adjuvant")]),
    ]

    rows = []          # (y, label, k, n, p, lo, hi)
    headers = []       # (y, text)
    y = 8.0
    for gname, col, items in groups:
        headers.append((y, gname))
        y -= 0.95
        for key, lab in items:
            sub = m[m[col] == key]
            k, n = int(sub["switch"].sum()), len(sub)
            p = k / n * 100 if n else 0.0
            lo, hi = proportion_confint(k, n, method="wilson")
            rows.append((y, lab, k, n, p, lo * 100, hi * 100))
            y -= 1.0
        y -= 0.55       # gap between groups
    divider_y = (rows[2][0] + rows[3][0]) / 2

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_xlim(-1.5, 40)
    ax.set_ylim(rows[-1][0] - 0.7, headers[0][0] + 0.6)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(["0", "10", "20", "30", "40"], fontsize=8, color=INK_2)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3, colors=INK_2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.set_xlabel("Outcome-switching prevalence (%)", fontsize=8.4, color=INK)
    ax.grid(axis="x", color="#ececec", linewidth=0.6, zorder=0)
    ax.axhline(divider_y, color="#d0d0d0", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)

    for hy, text in headers:
        ax.text(-0.205, hy, text, transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=8.4, fontweight="bold", color=INK, clip_on=False)

    for yr, lab, k, n, p, lo, hi in rows:
        ax.plot([lo, hi], [yr, yr], color=INK, linewidth=1.0, zorder=3,
                solid_capstyle="round")
        ax.scatter([p], [yr], s=22, facecolor="#ffffff", edgecolor=INK, linewidth=1.15,
                   zorder=5)
        ax.text(-0.03, yr + 0.14, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8.2, fontweight="bold", color=INK, clip_on=False)
        ax.text(-0.03, yr - 0.17, f"{k}/{n}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.1, color=INK_2, clip_on=False)
        pa = "left" if p < 3 else "center"
        ax.text(max(p, 0.4), yr - 0.34, f"{p:.1f}%", ha=pa, va="center", fontsize=7.2,
                color=INK_2)

    ax.set_title("Outcome-switching prevalence by breast-cancer subtype and treatment setting",
                 fontsize=9.4, fontweight="bold", color=INK, pad=9, loc="left")

    fig.subplots_adjust(left=0.16, right=0.975, top=0.86, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_switching_by_stratum.{ext}", facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.09, dpi=300)
    plt.close(fig)
    print(f"wrote {OUT/'fig4_switching_by_stratum.png'} and .pdf")


if __name__ == "__main__":
    study_flow()
    similarity_beeswarm()
    posterior_forest()
    switching_by_stratum()
