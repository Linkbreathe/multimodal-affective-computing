"""Generate core paper figures from existing local artifacts.

The script intentionally reads only local, already-generated artifacts so the
figures are reproducible without re-running the preprocessing pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "reports" / "core_figures_nature_2026-07-04"
PNG_DPI = 300
TIFF_DPI = 600
RNG = np.random.default_rng(20260704)

COND_ORDER = [f"C{i}" for i in range(1, 10)]
PALETTE = {
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "green": "#2E9E44",
    "green_soft": "#DDF3DE",
    "purple": "#9A4D8E",
    "purple_soft": "#E4CCD8",
    "gray": "#767676",
    "light_gray": "#D8D8D8",
    "dark": "#272727",
    "off_black": "#4D4D4D",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.labelcolor": PALETTE["dark"],
        "xtick.color": PALETTE["dark"],
        "ytick.color": PALETTE["dark"],
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    stem: str,
    manifest: list[dict[str, str]],
    title: str,
    paper_position: str,
    sources: Iterable[Path],
    notes: str,
) -> None:
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    pdf = OUT / f"{stem}.pdf"
    tiff = OUT / f"{stem}.tiff"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(tiff, dpi=TIFF_DPI, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "figure": stem,
            "title": title,
            "paper_position": paper_position,
            "svg": rel(svg),
            "png": rel(png),
            "pdf": rel(pdf),
            "tiff": rel(tiff),
            "sources": "; ".join(rel(p) for p in sources),
            "notes": notes,
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color="#EEEEEE", linewidth=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7, width=0.7, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["dark"],
    )


def add_panel_labels(axes: Iterable[plt.Axes], labels: str = "abcdefghijklmnopqrstuvwxyz") -> None:
    for ax, label in zip(axes, labels):
        add_panel_label(ax, label)


def ordered_condition_label_map(df: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for cond in COND_ORDER:
        part = df.loc[df["condition"].eq(cond)]
        if part.empty:
            labels[cond] = cond
            continue
        intensity = pick_first(part, ["intensity_level_cond", "intensity_cond", "intensity"])
        frequency = pick_first(part, ["frequency_level_cond", "frequency_cond", "frequency"])
        if intensity is None or frequency is None:
            labels[cond] = cond
        else:
            labels[cond] = f"{cond}\nI={short_value(intensity)}\nF={short_value(frequency)}"
    return labels


def condition_code_labels() -> dict[str, str]:
    return {cond: cond for cond in COND_ORDER}


def condition_key_text() -> str:
    return "C1-C9 I/F order: L/L, L/M, L/H, M/L, M/M, M/H, H/L, H/M, H/H"


def pick_first(df: pd.DataFrame, columns: list[str]):
    for col in columns:
        if col in df.columns:
            values = df[col].dropna()
            if not values.empty:
                return values.iloc[0]
    return None


def short_value(value) -> str:
    if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
        return f"{float(value):.2g}"
    text = str(value)
    return text.replace("Medium", "Med")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def boxplot_by_condition(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    color: str,
    labels: dict[str, str] | None = None,
    log_y: bool = False,
) -> None:
    data: list[np.ndarray] = []
    positions: list[int] = []
    for pos, cond in enumerate(COND_ORDER, start=1):
        values = numeric_series(df.loc[df["condition"].eq(cond)], column).dropna()
        if log_y:
            values = values.loc[values > 0]
        data.append(values.to_numpy())
        positions.append(pos)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 0.9},
        boxprops={"linewidth": 0.75, "edgecolor": PALETTE["dark"]},
        whiskerprops={"linewidth": 0.75, "color": PALETTE["dark"]},
        capprops={"linewidth": 0.75, "color": PALETTE["dark"]},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.24)

    for pos, values in zip(positions, data):
        if len(values) == 0:
            continue
        jitter = RNG.normal(0.0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=8,
            color=color,
            alpha=0.55,
            linewidths=0,
        )

    ax.set_title(title, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels.get(c, c) if labels else c for c in COND_ORDER], fontsize=5.8)
    if log_y:
        ax.set_yscale("log")
    style_axis(ax)


def read_condition_delta() -> pd.DataFrame:
    path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    cols = {
        "participant_id",
        "condition",
        "intensity_cond",
        "frequency_cond",
        "intensity_level_cond",
        "frequency_level_cond",
        "relaxation_cond",
        "discomfort_cond",
        "calm_cond",
        "ecg_hr_bpm__mean_cond",
        "median_alpha_power_channel_mean_cond",
        "median_beta_power_channel_mean_cond",
        "delta_ecg_hr_bpm__mean",
    }
    return pd.read_csv(path, usecols=lambda col: col in cols)


def figure_01_dose_summary(cond_df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    labels = condition_code_labels()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.6), constrained_layout=True)
    specs = [
        ("relaxation_cond", "Relaxation", "Label value", PALETTE["blue"], False),
        ("discomfort_cond", "Discomfort", "Label value", PALETTE["red"], False),
        ("calm_cond", "Calm", "Label value", PALETTE["green"], False),
        ("ecg_hr_bpm__mean_cond", "Heart rate", "BPM", PALETTE["orange"], False),
        (
            "median_alpha_power_channel_mean_cond",
            "Alpha power",
            "Median channel power",
            PALETTE["purple"],
            True,
        ),
    ]
    for ax, spec in zip(axes.flat, specs):
        boxplot_by_condition(
            ax,
            cond_df,
            column=spec[0],
            ylabel=spec[2],
            title=spec[1],
            color=spec[3],
            labels=labels,
            log_y=spec[4],
        )
    key_ax = axes.flat[-1]
    key_ax.axis("off")
    key_ax.text(
        0.02,
        0.70,
        "Condition key",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="center",
        color=PALETTE["dark"],
    )
    key_ax.text(
        0.02,
        0.48,
        "C1-C9 intensity/frequency order:\n"
        "L/L, L/M, L/H\n"
        "M/L, M/M, M/H\n"
        "H/L, H/M, H/H",
        fontsize=7,
        ha="left",
        va="center",
        color=PALETTE["off_black"],
        linespacing=1.35,
    )
    add_panel_labels(list(axes.flat)[:5])
    fig.suptitle("Dose response summary by condition", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_01_dose_response_3x3_summary",
        manifest,
        "Dose response 3x3 summary",
        "sec5.1.7 fig:res:study1",
        [ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"],
        "Participant-condition distributions; alpha is log-scaled because values are strictly positive and skewed.",
    )


def figure_02_03_eeg_boxes(cond_df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    labels = condition_code_labels()
    for stem, title, column, color, position in [
        (
            "fig_02_alpha_by_condition_boxplot",
            "Alpha power by condition",
            "median_alpha_power_channel_mean_cond",
            PALETTE["purple"],
            "sec5.1.2",
        ),
        (
            "fig_03_beta_by_condition_boxplot",
            "Beta power by condition",
            "median_beta_power_channel_mean_cond",
            PALETTE["teal"],
            "sec5.1.3",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        boxplot_by_condition(
            ax,
            cond_df,
            column=column,
            ylabel="Median channel power",
            title=title,
            color=color,
            labels=labels,
            log_y=True,
        )
        total_n = int(numeric_series(cond_df, column).notna().sum())
        ax.text(
            0.01,
            0.98,
            f"Usable participant-condition rows: n={total_n}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            color=PALETTE["dark"],
        )
        save_figure(
            fig,
            stem,
            manifest,
            title,
            position,
            [ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"],
            "Uses EEG-gated condition-level artifact; log-scaled positive power values.",
        )


def figure_04_hr_phase(cond_df: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    phase_path = ROOT / "artifacts" / "phase_baseline_v1" / "condition_phase_delta_features.csv"
    stats_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "baseline_corrected_delta_factorial.csv"
    phase = pd.read_csv(phase_path, usecols=["participant_id", "condition", "ecg_hr_bpm__delta"])
    participant_means = phase.groupby("participant_id")["ecg_hr_bpm__delta"].mean().sort_index()
    cond_delta = cond_df[["condition", "delta_ecg_hr_bpm__mean"]].copy()
    stats = pd.read_csv(stats_path)
    hr_stats = stats.loc[stats["dv"].eq("delta_ecg_hr_bpm__mean"), ["source", "p_value"]]
    p_text = ", ".join(f"{row.source} p={row.p_value:.3f}" for row in hr_stats.itertuples())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    ax = axes[0]
    ax.axhline(0, color=PALETTE["dark"], linewidth=1)
    ax.boxplot(
        [participant_means.to_numpy()],
        positions=[1],
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 0.9},
        boxprops={"facecolor": PALETTE["orange"], "alpha": 0.25, "edgecolor": PALETTE["dark"], "linewidth": 0.75},
    )
    jitter = RNG.normal(0, 0.035, size=len(participant_means))
    ax.scatter(np.ones(len(participant_means)) + jitter, participant_means, color=PALETTE["orange"], s=18)
    ax.set_xticks([1])
    ax.set_xticklabels(["Start phase\nparticipant means"])
    ax.set_ylabel("Delta HR (BPM)")
    ax.set_title("Initial HR drop")
    mean_delta = participant_means.mean()
    median_delta = participant_means.median()
    ax.text(
        0.05,
        0.08,
        f"Mean={mean_delta:.2f} BPM\nMedian={median_delta:.2f} BPM\nParticipants={len(participant_means)}",
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
    )
    style_axis(ax)

    ax = axes[1]
    labels = {cond: cond for cond in COND_ORDER}
    boxplot_by_condition(
        ax,
        cond_delta,
        "delta_ecg_hr_bpm__mean",
        "Delta HR (BPM)",
        "Condition-level dose effects",
        PALETTE["blue"],
        labels=labels,
        log_y=False,
    )
    ax.axhline(0, color=PALETTE["dark"], linewidth=1)
    ax.text(0.02, 0.03, p_text, transform=ax.transAxes, fontsize=8, va="bottom")
    add_panel_labels(axes)

    save_figure(
        fig,
        "fig_04_hr_phase_drop_vs_dose_flat",
        manifest,
        "HR initial drop versus dose-flat condition effects",
        "sec5.1.5",
        [phase_path, stats_path, ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"],
        "Left uses phase-baseline participant means; right uses baseline-corrected condition deltas and factorial p-values.",
    )


def figure_05_label_distribution(manifest: list[dict[str, str]]) -> None:
    counts_path = ROOT / "artifacts" / "reports" / "label_distribution" / "tables" / "label_value_counts.csv"
    counts = pd.read_csv(counts_path)
    targets = ["relaxation", "discomfort"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), constrained_layout=True, sharey=True)
    for ax, label, color in zip(axes, targets, [PALETTE["blue"], PALETTE["red"]]):
        part = counts.loc[counts["label"].eq(label)].copy()
        ax.bar(part["normalized_value"], part["count"], width=0.055, color=color, alpha=0.82)
        ax.set_title(label.capitalize())
        ax.set_xlabel("Normalized label value")
        ax.set_ylabel("Count")
        ax.set_xlim(-0.05, 1.05)
        style_axis(ax)
        if label == "discomfort":
            floor = int(part.loc[part["normalized_value"].eq(0.0), "count"].sum())
            total = int(part["count"].sum())
            ax.text(
                0.03,
                0.94,
                f"Floor at 0: {floor}/{total}\n({floor / total:.1%})",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
    add_panel_labels(axes)
    fig.suptitle("Label floor effects at participant-condition level", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_05_label_floor_histograms",
        manifest,
        "Label distribution histograms",
        "sec5.1.6",
        [counts_path],
        "Shows relaxation and discomfort label counts; discomfort has a strong zero floor.",
    )


def get_random_band(random_metrics: dict, target: str, metric: str) -> tuple[float, float, float]:
    if target == "high_discomfort":
        key_map = {
            "roc_auc": "uniform_roc_auc_ci",
            "pr_auc": "uniform_pr_auc_ci",
        }
        band = random_metrics["random"]["binary_high_discomfort"][key_map[metric]]
    else:
        key_map = {
            "within_participant_spearman_mean": "within_mean_ci",
            "pairwise_ranking_accuracy": "pairwise_ci",
            "top1_condition_hit": "top1_ci",
            "top1_regret": "regret_ci",
        }
        band = random_metrics["random"]["continuous"][target]["uniform"][key_map[metric]]
    return float(band[0]), float(band[1]), float(band[2])


def draw_forest_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    x_label: str,
    xlim: tuple[float, float],
) -> None:
    y_positions = np.arange(len(rows), 0, -1)
    for y, row in zip(y_positions, rows):
        mean = float(row["random_mean"])
        lo = float(row["random_lo"])
        hi = float(row["random_hi"])
        ax.hlines(y, lo, hi, color=PALETTE["light_gray"], linewidth=6, zorder=1)
        ax.plot(mean, y, "|", color=PALETTE["gray"], markersize=12, markeredgewidth=1.4, zorder=2)
        for name, value, marker, color in row["points"]:
            ax.scatter(value, y, marker=marker, s=30, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["label"]) for row in rows])
    ax.set_xlim(*xlim)
    ax.set_xlabel(x_label)
    style_axis(ax, grid_axis="x")


def figure_06_random_floor_forest(manifest: list[dict[str, str]]) -> None:
    rand_path = ROOT / "artifacts" / "random_baseline" / "random_baseline_metrics.json"
    cont_path = ROOT / "artifacts" / "fusion_four_channel_ranking" / "continuous_ranking_metrics.csv"
    bin_path = ROOT / "artifacts" / "fusion_four_channel_ranking" / "binary_high_discomfort_metrics.csv"
    random_metrics = json.loads(rand_path.read_text(encoding="utf-8"))
    cont = pd.read_csv(cont_path)
    binary = pd.read_csv(bin_path)

    cont_targets = ["relaxation", "pleasantness", "calm"]
    left_rows: list[dict[str, object]] = []
    right_rows: list[dict[str, object]] = []
    for target in cont_targets:
        target_rows = cont.loc[cont["target"].eq(target)].copy()
        mean, lo, hi = get_random_band(random_metrics, target, "within_participant_spearman_mean")
        points = []
        for baseline, marker, color in [
            ("condition_only", "s", PALETTE["blue"]),
            ("history", "^", PALETTE["orange"]),
        ]:
            vals = target_rows.loc[target_rows["baseline"].eq(baseline), "within_participant_spearman_mean"]
            if not vals.empty:
                points.append((baseline, float(vals.iloc[0]), marker, color))
        combos = target_rows.loc[target_rows["record_type"].eq("combination")].dropna(
            subset=["within_participant_spearman_mean"]
        )
        if not combos.empty:
            best = combos.loc[combos["within_participant_spearman_mean"].idxmax()]
            points.append(("best_combo", float(best["within_participant_spearman_mean"]), "D", PALETTE["dark"]))
        left_rows.append(
            {
                "label": target,
                "random_mean": mean,
                "random_lo": lo,
                "random_hi": hi,
                "points": points,
            }
        )

        mean, lo, hi = get_random_band(random_metrics, target, "pairwise_ranking_accuracy")
        points = []
        for baseline, marker, color in [
            ("condition_only", "s", PALETTE["blue"]),
            ("history", "^", PALETTE["orange"]),
        ]:
            vals = target_rows.loc[target_rows["baseline"].eq(baseline), "pairwise_ranking_accuracy"]
            if not vals.empty:
                points.append((baseline, float(vals.iloc[0]), marker, color))
        combos = target_rows.loc[target_rows["record_type"].eq("combination")].dropna(
            subset=["pairwise_ranking_accuracy"]
        )
        if not combos.empty:
            best = combos.loc[combos["pairwise_ranking_accuracy"].idxmax()]
            points.append(("best_combo", float(best["pairwise_ranking_accuracy"]), "D", PALETTE["dark"]))
        right_rows.append(
            {
                "label": f"{target}\npairwise",
                "random_mean": mean,
                "random_lo": lo,
                "random_hi": hi,
                "points": points,
            }
        )

    mean, lo, hi = get_random_band(random_metrics, "high_discomfort", "roc_auc")
    points = []
    for baseline, marker, color in [
        ("condition_only", "s", PALETTE["blue"]),
        ("history", "^", PALETTE["orange"]),
    ]:
        vals = binary.loc[binary["baseline"].eq(baseline), "roc_auc"]
        if not vals.empty:
            points.append((baseline, float(vals.iloc[0]), marker, color))
    combos = binary.loc[binary["record_type"].eq("combination")].dropna(subset=["roc_auc"])
    if not combos.empty:
        best = combos.loc[combos["roc_auc"].idxmax()]
        points.append(("best_model", float(best["roc_auc"]), "D", PALETTE["dark"]))
    right_rows.append(
        {
            "label": "high discomfort\nROC AUC",
            "random_mean": mean,
            "random_lo": lo,
            "random_hi": hi,
            "points": points,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    draw_forest_panel(axes[0], left_rows, "Within-participant Spearman mean", (-0.35, 0.55))
    axes[0].set_title("Continuous targets: within-participant rho")
    draw_forest_panel(axes[1], right_rows, "Pairwise accuracy / ROC AUC", (0.30, 0.75))
    axes[1].set_title("Ranking and binary metrics")
    handles = [
        plt.Line2D([0], [0], color=PALETTE["light_gray"], linewidth=6, label="Random 95% band"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=PALETTE["blue"], label="Condition only"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor=PALETTE["orange"], label="History"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["dark"], label="Best combo/model"),
    ]
    add_panel_labels(axes)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.24, top=0.78, wspace=0.30)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.03), fontsize=7)
    fig.suptitle("Model scores against random-floor uncertainty bands", fontsize=10, fontweight="bold", y=0.95)
    save_figure(
        fig,
        "fig_06_random_floor_forest",
        manifest,
        "Random-floor forest plot",
        "sec5.2.9 tab:res:random",
        [rand_path, cont_path, bin_path],
        "Random bands are Monte Carlo 95% intervals; points are observed baselines or best combinations from ranking artifacts.",
    )


def figure_07_fusion_ablation(manifest: list[dict[str, str]]) -> None:
    metrics_path = ROOT / "artifacts" / "fusion_minimal" / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    combos = metrics.loc[metrics["record_type"].eq("combination")].copy()
    combos["mean_mae"] = combos[["relaxation_mae", "discomfort_mae"]].mean(axis=1)
    combos = combos.sort_values("mean_mae", ascending=True)
    y = np.arange(len(combos))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.1), sharey=True)
    for ax, target, color in [
        (axes[0], "relaxation_mae", PALETTE["blue"]),
        (axes[1], "discomfort_mae", PALETTE["red"]),
    ]:
        ax.barh(y, combos[target], color=color, alpha=0.72, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(target.replace("_", " ").upper())
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(combos["combination"])
        for baseline, line_color, linestyle in [
            ("condition_only", PALETTE["dark"], "-"),
            ("history", PALETTE["orange"], "--"),
            ("random_uniform", PALETTE["gray"], ":"),
        ]:
            value = float(metrics.loc[metrics["baseline"].eq(baseline), target].iloc[0])
            ax.axvline(value, color=line_color, linestyle=linestyle, linewidth=1.0)
        style_axis(ax, grid_axis="x")
    axes[0].set_title("Relaxation MAE")
    axes[1].set_title("Discomfort MAE")
    handles = [
        plt.Line2D([0], [0], color=PALETTE["dark"], linewidth=1.0, label="Condition only"),
        plt.Line2D([0], [0], color=PALETTE["orange"], linewidth=1.0, linestyle="--", label="History"),
        plt.Line2D([0], [0], color=PALETTE["gray"], linewidth=1.0, linestyle=":", label="Random uniform"),
    ]
    add_panel_labels(axes)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.12, top=0.86, wspace=0.08)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.00), fontsize=7)
    fig.suptitle("Minimal fusion and ablation benchmark", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_07_fusion_ablation_mae_bars",
        manifest,
        "Fusion and ablation MAE bars",
        "sec5.2.4/5.2.6 tab:res:fusion",
        [metrics_path],
        "Ridge minimal-fusion artifact; lower MAE is better; vertical lines show three baselines.",
    )


def figure_08_within_between(manifest: list[dict[str, str]]) -> None:
    metrics_path = ROOT / "artifacts" / "reports" / "condition_level_lopo_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = [
        ("Between participants\nLOPO", metrics["metrics"]["targets"]["relaxation"]["spearman"]),
        (
            "Within participant\nafter 2 anchors",
            metrics["metrics"]["personalized_calibration"]["2"]["relaxation_spearman"],
        ),
        (
            "Within participant\nafter 3 anchors",
            metrics["metrics"]["personalized_calibration"]["3"]["relaxation_spearman"],
        ),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    x = np.arange(len(values))
    vals = [float(v[1]) for v in values]
    ax.bar(x, vals, color=[PALETTE["gray"], PALETTE["teal"], PALETTE["blue"]], alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=PALETTE["dark"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in values])
    ax.set_ylabel("Relaxation Spearman rho")
    ax.set_ylim(min(-0.05, min(vals) - 0.04), max(vals) + 0.08)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.015, f"{val:.3f}", ha="center", fontsize=7)
    ax.set_title("Between-participant versus personalized signal")
    style_axis(ax)
    save_figure(
        fig,
        "fig_08_within_vs_between_spearman",
        manifest,
        "Within-participant versus between-participant Spearman",
        "sec5.2.5",
        [metrics_path],
        "Summary-level version; participant-wise scatter would require raw per-participant calibration curves.",
    )


def load_decisions() -> pd.DataFrame:
    decision_root = ROOT / "artifacts" / "realtime" / "adaptive_control"
    records: list[dict[str, object]] = []
    for path in sorted(decision_root.glob("*/decisions.jsonl")):
        session = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                decision = item.get("decision", {})
                window_end = item.get("window_end_ms")
                issued = decision.get("issued_unix_ms")
                latency = None
                if isinstance(window_end, (int, float)) and isinstance(issued, (int, float)):
                    latency = float(issued) - float(window_end)
                records.append(
                    {
                        "session": session,
                        "cycle_index": decision.get("cycle_index"),
                        "issued_unix_ms": issued,
                        "latency_ms": latency,
                        "action": decision.get("action"),
                        "current_condition": decision.get("current_condition"),
                        "target_condition": decision.get("target_condition"),
                    }
                )
    return pd.DataFrame.from_records(records)


def figure_09_latency(decisions: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    source_paths = sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/decisions.jsonl"))
    latency = pd.to_numeric(decisions["latency_ms"], errors="coerce").dropna()
    latency = latency.loc[latency >= 0]
    p95 = float(np.percentile(latency, 95)) if len(latency) else math.nan
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    sorted_latency = np.sort(latency.to_numpy())
    y = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
    axes[0].plot(sorted_latency + 1, y, color=PALETTE["blue"], linewidth=1.4)
    axes[0].axvline(10000 + 1, color=PALETTE["red"], linestyle="--", linewidth=1.0)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Latency + 1 ms (log scale)")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("Latency ECDF")
    axes[0].text(0.04, 0.08, f"n={len(latency)}\np95={p95:.1f} ms", transform=axes[0].transAxes, fontsize=7)
    style_axis(axes[0], grid_axis="both")

    bins = np.logspace(0, math.log10(max(10001, sorted_latency.max() + 1)), 28)
    axes[1].hist(latency + 1, bins=bins, color=PALETTE["teal"], alpha=0.78)
    axes[1].axvline(10000 + 1, color=PALETTE["red"], linestyle="--", linewidth=1.0, label="10 s budget")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Latency + 1 ms (log scale)")
    axes[1].set_ylabel("Window count")
    axes[1].set_title("Latency distribution")
    axes[1].legend(frameon=False)
    style_axis(axes[1], grid_axis="both")
    add_panel_labels(axes)
    fig.suptitle("Adaptive replay latency against 10 s budget", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_09_latency_ecdf_loghist",
        manifest,
        "Latency ECDF and log histogram",
        "sec5.2.8 tab:res:latency",
        source_paths,
        "Computed directly from adaptive_control decisions.jsonl files as issued_unix_ms minus window_end_ms.",
    )


def condition_index(value) -> float:
    if not isinstance(value, str) or not value.startswith("C"):
        return math.nan
    try:
        return float(value[1:])
    except ValueError:
        return math.nan


def figure_10_adaptive_trajectory(decisions: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    if decisions.empty:
        return
    work = decisions.copy()
    work["plot_condition"] = np.where(
        work["target_condition"].notna(),
        work["target_condition"],
        work["current_condition"],
    )
    work["condition_index"] = work["plot_condition"].map(condition_index)
    session_scores = []
    for session, part in work.groupby("session"):
        changed = part["condition_index"].dropna().diff().fillna(0).ne(0).sum()
        session_scores.append((changed, len(part), session))
    _, _, session = max(session_scores)
    part = work.loc[work["session"].eq(session)].sort_values(["issued_unix_ms", "cycle_index"]).copy()
    part = part.dropna(subset=["condition_index", "issued_unix_ms"])
    part["time_min"] = (part["issued_unix_ms"] - part["issued_unix_ms"].min()) / 60000.0

    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    ax.step(part["time_min"], part["condition_index"], where="post", color=PALETTE["blue"], linewidth=1.4)
    apply_part = part.loc[part["action"].eq("apply")]
    hold_part = part.loc[part["action"].ne("apply")]
    if not apply_part.empty:
        ax.scatter(apply_part["time_min"], apply_part["condition_index"], color=PALETTE["green"], s=18, label="apply")
    if not hold_part.empty:
        ax.scatter(hold_part["time_min"], hold_part["condition_index"], color=PALETTE["orange"], s=12, label="hold/other")
    ax.set_yticks(range(1, 10))
    ax.set_ylabel("Target/current condition index")
    ax.set_xlabel("Replay time (min)")
    ax.set_title(f"Adaptive condition trajectory: {session[:8]}...")
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    source = ROOT / "artifacts" / "realtime" / "adaptive_control" / session / "decisions.jsonl"
    save_figure(
        fig,
        "fig_10_adaptive_condition_trajectory",
        manifest,
        "Adaptive condition trajectory",
        "sec5.3.4",
        [source],
        "Automatically selected the adaptive_control replay with the most condition changes.",
    )


def figure_12_ecg_repair(manifest: list[dict[str, str]]) -> None:
    legacy_path = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"
    neurokit_path = ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv"
    usecols = ["participant_id", "condition", "ecg_hr_bpm__mean", "ecg_hr_bpm__median"]
    legacy = pd.read_csv(legacy_path, usecols=lambda col: col in usecols).rename(
        columns={"ecg_hr_bpm__mean": "legacy_hr_mean", "ecg_hr_bpm__median": "legacy_hr_median"}
    )
    neuro = pd.read_csv(neurokit_path, usecols=lambda col: col in usecols).rename(
        columns={"ecg_hr_bpm__mean": "neurokit_hr_mean", "ecg_hr_bpm__median": "neurokit_hr_median"}
    )
    paired = legacy.merge(neuro, on=["participant_id", "condition"], how="inner")
    for col in ["legacy_hr_mean", "neurokit_hr_mean", "legacy_hr_median", "neurokit_hr_median"]:
        paired[col] = pd.to_numeric(paired[col], errors="coerce")
    paired = paired.dropna(subset=["legacy_hr_mean", "neurokit_hr_mean"])
    paired["ratio"] = paired["legacy_hr_mean"] / paired["neurokit_hr_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    data = [paired["legacy_hr_mean"].to_numpy(), paired["neurokit_hr_mean"].to_numpy()]
    bp = axes[0].boxplot(
        data,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 0.9},
    )
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Legacy", "NeuroKit"])
    for patch, color in zip(bp["boxes"], [PALETTE["red"], PALETTE["blue"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    for i, values in enumerate(data, start=1):
        jitter = RNG.normal(0, 0.035, len(values))
        axes[0].scatter(np.full(len(values), i) + jitter, values, s=7, alpha=0.38, color=PALETTE["dark"])
    axes[0].set_ylabel("Condition HR mean (BPM)")
    axes[0].set_title("HR distribution before/after repair")
    style_axis(axes[0])

    ratio = paired["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(ratio, bins=28, color=PALETTE["teal"], alpha=0.82)
    axes[1].axvline(1.0, color=PALETTE["dark"], linewidth=0.9)
    axes[1].axvline(3.0, color=PALETTE["orange"], linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Legacy HR / NeuroKit HR")
    axes[1].set_ylabel("Participant-condition count")
    axes[1].set_title("Inflation ratio")
    axes[1].text(
        0.98,
        0.94,
        f"n={len(paired)}\nmedian ratio={ratio.median():.2f}\n>3x: {(ratio > 3).sum()}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    style_axis(axes[1])
    add_panel_labels(axes)
    fig.suptitle("ECG HR repair: legacy extractor versus NeuroKit", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_12_ecg_neurokit_repair_distribution",
        manifest,
        "ECG repair before/after distribution",
        "sec4.1.2",
        [legacy_path, neurokit_path],
        "Matched participant-condition HR means; compares legacy video_ml features with NeuroKit recompute features.",
    )


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.22, height: float = 0.16):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.12,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.2, color=PALETTE["dark"])
    return patch


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#565656") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


def figure_c1_architecture(manifest: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ((0.05, 0.60), "Raw streams\nEEG / ECG / Head\nEye / Video", PALETTE["blue"]),
        ((0.30, 0.60), "10 s windows\nfeature extraction\nQC gates", PALETTE["teal"]),
        ((0.55, 0.60), "Participant-condition\nlabels\nn=135", PALETTE["green"]),
        ((0.30, 0.24), "Stage 1\nOffline evidence\nLOPO + random floor", PALETTE["purple"]),
        ((0.55, 0.24), "Stage 2\nShadow replay\nstate estimator + policy", PALETTE["orange"]),
        ((0.78, 0.42), "Unity adaptive\ncondition profile\nC1-C9", PALETTE["red"]),
    ]
    for xy, text, color in boxes:
        add_box(ax, xy, text, color, width=0.19, height=0.16)
    add_arrow(ax, (0.24, 0.68), (0.30, 0.68))
    add_arrow(ax, (0.49, 0.68), (0.55, 0.68))
    add_arrow(ax, (0.395, 0.60), (0.395, 0.40))
    add_arrow(ax, (0.60, 0.60), (0.46, 0.40))
    add_arrow(ax, (0.49, 0.32), (0.55, 0.32))
    add_arrow(ax, (0.74, 0.32), (0.78, 0.50))
    ax.text(
        0.50,
        0.90,
        "Two-stage architecture: offline validation before adaptive-control use",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.06,
        "Window slices support feature extraction and replay; participant-condition labels remain the supervised unit.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    save_figure(
        fig,
        "fig_C1_two_stage_architecture",
        manifest,
        "Two-stage architecture schematic",
        "fig:intro:overview",
        [],
        "Schematic drawn from repository pipeline structure and existing artifact contract.",
    )


def figure_c2_closed_loop(manifest: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "Sensors\nLSL streams": (0.395, 0.79),
        "Windowed\nfeatures": (0.70, 0.62),
        "State\nestimator": (0.70, 0.34),
        "Adaptive\npolicy": (0.395, 0.15),
        "UDP command\nhold/apply": (0.09, 0.34),
        "Unity stimulus\nC1-C9": (0.09, 0.62),
    }
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["purple"], PALETTE["orange"], PALETTE["red"], PALETTE["green"]]
    for (text, xy), color in zip(nodes.items(), colors):
        add_box(ax, xy, text, color, width=0.21, height=0.14)

    loop_edges = [
        ((0.605, 0.86), (0.70, 0.71)),
        ((0.805, 0.62), (0.805, 0.48)),
        ((0.70, 0.39), (0.605, 0.23)),
        ((0.395, 0.22), (0.30, 0.39)),
        ((0.195, 0.48), (0.195, 0.62)),
        ((0.30, 0.71), (0.395, 0.86)),
    ]
    for start, end in loop_edges:
        add_arrow(ax, start, end)

    add_box(ax, (0.385, 0.47), "Safety gate\nshadow-first\nbudget checks", PALETTE["dark"], width=0.23, height=0.13)
    add_arrow(ax, (0.70, 0.44), (0.615, 0.535), color="#777777")
    add_arrow(ax, (0.50, 0.47), (0.50, 0.29), color="#777777")
    ax.text(0.50, 0.94, "Adaptive closed-loop replay schematic", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        0.50,
        0.05,
        "The controller can emit hold or apply decisions; current artifacts support shadow replay verification.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    save_figure(
        fig,
        "fig_C2_adaptive_closed_loop",
        manifest,
        "Adaptive closed-loop schematic",
        "fig:exp:loop",
        [],
        "Schematic rendered as a figure asset; statistical evidence comes from separate replay artifacts.",
    )


def write_manifest(manifest: list[dict[str, str]]) -> None:
    manifest_path = OUT / "figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["figure", "title", "paper_position", "svg", "png", "pdf", "tiff", "sources", "notes"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def write_figure_contract() -> None:
    contract = """# Figure Contract

Core conclusion:
The manuscript figures should show that condition-level subjective and physiological effects are weak or bounded, while model and adaptive-control claims must be interpreted against random-floor uncertainty and replay safety evidence.

Figure archetype:
Quantitative grid for statistical result panels, plus schematic-led composites for the introduction and closed-loop workflow.

Target journal/output:
Nature-family/high-impact manuscript figures; Python/matplotlib backend only; editable SVG primary output, PDF vector copy, PNG preview, and 600 dpi TIFF preview.

Panel map:
- fig_01 to fig_05: RQ1 dose response, physiology, and label floor evidence.
- fig_06: Hero random-floor uncertainty panel.
- fig_07: RQ2 fusion/ablation benchmark panel.
- fig_08 to fig_10: Personalized signal, latency, and adaptive replay support.
- fig_12: ECG repair validation.
- fig_C1 and fig_C2: Architecture and closed-loop schematics.

Evidence hierarchy:
- Hero evidence: fig_06 random-floor forest plot.
- Primary support: fig_01 dose response summary and fig_07 fusion/ablation bars.
- Validation/support: fig_04, fig_05, fig_08, fig_09, fig_10, fig_12.
- Schematic context: fig_C1 and fig_C2.

Reviewer risks:
- EEG alpha reactivity is not plotted because no open/closed-eye artifact was found.
- ECG repair panel reflects matched legacy versus NeuroKit artifacts in this checkout; any prose claiming broader 3-5x inflation should be checked against source evidence.
- Some panels are summary-level figures; participant-wise scatter or ECDF variants require the underlying raw/session-level data when not already present.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")


def main() -> None:
    ensure_out()
    manifest: list[dict[str, str]] = []
    cond_df = read_condition_delta()

    figure_01_dose_summary(cond_df, manifest)
    figure_02_03_eeg_boxes(cond_df, manifest)
    figure_04_hr_phase(cond_df, manifest)
    figure_05_label_distribution(manifest)
    figure_06_random_floor_forest(manifest)
    figure_07_fusion_ablation(manifest)
    figure_08_within_between(manifest)
    decisions = load_decisions()
    figure_09_latency(decisions, manifest)
    figure_10_adaptive_trajectory(decisions, manifest)
    figure_12_ecg_repair(manifest)
    figure_c1_architecture(manifest)
    figure_c2_closed_loop(manifest)
    write_manifest(manifest)
    write_figure_contract()
    print(f"Wrote {len(manifest)} figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
