"""Generate thesis-priority manuscript figures from local artifacts.

The figures are designed for the current thesis revision request. The script
does not fabricate unavailable validation data: missing eye-open/closed alpha
and raw ECG R-peak trace panels are explicitly marked as artifact gaps.
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "reports" / "thesis_priority_figures_2026-07-04"
SOURCE_OUT = OUT / "source_data"
PNG_DPI = 300
TIFF_DPI = 600
RNG = np.random.default_rng(20260704)

COND_ORDER = [f"C{i}" for i in range(1, 10)]
FREQ_LABELS = ["Low", "Medium", "High"]
INTENSITY_LABELS = ["Low", "Medium", "High"]

PALETTE = {
    "blue": "#2F6DB3",
    "blue_soft": "#C7D9EE",
    "teal": "#2B9AA0",
    "teal_soft": "#C7E7E5",
    "orange": "#D9852B",
    "orange_soft": "#F3D7B8",
    "red": "#B54545",
    "red_soft": "#F0C5C0",
    "green": "#2A8C4A",
    "green_soft": "#CFE8D6",
    "purple": "#7B4BB2",
    "purple_soft": "#D8C8EA",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
    "very_light_gray": "#F1F1F1",
    "dark": "#2C2C2C",
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
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def save_source(stem: str, df: pd.DataFrame) -> Path:
    path = SOURCE_OUT / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def save_figure(
    fig: plt.Figure,
    stem: str,
    manifest: list[dict[str, str]],
    title: str,
    paper_position: str,
    sources: Iterable[Path],
    notes: str,
) -> None:
    svg = OUT / f"{stem}.svg"
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"
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
        ax.grid(True, axis=grid_axis, color="#ECECEC", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7, width=0.7, length=3, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["dark"],
    )


def add_panel_labels(axes: Iterable[plt.Axes], labels: str = "abcdefghijklmnopqrstuvwxyz") -> None:
    for ax, label in zip(axes, labels):
        add_panel_label(ax, label)


def ci95(values: pd.Series) -> tuple[float, float, float]:
    vals = numeric(values).dropna()
    if vals.empty:
        return math.nan, math.nan, math.nan
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, mean, mean
    half = 1.96 * float(vals.std(ddof=1)) / math.sqrt(len(vals))
    return mean, mean - half, mean + half


def cond_to_grid(condition: str) -> tuple[int, int]:
    idx = int(str(condition).replace("C", "")) - 1
    return idx % 3, idx // 3


def grid_from_condition_summary(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((3, 3), np.nan)
    counts = np.zeros((3, 3), dtype=int)
    labels = np.empty((3, 3), dtype=object)
    labels[:, :] = ""
    for cond in COND_ORDER:
        part = df.loc[df["condition"].eq(cond)]
        if part.empty:
            continue
        x = int(numeric(part["frequency_index_cond"]).dropna().iloc[0])
        y = int(numeric(part["intensity_index_cond"]).dropna().iloc[0])
        vals = numeric(part[value_col]).dropna()
        if vals.empty:
            continue
        values[y, x] = float(vals.mean())
        counts[y, x] = int(vals.shape[0])
        labels[y, x] = cond
    return values, counts, labels


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    im = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_yticklabels(INTENSITY_LABELS)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in range(3):
        for x in range(3):
            val = values[y, x]
            if np.isnan(val):
                continue
            ax.text(
                x,
                y,
                f"{labels[y, x]}\n{val:.2f}\nn={counts[y, x]}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=PALETTE["dark"],
            )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)


def participant_status(values: pd.Series) -> float:
    vals = numeric(values).dropna()
    if vals.empty:
        return 0.0
    mean = float(vals.mean())
    if mean >= 0.8:
        return 1.0
    if mean > 0:
        return 0.5
    return 0.0


def add_unavailable_panel(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        Rectangle(
            (0.05, 0.15),
            0.90,
            0.70,
            facecolor=PALETTE["very_light_gray"],
            edgecolor=PALETTE["light_gray"],
            linewidth=0.9,
        )
    )
    ax.text(
        0.5,
        0.55,
        "Artifact gap",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=6.5, color=PALETTE["gray"])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_01_study_procedure(manifest: list[dict[str, str]]) -> None:
    cond_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    win_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "windows_with_precondition_baseline.csv"
    cond_df = pd.read_csv(cond_path, usecols=["participant_id", "condition", "window_count_cond"])
    n_labels = len(cond_df)
    n_participants = cond_df["participant_id"].nunique()
    n_windows = int(numeric(cond_df["window_count_cond"]).sum())

    fig = plt.figure(figsize=(9.6, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.05, 1.0])
    ax_session = fig.add_subplot(gs[0, 0])
    ax_unit = fig.add_subplot(gs[1, 0])
    ax_flow = fig.add_subplot(gs[2, 0])

    for ax in [ax_session, ax_unit, ax_flow]:
        ax.set_axis_off()

    # Session-level schematic, compressed after adaptation.
    ax_session.set_xlim(0, 100)
    ax_session.set_ylim(0, 1)
    ax_session.text(0, 0.92, "Session-level procedure", fontsize=8, fontweight="bold")
    blocks = [
        (2, 30, "45 s neutral baseline", PALETTE["light_gray"], None),
        (34, 10, "10 s adaptation", PALETTE["orange_soft"], None),
        (48, 45, "9 randomized conditions", PALETTE["blue_soft"], None),
    ]
    for x, width, label, color, hatch in blocks:
        ax_session.add_patch(Rectangle((x, 0.36), width, 0.28, facecolor=color, edgecolor=PALETTE["dark"], linewidth=0.7, hatch=hatch))
        if "randomized" not in label:
            ax_session.text(x + width / 2, 0.50, label, ha="center", va="center", fontsize=7)
        ax_session.plot([x, x], [0.30, 0.70], color=PALETTE["dark"], linewidth=0.9)
    ax_session.text(48 + 45 / 2, 0.78, "9 randomized conditions", ha="center", va="bottom", fontsize=7)
    ax_session.plot([93, 93], [0.30, 0.70], color=PALETTE["dark"], linewidth=0.9)
    for i in range(9):
        x = 50 + i * 4.5
        ax_session.add_patch(Rectangle((x, 0.30), 3.2, 0.40, facecolor="white", edgecolor=PALETTE["blue"], linewidth=0.6))
        ax_session.text(x + 1.6, 0.50, f"C{i + 1}", ha="center", va="center", fontsize=6)
    ax_session.text(2, 0.18, "black solid ticks = LSL event markers", fontsize=6.5, color=PALETTE["dark"])

    # One-condition data unit.
    ax_unit.set_xlim(0, 115)
    ax_unit.set_ylim(0, 1)
    ax_unit.text(0, 0.94, "Participant-condition data unit", fontsize=8, fontweight="bold")
    unit_blocks = [
        (2, 6, "1 s\nrecenter", "#FFFFFF", None),
        (9, 20, "20 s pre-condition baseline", PALETTE["light_gray"], None),
        (33, 70, "70 s formal viewing", PALETTE["blue_soft"], None),
        (104, 10, "questionnaire\nexcluded", PALETTE["orange_soft"], "///"),
    ]
    for x, width, label, color, hatch in unit_blocks:
        ax_unit.add_patch(Rectangle((x, 0.34), width, 0.30, facecolor=color, edgecolor=PALETTE["dark"], linewidth=0.7, hatch=hatch))
        ax_unit.text(x + width / 2, 0.49, label, ha="center", va="center", fontsize=6.1)
        ax_unit.plot([x, x], [0.28, 0.71], color=PALETTE["dark"], linewidth=0.9)
    ax_unit.plot([114, 114], [0.28, 0.71], color=PALETTE["dark"], linewidth=0.9)
    for x in list(range(13, 30, 10)) + list(range(43, 104, 10)):
        ax_unit.plot([x, x], [0.23, 0.76], color=PALETTE["blue"], linestyle="--", linewidth=0.8)
    ax_unit.text(2, 0.15, "blue dashed ticks = 10 s feature windows", fontsize=6.5, color=PALETTE["blue"])

    # Analysis-unit flow.
    ax_flow.set_xlim(0, 100)
    ax_flow.set_ylim(0, 1)
    flow = [
        (2, "10 s feature windows\nn=%d" % n_windows, PALETTE["blue_soft"]),
        (30, "aggregate within\nparticipant-condition", PALETTE["very_light_gray"]),
        (58, "supervised labels\nn=%d (%d x 9)" % (n_labels, n_participants), PALETTE["green_soft"]),
        (84, "LOPO split\nhold out one participant", PALETTE["purple_soft"]),
    ]
    for x, label, color in flow:
        ax_flow.add_patch(FancyBboxPatch((x, 0.30), 18, 0.42, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=color, edgecolor=PALETTE["dark"], linewidth=0.75))
        ax_flow.text(x + 9, 0.51, label, ha="center", va="center", fontsize=7)
    for x0, x1 in [(20, 30), (48, 58), (76, 84)]:
        ax_flow.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51), arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color=PALETTE["dark"]))
    for i in range(15):
        x = 86 + (i % 5) * 2.4
        y = 0.16 - (i // 5) * 0.055
        face = "white" if i == 4 else PALETTE["purple_soft"]
        ax_flow.add_patch(Rectangle((x, y), 1.6, 0.035, facecolor=face, edgecolor=PALETTE["dark"], linewidth=0.35))
    ax_flow.text(84, 0.02, "white tile = held-out participant", fontsize=6.2, color=PALETTE["dark"])

    source = save_source(
        "fig_01_study_procedure_counts",
        pd.DataFrame(
            [
                {
                    "complete_10s_windows": n_windows,
                    "participant_condition_labels": n_labels,
                    "participants": n_participants,
                    "conditions_per_participant": 9,
                }
            ]
        ),
    )
    fig.suptitle("Study procedure and analysis unit", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_01_study_procedure_data_unit",
        manifest,
        "Study procedure and analysis unit",
        "Methods, data processing",
        [cond_path, win_path, source],
        "Schematic emphasizes that 10 s windows are feature inputs and participant-condition rows are the supervised labels.",
    )


def figure_02_measurement_validation_qc(manifest: list[dict[str, str]]) -> None:
    cond_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    legacy_path = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"
    neurokit_path = ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv"

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    axes = axes.ravel()
    add_unavailable_panel(
        axes[0],
        "EEG alpha reactivity",
        "Need eye-open/closed alpha artifact\nbefore paired validation can be drawn.",
    )
    add_unavailable_panel(
        axes[1],
        "ECG detector trace audit",
        "Need raw 10 s ECG trace with saved\nlegacy and NeuroKit R-peak indices.",
    )

    # HR correction paired plot, participant-level means.
    usecols = ["participant_id", "condition", "ecg_hr_bpm__mean"]
    legacy = pd.read_csv(legacy_path, usecols=lambda c: c in usecols).rename(columns={"ecg_hr_bpm__mean": "legacy_hr"})
    neuro = pd.read_csv(neurokit_path, usecols=lambda c: c in usecols).rename(columns={"ecg_hr_bpm__mean": "neurokit_hr"})
    paired = legacy.merge(neuro, on=["participant_id", "condition"], how="inner")
    paired["legacy_hr"] = numeric(paired["legacy_hr"])
    paired["neurokit_hr"] = numeric(paired["neurokit_hr"])
    participant_hr = paired.groupby("participant_id", as_index=False)[["legacy_hr", "neurokit_hr"]].mean()
    source_hr = save_source("fig_02_hr_correction_participant_means", participant_hr)

    ax = axes[2]
    ax.axhspan(60, 100, color=PALETTE["green_soft"], alpha=0.55, zorder=0)
    for _, row in participant_hr.iterrows():
        ax.plot([0, 1], [row["legacy_hr"], row["neurokit_hr"]], color=PALETTE["light_gray"], linewidth=0.8, zorder=1)
    ax.scatter(np.zeros(len(participant_hr)), participant_hr["legacy_hr"], color=PALETTE["red"], s=18, zorder=2, label="Legacy")
    ax.scatter(np.ones(len(participant_hr)), participant_hr["neurokit_hr"], color=PALETTE["blue"], s=18, zorder=2, label="NeuroKit")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legacy", "NeuroKit"])
    ax.set_ylabel("Participant mean HR (BPM)")
    ax.set_title("Heart-rate correction")
    ratio = (participant_hr["legacy_hr"] / participant_hr["neurokit_hr"]).replace([np.inf, -np.inf], np.nan)
    ax.text(0.02, 0.96, f"median ratio={ratio.median():.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    style_axis(ax)

    # Modality coverage heatmap.
    qc_cols = [
        "participant_id",
        "qc_eeg_usable__mean_cond",
        "qc_ecg_usable__mean_cond",
        "qc_eye_usable__mean_cond",
        "qc_head_usable__mean_cond",
        "qc_video_usable__mean_cond",
        "qc_eeg_disabled_by_participant_qc__mean_cond",
    ]
    cond = pd.read_csv(cond_path, usecols=lambda c: c in qc_cols)
    participants = sorted(cond["participant_id"].dropna().unique())
    modalities = [
        ("EEG", "qc_eeg_usable__mean_cond"),
        ("ECG", "qc_ecg_usable__mean_cond"),
        ("Eye", "qc_eye_usable__mean_cond"),
        ("Head", "qc_head_usable__mean_cond"),
        ("Video", "qc_video_usable__mean_cond"),
    ]
    coverage = np.zeros((len(participants), len(modalities)))
    rows = []
    for i, participant in enumerate(participants):
        part = cond.loc[cond["participant_id"].eq(participant)]
        eeg_disabled = False
        if "qc_eeg_disabled_by_participant_qc__mean_cond" in part.columns:
            eeg_disabled = numeric(part["qc_eeg_disabled_by_participant_qc__mean_cond"]).mean() > 0.5
        for j, (modality, col) in enumerate(modalities):
            status = participant_status(part[col]) if col in part.columns else 0.0
            if modality == "EEG" and eeg_disabled:
                status = 0.0
            coverage[i, j] = status
            rows.append({"participant_id": participant, "modality": modality, "status": status})
    source_cov = save_source("fig_02_modality_coverage_status", pd.DataFrame(rows))

    ax = axes[3]
    cmap = ListedColormap([PALETTE["light_gray"], "#F0C85A", PALETTE["green"]])
    ax.imshow(coverage, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(modalities)))
    ax.set_xticklabels([m[0] for m in modalities])
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants, fontsize=6)
    ax.set_title("Modality coverage")
    ax.set_xticks(np.arange(-0.5, len(modalities), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(participants), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(1.02, 0.90, "valid", transform=ax.transAxes, color=PALETTE["green"], fontsize=6.5)
    ax.text(1.02, 0.82, "partial", transform=ax.transAxes, color="#9C7615", fontsize=6.5)
    ax.text(1.02, 0.74, "missing", transform=ax.transAxes, color=PALETTE["gray"], fontsize=6.5)

    add_panel_labels(axes)
    fig.suptitle("Measurement validation and QC status", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_02_measurement_validation_qc",
        manifest,
        "Measurement validation and QC status",
        "Experiment validation, sec4.1",
        [cond_path, legacy_path, neurokit_path, source_hr, source_cov],
        "Alpha reactivity and raw ECG trace panels are marked as artifact gaps; HR correction and coverage use current local artifacts.",
    )


def figure_03_study1_dose_response(manifest: list[dict[str, str]]) -> None:
    cond_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    phase_path = ROOT / "artifacts" / "phase_baseline_v1" / "condition_phase_delta_features.csv"
    cond = pd.read_csv(cond_path)
    phase = pd.read_csv(phase_path)

    fig = plt.figure(figsize=(10.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # A. 3 x 3 condition map.
    ax = axes[0]
    ax.set_xlim(-0.5, 2.55)
    ax.set_ylim(-0.5, 2.55)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_yticklabels(INTENSITY_LABELS)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Intensity")
    ax.set_title("Condition map")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color=PALETTE["light_gray"], linewidth=0.8)
    for cond_id in COND_ORDER:
        x, y = cond_to_grid(cond_id)
        part = cond.loc[cond["condition"].eq(cond_id)].iloc[0]
        ax.add_patch(Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, facecolor="white", edgecolor=PALETTE["dark"], linewidth=0.7))
        ax.text(
            x,
            y,
            f"{cond_id}\nI={part['intensity_cond']:.2f}\nF={part['frequency_cond']:.2f}",
            ha="center",
            va="center",
            fontsize=6.2,
        )
    intensity_levels = sorted(cond["intensity_cond"].dropna().unique())
    frequency_levels = sorted(cond["frequency_cond"].dropna().unique())
    b_i = float(numeric(cond["baseline_intensity_applied_start"]).median())
    b_f = float(numeric(cond["baseline_frequency_applied_start"]).median())
    y_star = np.interp(b_i, intensity_levels, [0, 1, 2])
    x_star = np.interp(b_f, frequency_levels, [0, 1, 2], left=-0.3, right=2.35)
    ax.scatter([x_star], [y_star], marker="*", s=90, color=PALETTE["gray"], edgecolor=PALETTE["dark"], linewidth=0.4, zorder=4)
    ax.text(
        min(x_star, 2.40),
        y_star - 0.33,
        "neutral baseline",
        ha="right",
        va="top",
        fontsize=5.8,
        color=PALETTE["gray"],
    )
    style_axis(ax, "none")

    # B-D. Label heatmaps.
    heatmap_sources = []
    for ax, target, title, cmap, vmin, vmax in [
        (axes[1], "relaxation_cond", "Relaxation", "Greens", 0, 1),
        (axes[2], "calm_cond", "Calm", "Greens", 0, 1),
        (axes[3], "discomfort_cond", "Discomfort", "OrRd", 0, 1),
    ]:
        values, counts, labels = grid_from_condition_summary(cond, target)
        draw_heatmap(ax, values, counts, labels, title, cmap, vmin, vmax)
        source = save_source(
            f"fig_03_{target}_condition_means",
            pd.DataFrame(
                [
                    {
                        "condition": labels[y, x],
                        "intensity_index": y,
                        "frequency_index": x,
                        "mean": values[y, x],
                        "n": counts[y, x],
                    }
                    for y in range(3)
                    for x in range(3)
                    if labels[y, x]
                ]
            ),
        )
        heatmap_sources.append(source)

    # E. Calm marginal response by intensity and frequency.
    ax = axes[4]
    tmp = cond[["participant_id", "intensity_index_cond", "frequency_index_cond", "calm_cond"]].copy()
    tmp["calm_cond"] = numeric(tmp["calm_cond"])
    tmp["intensity_index_cond"] = numeric(tmp["intensity_index_cond"])
    tmp["frequency_index_cond"] = numeric(tmp["frequency_index_cond"])
    source_calm = save_source("fig_03_calm_marginal_source", tmp.dropna())
    line_styles = {0: "-", 1: "--", 2: ":"}
    for f_idx, style in line_styles.items():
        f_part = tmp.loc[tmp["frequency_index_cond"].eq(f_idx)]
        for _, p_part in f_part.groupby("participant_id"):
            ordered = p_part.groupby("intensity_index_cond")["calm_cond"].mean().reindex([0, 1, 2])
            if ordered.notna().sum() >= 2:
                ax.plot([0, 1, 2], ordered, color=PALETTE["light_gray"], linewidth=0.5, alpha=0.35)
        means = []
        lows = []
        highs = []
        for i_idx in [0, 1, 2]:
            vals = f_part.loc[f_part["intensity_index_cond"].eq(i_idx), "calm_cond"]
            mean, lo, hi = ci95(vals)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        ax.fill_between([0, 1, 2], lows, highs, color=PALETTE["teal"], alpha=0.08)
        ax.plot([0, 1, 2], means, linestyle=style, color=PALETTE["teal"], linewidth=1.5, label=f"F {FREQ_LABELS[f_idx]}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(INTENSITY_LABELS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Calm rating")
    ax.set_title("Calm marginal response")
    ax.legend(fontsize=6, loc="best")
    style_axis(ax)

    # F. HR onset paired slope.
    ax = axes[5]
    phase["ecg_hr_bpm__pre"] = numeric(phase["ecg_hr_bpm__pre"])
    phase["ecg_hr_bpm__viewing"] = numeric(phase["ecg_hr_bpm__viewing"])
    participant_phase = phase.groupby("participant_id", as_index=False)[["ecg_hr_bpm__pre", "ecg_hr_bpm__viewing"]].mean()
    participant_phase["delta"] = participant_phase["ecg_hr_bpm__viewing"] - participant_phase["ecg_hr_bpm__pre"]
    source_hr = save_source("fig_03_hr_onset_participant_means", participant_phase)
    for _, row in participant_phase.iterrows():
        ax.plot([0, 1], [row["ecg_hr_bpm__pre"], row["ecg_hr_bpm__viewing"]], color=PALETTE["light_gray"], linewidth=0.8)
    med_pre = float(participant_phase["ecg_hr_bpm__pre"].median())
    med_view = float(participant_phase["ecg_hr_bpm__viewing"].median())
    ax.plot([0, 1], [med_pre, med_view], color=PALETTE["dark"], linewidth=2.0, label="Median")
    ax.scatter(np.zeros(len(participant_phase)), participant_phase["ecg_hr_bpm__pre"], color="white", edgecolor=PALETTE["dark"], s=18, zorder=3)
    ax.scatter(np.ones(len(participant_phase)), participant_phase["ecg_hr_bpm__viewing"], color=PALETTE["dark"], s=18, zorder=3)
    participant_delta = float(participant_phase["delta"].median())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-condition\nbaseline", "Formal\nviewing"])
    ax.set_ylabel("HR (BPM)")
    ax.set_title("HR onset change")
    ax.text(
        0.03,
        0.08,
        f"participant-delta median={participant_delta:.2f} BPM\nreported phase test: -1.9 BPM, p=0.001",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    style_axis(ax)

    add_panel_labels(axes)
    fig.suptitle("Study 1 dose-response evidence", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_03_study1_dose_response_main",
        manifest,
        "Study 1 dose-response evidence",
        "Results, RQ1",
        [cond_path, phase_path, source_calm, source_hr, *heatmap_sources],
        "Shows condition grid, participant-condition label heatmaps, calm intensity pattern, and HR onset drop without treating windows as independent labels.",
    )


def box_strip_by_condition(ax: plt.Axes, df: pd.DataFrame, col: str, ylabel: str, title: str, color: str) -> None:
    data = []
    positions = np.arange(1, 10)
    for cond in COND_ORDER:
        values = numeric(df.loc[df["condition"].eq(cond), col]).dropna().to_numpy()
        data.append(values)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 1.0},
        boxprops={"edgecolor": PALETTE["dark"], "linewidth": 0.7},
        whiskerprops={"color": PALETTE["dark"], "linewidth": 0.7},
        capprops={"color": PALETTE["dark"], "linewidth": 0.7},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    for pos, vals in zip(positions, data):
        if len(vals):
            jitter = RNG.normal(0, 0.045, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, color=PALETTE["gray"], s=8, alpha=0.55, linewidths=0)
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(COND_ORDER)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def figure_04_physiological_null_results(manifest: list[dict[str, str]]) -> None:
    cond_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    cols = [
        "condition",
        "delta_median_alpha_power_channel_mean",
        "delta_median_beta_power_channel_mean",
        "delta_ecg_hr_bpm__mean",
        "delta_ecg_hrv_60s_rmssd_ms__mean",
    ]
    df = pd.read_csv(cond_path, usecols=lambda c: c in cols)
    source = save_source("fig_04_physiological_null_distributions", df)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8), constrained_layout=True)
    panels = [
        ("delta_median_alpha_power_channel_mean", "Delta alpha power", "Alpha delta", PALETTE["teal"]),
        ("delta_median_beta_power_channel_mean", "Delta beta power", "Beta delta", PALETTE["purple"]),
        ("delta_ecg_hr_bpm__mean", "Delta HR (BPM)", "Heart-rate delta", PALETTE["blue"]),
        ("delta_ecg_hrv_60s_rmssd_ms__mean", "Delta RMSSD (ms)", "RMSSD delta", PALETTE["orange"]),
    ]
    for ax, (col, ylabel, title, color) in zip(axes.ravel(), panels):
        box_strip_by_condition(ax, df, col, ylabel, title, color)
        if col == "delta_ecg_hrv_60s_rmssd_ms__mean":
            ax.set_yscale("symlog", linthresh=100)
            ax.set_title("RMSSD delta (symlog)")
    add_panel_labels(axes.ravel())
    fig.suptitle("Physiological dose-effect distributions", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_04_physiological_null_results",
        manifest,
        "Physiological dose-effect distributions",
        "Results or Appendix, RQ1 physiology",
        [cond_path, source],
        "Baseline-corrected participant-condition distributions by C1-C9; horizontal zero line supports visible null-result reporting.",
    )


def figure_05_model_comparison_gate(manifest: list[dict[str, str]]) -> None:
    fusion_path = ROOT / "artifacts" / "fusion_minimal" / "metrics.csv"
    df = pd.read_csv(fusion_path)
    combos = df.loc[df["record_type"].eq("combination")].copy()
    baselines = df.loc[df["record_type"].eq("baseline")].copy()
    for col in ["relaxation_mae", "discomfort_mae", "discomfort_high_recall", "discomfort_high_false_negatives"]:
        combos[col] = numeric(combos[col])
        baselines[col] = numeric(baselines[col])
    combos["sort_key"] = combos[["relaxation_mae", "discomfort_mae"]].rank(pct=True).mean(axis=1)
    combos = combos.sort_values("sort_key", ascending=True).reset_index(drop=True)
    source = save_source("fig_05_fusion_minimal_metrics", pd.concat([baselines, combos], ignore_index=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 6.0), sharey=True, constrained_layout=True)
    y = np.arange(len(combos))
    labels = combos["combination"].tolist()

    def baseline_value(name: str, col: str) -> float:
        part = baselines.loc[baselines["baseline"].eq(name), col]
        return float(part.iloc[0]) if len(part) else math.nan

    for ax, col, title, best_color in [
        (axes[0], "relaxation_mae", "Relaxation MAE", PALETTE["teal"]),
        (axes[1], "discomfort_mae", "Discomfort MAE", PALETTE["orange"]),
    ]:
        values = combos[col].to_numpy(dtype=float)
        best_idx = int(np.nanargmin(values))
        ax.hlines(y, xmin=np.nanmin(values) * 0.92, xmax=values, color=PALETTE["light_gray"], linewidth=0.8)
        colors = [best_color if i == best_idx else PALETTE["gray"] for i in range(len(values))]
        ax.scatter(values, y, color=colors, s=24, zorder=3)
        ax.axvline(baseline_value("condition_only", col), color=PALETTE["dark"], linewidth=1.0, label="Condition-only")
        ax.axvline(baseline_value("history", col), color=PALETTE["dark"], linestyle="--", linewidth=1.0, label="History")
        ax.axvline(baseline_value("random_uniform", col), color=PALETTE["gray"], linestyle=":", linewidth=1.2, label="Random")
        ax.set_xlabel("MAE (lower is better)")
        ax.set_title(title)
        style_axis(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6, loc="lower right")

    ax = axes[2]
    recall = combos["discomfort_high_recall"].to_numpy(dtype=float)
    fn = combos["discomfort_high_false_negatives"].fillna(-1).astype(int).to_numpy()
    best_idx = int(np.nanargmax(recall))
    ax.hlines(y, xmin=0, xmax=recall, color=PALETTE["light_gray"], linewidth=0.8)
    ax.scatter(recall, y, color=[PALETTE["orange"] if i == best_idx else PALETTE["gray"] for i in range(len(recall))], s=24, zorder=3)
    for yi, x, f in zip(y, recall, fn):
        if np.isfinite(x):
            ax.text(min(x + 0.025, 1.02), yi, f"FN={f}", va="center", fontsize=5.8, color=PALETTE["gray"])
    for name, style, color in [
        ("condition_only", "-", PALETTE["dark"]),
        ("history", "--", PALETTE["dark"]),
        ("random_uniform", ":", PALETTE["gray"]),
    ]:
        ax.axvline(baseline_value(name, "discomfort_high_recall"), color=color, linestyle=style, linewidth=1.0)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("High-discomfort recall")
    ax.set_title("Safety recall")
    style_axis(ax)

    add_panel_labels(axes)
    fig.suptitle("Model comparison and deployment gate evidence", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_05_model_comparison_deployment_gate",
        manifest,
        "Model comparison and deployment gate evidence",
        "Results, RQ2",
        [fusion_path, source],
        "Horizontal lollipop panels show that fusion combinations remain research-only and are compared against condition-only, history, and random baselines.",
    )


def within_participant_spearman(labels: pd.DataFrame, prediction: np.ndarray, target: str) -> float:
    tmp = labels[["participant_id", target]].copy()
    tmp["prediction"] = prediction
    vals = []
    for _, part in tmp.groupby("participant_id"):
        y = numeric(part[target])
        p = numeric(part["prediction"])
        if y.nunique(dropna=True) < 2 or p.nunique(dropna=True) < 2:
            continue
        vals.append(float(y.rank().corr(p.rank())))
    return float(np.nanmean(vals)) if vals else math.nan


def simulate_random_floor(labels: pd.DataFrame, target: str, repeats: int = 1000) -> pd.DataFrame:
    y = numeric(labels[target]).to_numpy(dtype=float)
    draws = []
    for i in range(repeats):
        pred = RNG.uniform(0, 1, size=len(y))
        mae = float(np.nanmean(np.abs(y - pred)))
        rho = within_participant_spearman(labels, pred, target)
        draws.append({"simulation": i, "target": target, "mae": mae, "within_participant_spearman": rho})
    return pd.DataFrame(draws)


def figure_06_random_floor_generalization_gap(manifest: list[dict[str, str]]) -> None:
    cond_path = ROOT / "artifacts" / "reports" / "supplementary_round2" / "condition_baseline_merged_for_delta.csv"
    ranking_path = ROOT / "artifacts" / "fusion_four_channel_ranking" / "continuous_ranking_metrics.csv"
    fusion_path = ROOT / "artifacts" / "fusion_minimal" / "metrics.csv"
    lopo_path = ROOT / "artifacts" / "reports" / "condition_level_lopo_metrics.json"

    labels = pd.read_csv(cond_path, usecols=["participant_id", "relaxation_cond"]).rename(columns={"relaxation_cond": "relaxation"})
    random_draws = simulate_random_floor(labels, "relaxation", repeats=1000)
    source_random = save_source("fig_06_random_floor_simulations", random_draws)

    ranking = pd.read_csv(ranking_path)
    fusion = pd.read_csv(fusion_path)
    with lopo_path.open("r", encoding="utf-8") as f:
        lopo = json.load(f)["metrics"]

    cond_rho = float(
        ranking.loc[
            ranking["record_type"].eq("baseline")
            & ranking["baseline"].eq("condition_only")
            & ranking["target"].eq("relaxation"),
            "within_participant_spearman_mean",
        ].iloc[0]
    )
    best_rho_row = ranking.loc[ranking["record_type"].eq("combination") & ranking["target"].eq("relaxation")].copy()
    best_rho_row["within_participant_spearman_mean"] = numeric(best_rho_row["within_participant_spearman_mean"])
    best_rho = float(best_rho_row["within_participant_spearman_mean"].max())
    best_rho_label = str(best_rho_row.loc[best_rho_row["within_participant_spearman_mean"].idxmax(), "combination"])

    cond_mae = float(fusion.loc[fusion["record_type"].eq("baseline") & fusion["baseline"].eq("condition_only"), "relaxation_mae"].iloc[0])
    combo_mae = fusion.loc[fusion["record_type"].eq("combination")].copy()
    combo_mae["relaxation_mae"] = numeric(combo_mae["relaxation_mae"])
    best_mae = float(combo_mae["relaxation_mae"].min())
    best_mae_label = str(combo_mae.loc[combo_mae["relaxation_mae"].idxmin(), "combination"])

    gap_rows = pd.DataFrame(
        [
            {
                "metric": "Spearman rho",
                "LOPO": lopo["targets"]["relaxation"]["spearman"],
                "Subject-dependent": lopo["personalized_calibration"]["3"]["relaxation_spearman"],
                "direction": "higher is better",
            },
            {
                "metric": "MAE",
                "LOPO": lopo["targets"]["relaxation"]["mae"],
                "Subject-dependent": lopo["personalized_calibration"]["3"]["relaxation_mae"],
                "direction": "lower is better",
            },
        ]
    )
    source_gap = save_source("fig_06_generalization_gap_metrics", gap_rows)

    fig = plt.figure(figsize=(10.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_mae = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[0, 2])

    rho_vals = random_draws["within_participant_spearman"].dropna()
    lo, hi = np.percentile(rho_vals, [2.5, 97.5])
    ax_rho.axvspan(lo, hi, color=PALETTE["light_gray"], alpha=0.45, label="Random 95% band")
    ax_rho.hist(rho_vals, bins=32, color=PALETTE["gray"], alpha=0.55, density=True)
    ax_rho.axvline(cond_rho, color=PALETTE["blue"], linewidth=1.5, label="Condition-only")
    ax_rho.axvline(best_rho, color=PALETTE["orange"], linewidth=1.5, label=f"Best {best_rho_label}")
    ax_rho.set_xlabel("Within-participant Spearman")
    ax_rho.set_ylabel("Density")
    ax_rho.set_title("Random floor: ranking")
    ax_rho.legend(fontsize=6, loc="upper left")
    style_axis(ax_rho)

    mae_vals = random_draws["mae"].dropna()
    lo, hi = np.percentile(mae_vals, [2.5, 97.5])
    ax_mae.axvspan(lo, hi, color=PALETTE["light_gray"], alpha=0.45)
    ax_mae.hist(mae_vals, bins=32, color=PALETTE["gray"], alpha=0.55, density=True)
    ax_mae.axvline(cond_mae, color=PALETTE["blue"], linewidth=1.5, label="Condition-only")
    ax_mae.axvline(best_mae, color=PALETTE["orange"], linewidth=1.5, label=f"Best {best_mae_label}")
    ax_mae.set_xlabel("Relaxation MAE")
    ax_mae.set_ylabel("Density")
    ax_mae.set_title("Random floor: error")
    style_axis(ax_mae)

    y = np.arange(len(gap_rows))
    for i, row in gap_rows.iterrows():
        ax_gap.plot([row["LOPO"], row["Subject-dependent"]], [i, i], color=PALETTE["light_gray"], linewidth=1.2)
        ax_gap.scatter(row["LOPO"], i, facecolor="white", edgecolor=PALETTE["dark"], s=42, linewidth=1.0, label="LOPO" if i == 0 else None)
        ax_gap.scatter(row["Subject-dependent"], i, color=PALETTE["teal"], s=42, label="Subject-dependent" if i == 0 else None)
        ax_gap.text(row["LOPO"], i + 0.12, f"{row['LOPO']:.3f}", ha="center", fontsize=6)
        ax_gap.text(row["Subject-dependent"], i - 0.18, f"{row['Subject-dependent']:.3f}", ha="center", fontsize=6)
    ax_gap.set_yticks(y)
    ax_gap.set_yticklabels(gap_rows["metric"])
    ax_gap.set_ylim(-0.45, len(gap_rows) - 0.55 + 0.35)
    ax_gap.set_xlabel("Metric value")
    ax_gap.set_title("Generalization gap")
    ax_gap.text(
        0.03,
        0.50,
        "rho: higher better\nMAE: lower better",
        transform=ax_gap.transAxes,
        fontsize=6.2,
        color=PALETTE["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    ax_gap.legend(fontsize=6, loc="upper right")
    style_axis(ax_gap)

    add_panel_labels([ax_rho, ax_mae, ax_gap])
    fig.suptitle("Random floor and subject-generalization gap", fontsize=10, fontweight="bold")
    save_figure(
        fig,
        "fig_06_random_floor_generalization_gap",
        manifest,
        "Random floor and subject-generalization gap",
        "Results and Discussion, RQ2",
        [cond_path, ranking_path, fusion_path, lopo_path, source_random, source_gap],
        "Random predictor distributions are regenerated deterministically from the 135 labels; vertical lines use existing condition-only and best-combination metrics.",
    )


def read_decisions() -> pd.DataFrame:
    root = ROOT / "artifacts" / "realtime" / "adaptive_control"
    rows = []
    for path in sorted(root.glob("*/decisions.jsonl")):
        session_id = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            decision = record.get("decision", {})
            current = decision.get("current_condition")
            target = decision.get("target_condition") or current
            target_idx = int(str(target).replace("C", "")) if target else np.nan
            current_idx = int(str(current).replace("C", "")) if current else np.nan
            issued = decision.get("issued_unix_ms")
            window_end = record.get("window_end_ms")
            latency = float(issued - window_end) if issued is not None and window_end is not None else np.nan
            utils = record.get("candidate_utilities") or {}
            target_utility = utils.get(str(target), np.nan)
            rows.append(
                {
                    "session_id": session_id,
                    "cycle_index": decision.get("cycle_index"),
                    "window_end_ms": window_end,
                    "issued_unix_ms": issued,
                    "latency_ms": latency,
                    "action": decision.get("action"),
                    "current_condition": current,
                    "target_condition": target,
                    "current_index": current_idx,
                    "target_index": target_idx,
                    "predicted_relaxation": decision.get("predicted_relaxation"),
                    "predicted_discomfort": decision.get("predicted_discomfort"),
                    "utility_delta": decision.get("utility_delta"),
                    "target_utility": target_utility,
                }
            )
    df = pd.DataFrame(rows)
    for col in ["cycle_index", "window_end_ms", "issued_unix_ms", "latency_ms", "current_index", "target_index", "predicted_relaxation", "predicted_discomfort", "utility_delta", "target_utility"]:
        if col in df.columns:
            df[col] = numeric(df[col])
    return df


def figure_07_closed_loop_replay_latency(manifest: list[dict[str, str]]) -> None:
    latency_path = ROOT / "artifacts" / "reports" / "supplementary" / "latency_benchmark.csv"
    decisions = read_decisions()
    source_decisions = save_source("fig_07_adaptive_decisions_parsed", decisions)

    transition_counts = (
        decisions.sort_values(["session_id", "cycle_index"])
        .groupby("session_id")["target_index"]
        .apply(lambda s: int((s.ffill().diff().abs() > 0).sum()))
        .sort_values(ascending=False)
    )
    session_id = str(transition_counts.index[0])
    session = decisions.loc[decisions["session_id"].eq(session_id)].sort_values("cycle_index").copy()
    session["time_min"] = (session["window_end_ms"] - session["window_end_ms"].min()) / 60000.0

    fig = plt.figure(figsize=(10.4, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.35])
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_series = fig.add_subplot(gs[0, 1])
    ax_latency = fig.add_subplot(gs[1, :])

    # A. Adaptive grid trajectory.
    load_colors = plt.cm.RdYlGn_r(np.linspace(0.18, 0.82, 9))
    ax_grid.set_xlim(-0.6, 2.6)
    ax_grid.set_ylim(-0.6, 2.6)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks(range(3))
    ax_grid.set_yticks(range(3))
    ax_grid.set_xticklabels(FREQ_LABELS)
    ax_grid.set_yticklabels(INTENSITY_LABELS)
    ax_grid.set_xlabel("Frequency")
    ax_grid.set_ylabel("Intensity")
    ax_grid.set_title("Replay condition transitions")
    for cond in COND_ORDER:
        idx = int(cond[1:])
        x, y = cond_to_grid(cond)
        ax_grid.add_patch(Rectangle((x - 0.45, y - 0.45), 0.90, 0.90, facecolor=load_colors[idx - 1], edgecolor="white", linewidth=1.0))
        ax_grid.text(x, y, cond, ha="center", va="center", fontsize=8, fontweight="bold", color=PALETTE["dark"])
    changes = session.loc[session["target_index"].notna(), ["target_index", "cycle_index"]].copy()
    changes = changes.loc[changes["target_index"].ne(changes["target_index"].shift())]
    transitions = list(zip(changes["target_index"].astype(int).tolist()[:-1], changes["target_index"].astype(int).tolist()[1:]))
    transitions = transitions[:24]
    for k, (src, dst) in enumerate(transitions):
        sx, sy = cond_to_grid(f"C{src}")
        dx, dy = cond_to_grid(f"C{dst}")
        color = plt.cm.Purples(0.35 + 0.55 * (k + 1) / max(1, len(transitions)))
        ax_grid.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (dx, dy),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.1,
                color=color,
                alpha=0.85,
                shrinkA=16,
                shrinkB=16,
                connectionstyle="arc3,rad=0.12",
            )
        )
    style_axis(ax_grid, "none")

    # B. Replay time series.
    ax_series.plot(session["time_min"], session["predicted_relaxation"], color=PALETTE["teal"], linewidth=1.4, label="Predicted relaxation")
    ax_series.plot(session["time_min"], session["predicted_discomfort"], color=PALETTE["orange"], linewidth=1.4, label="Predicted discomfort")
    ax_series.plot(session["time_min"], session["utility_delta"], color=PALETTE["dark"], linewidth=1.1, label="Utility delta")
    ax_series.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=0.8)
    ax_series.set_xlabel("Replay time (min)")
    ax_series.set_ylabel("Prediction / utility")
    ax_series.set_title(f"Controller time series ({session_id[:8]}...)")
    ax_series.legend(fontsize=6, ncol=2, loc="upper right")
    ymin, ymax = ax_series.get_ylim()
    band_y = ymin + 0.03 * (ymax - ymin)
    band_h = 0.04 * (ymax - ymin)
    for _, row in session.iterrows():
        if pd.isna(row["target_index"]):
            continue
        color = load_colors[int(row["target_index"]) - 1]
        ax_series.add_patch(Rectangle((row["time_min"], band_y), 0.11, band_h, facecolor=color, edgecolor="none", alpha=0.85))
    ax_series.text(
        0.02,
        0.86,
        "bottom band = target condition",
        transform=ax_series.transAxes,
        fontsize=6.2,
        color=PALETTE["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    style_axis(ax_series)

    # C. Latency strip plot.
    lat = decisions.dropna(subset=["latency_ms"]).copy()
    session_order = lat.groupby("session_id")["latency_ms"].median().sort_values().index.tolist()
    source_latency = save_source("fig_07_latency_decisions", lat[["session_id", "cycle_index", "latency_ms"]])
    for i, sid in enumerate(session_order):
        vals = lat.loc[lat["session_id"].eq(sid), "latency_ms"].clip(lower=1)
        x = np.full(len(vals), i) + RNG.normal(0, 0.055, len(vals))
        ax_latency.scatter(x, vals, color=PALETTE["gray"], alpha=0.45, s=10, linewidths=0)
        med = float(vals.median())
        ax_latency.plot([i - 0.20, i + 0.20], [med, med], color=PALETTE["dark"], linewidth=1.2)
    ax_latency.axhline(10000, color=PALETTE["red"], linestyle="--", linewidth=1.0, label="10,000 ms cadence")
    ax_latency.set_yscale("log")
    ax_latency.set_xticks(range(len(session_order)))
    ax_latency.set_xticklabels([sid[:8] for sid in session_order], rotation=25, ha="right")
    ax_latency.set_ylabel("Decision latency (ms, log scale)")
    ax_latency.set_xlabel("Replay session")
    ax_latency.set_title("Latency by adaptive replay session")
    ax_latency.legend(fontsize=6, loc="upper right")
    style_axis(ax_latency)

    add_panel_labels([ax_grid, ax_series, ax_latency])
    fig.suptitle("Closed-loop replay behavior and latency", fontsize=10, fontweight="bold")
    source_paths = [latency_path, source_decisions, source_latency]
    source_paths.extend(sorted((ROOT / "artifacts" / "realtime" / "adaptive_control").glob("*/decisions.jsonl")))
    save_figure(
        fig,
        "fig_07_closed_loop_replay_latency",
        manifest,
        "Closed-loop replay behavior and latency",
        "Results, RQ3",
        source_paths,
        "Representative replay is selected by maximum target-condition transitions; latency panel uses all adaptive_control decisions.",
    )


def write_manifest(manifest: list[dict[str, str]]) -> None:
    path = OUT / "figure_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure", "title", "paper_position", "svg", "png", "pdf", "tiff", "sources", "notes"],
        )
        writer.writeheader()
        writer.writerows(manifest)


def write_contract(manifest: list[dict[str, str]]) -> None:
    lines = [
        "# Thesis Priority Figure Contract",
        "",
        "Generated from local artifacts only. SVG is the primary editable vector format; PDF/PNG/TIFF are exported siblings.",
        "",
        "## Scope",
        "",
        "- fig_01 documents the data unit: 10 s windows are feature inputs; participant-condition rows are supervised labels.",
        "- fig_02 is a mixed validation/QC figure. Alpha reactivity and raw ECG trace panels remain explicit artifact gaps.",
        "- fig_03 is the RQ1 Results main figure replacing the dose-response placeholder.",
        "- fig_04 makes physiological null-result distributions visible.",
        "- fig_05 summarizes RQ2 fusion/baseline comparisons and deployment-gate evidence.",
        "- fig_06 visualizes random-floor proximity and subject-generalization gap.",
        "- fig_07 visualizes RQ3 replay controller behavior and latency without implying intervention efficacy.",
        "",
        "## Caveats",
        "",
        "- No eye-open/closed EEG alpha artifact was found in the current workspace.",
        "- No raw ECG trace with saved legacy and NeuroKit R-peak indices was found in the current workspace.",
        "- The HR onset panel plots participant-level means from available phase artifacts and annotates the reported phase-level test separately.",
        "- `combine.png` was not regenerated here; its panel text should avoid claiming biosignal-driven transformation for Study 1.",
        "",
        "## Outputs",
        "",
    ]
    for row in manifest:
        lines.append(f"- {row['figure']}: {row['svg']}")
    (OUT / "figure_contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_out()
    manifest: list[dict[str, str]] = []
    figure_01_study_procedure(manifest)
    figure_02_measurement_validation_qc(manifest)
    figure_03_study1_dose_response(manifest)
    figure_04_physiological_null_results(manifest)
    figure_05_model_comparison_gate(manifest)
    figure_06_random_floor_generalization_gap(manifest)
    figure_07_closed_loop_replay_latency(manifest)
    write_manifest(manifest)
    write_contract(manifest)
    print(f"Wrote {len(manifest)} thesis-priority figures to {rel(OUT)}")


if __name__ == "__main__":
    main()
