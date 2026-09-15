#!/usr/bin/env python3
"""Create publication-grade figures for the frozen-feature fusion evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "relax" / "foundation_compression_fusion_20260717"
)

METHODS = [
    "joint_block_balanced_pca8",
    "modality_pca2_additive",
    "supervised_pls2",
    "linear_shared_private",
    "modality_expert_simplex",
    "modality_expert_simplex_foundation_only",
]
METHOD_LABELS = {
    "joint_block_balanced_pca8": "Joint balanced PCA8",
    "modality_pca2_additive": "Modality PCA2",
    "supervised_pls2": "Supervised PLS2",
    "linear_shared_private": "Linear shared–private",
    "modality_expert_simplex": "Expert simplex + head",
    "modality_expert_simplex_foundation_only": "Expert simplex, foundation only",
}
SHORT_LABELS = {
    "joint_block_balanced_pca8": "Joint PCA",
    "modality_pca2_additive": "Modality PCA",
    "supervised_pls2": "PLS",
    "linear_shared_private": "Shared–private",
    "modality_expert_simplex": "Expert + head",
    "modality_expert_simplex_foundation_only": "Expert foundation",
}
METHOD_COLORS = {
    "joint_block_balanced_pca8": "#484878",
    "modality_pca2_additive": "#7884B4",
    "supervised_pls2": "#A8A8A8",
    "linear_shared_private": "#42949E",
    "modality_expert_simplex": "#B64342",
    "modality_expert_simplex_foundation_only": "#E9A6A1",
}
MODALITY_COLORS = {
    "eeg": "#A8A8A8",
    "ecg": "#3775BA",
    "eye": "#E2AD45",
    "head": "#42949E",
    "video": "#9A4D8E",
}
TARGET_COLORS = {"relaxation": "#3775BA", "discomfort": "#B64342"}
TARGET_MARKERS = {"relaxation": "o", "discomfort": "s"}
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_better_worse", ["#3775BA", "#F7F7F7", "#B64342"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["xtick.labelsize"] = 6.5
    plt.rcParams["ytick.labelsize"] = 6.5
    plt.rcParams["legend.fontsize"] = 6.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["legend.frameon"] = False


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, required, path)
    return frame


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def symmetric_norm(values: np.ndarray, floor: float = 0.25) -> TwoSlopeNorm:
    limit = max(float(np.nanmax(np.abs(values))), floor)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def save_bundle(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        (".png", {"dpi": 300}),
    ):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def make_outcome_figure(
    evaluation_dir: Path, output_dir: Path
) -> tuple[list[Path], list[Path]]:
    candidate = read_csv(
        evaluation_dir / "candidate_summary.csv",
        [
            "candidate",
            "relaxation_delta_mean",
            "discomfort_delta_mean",
            "macro_delta_mean",
        ],
    )
    paired = read_csv(
        evaluation_dir / "paired_statistics_18_test_holm.csv",
        [
            "candidate",
            "outcome",
            "mean_paired_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "holm_p",
            "holm_significant_0_05",
        ],
    )
    participant = read_csv(
        evaluation_dir / "participant_metrics.csv",
        ["candidate", "seed", "participant_id", "outcome", "delta_vs_condition"],
    )

    formal = candidate[candidate["candidate"].isin(METHODS)].copy()
    if set(formal["candidate"]) != set(METHODS):
        raise ValueError("Candidate summary does not contain the exact formal panel")
    macro = paired[
        paired["candidate"].isin(METHODS) & paired["outcome"].eq("macro")
    ].copy()
    formal = formal.merge(macro, on="candidate", validate="one_to_one")
    formal["display_label"] = formal["candidate"].map(METHOD_LABELS)
    forest_order = (
        formal.sort_values(["mean_paired_delta", "candidate"])["candidate"].tolist()
    )
    formal["forest_order"] = formal["candidate"].map(
        {method: index for index, method in enumerate(forest_order)}
    )
    formal = formal.sort_values("forest_order")

    participant_macro = participant[
        participant["candidate"].isin(METHODS) & participant["outcome"].eq("macro")
    ].copy()
    participant_macro = (
        participant_macro.groupby(["candidate", "participant_id"], as_index=False)
        .agg(
            seed_averaged_macro_delta=("delta_vs_condition", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    if not participant_macro["seed_count"].eq(3).all():
        raise ValueError("Participant heatmap does not contain all three seeds")
    participant_macro["display_label"] = participant_macro["candidate"].map(METHOD_LABELS)
    participants = sorted(participant_macro["participant_id"].unique())
    heatmap = (
        participant_macro.pivot(
            index="candidate",
            columns="participant_id",
            values="seed_averaged_macro_delta",
        )
        .loc[forest_order, participants]
        .to_numpy()
        * 1_000
    )

    source_paths = [
        output_dir / "figure1_formal_macro_effects_source.csv",
        output_dir / "figure1_target_tradeoff_source.csv",
        output_dir / "figure1_participant_macro_deltas_source.csv",
    ]
    formal.to_csv(source_paths[0], index=False)
    formal[
        [
            "candidate",
            "display_label",
            "relaxation_delta_mean",
            "discomfort_delta_mean",
            "macro_delta_mean",
        ]
    ].to_csv(source_paths[1], index=False)
    participant_macro.to_csv(source_paths[2], index=False)

    fig = plt.figure(figsize=(7.20, 6.55))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.2],
        width_ratios=[1.18, 1.0],
        hspace=0.42,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    y = np.arange(len(formal))[::-1]
    for yi, row in zip(y, formal.itertuples(index=False)):
        color = METHOD_COLORS[row.candidate]
        ax_a.plot(
            [row.bootstrap_ci_low * 1_000, row.bootstrap_ci_high * 1_000],
            [yi, yi],
            color=color,
            lw=1.5,
            solid_capstyle="round",
        )
        ax_a.scatter(
            row.mean_paired_delta * 1_000,
            yi,
            s=28 if row.candidate == "modality_expert_simplex" else 20,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    ax_a.axvline(0, color="#767676", ls="--", lw=0.9)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(formal["display_label"])
    ax_a.set_xlabel(r"Macro MAE difference vs Condition-only ($\times 10^{-3}$)")
    ax_a.set_title("Participant-clustered macro effects")
    ax_a.grid(axis="x", color="#E3E3E3", lw=0.55)
    ax_a.text(
        0.98,
        0.97,
        "0/18 Holm-significant",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        color="#606060",
        fontsize=6.3,
    )
    panel_label(ax_a, "a")

    x = formal["relaxation_delta_mean"].to_numpy() * 1_000
    y_delta = formal["discomfort_delta_mean"].to_numpy() * 1_000
    x_margin = max(np.ptp(x) * 0.22, 0.45)
    y_margin = max(np.ptp(y_delta) * 0.22, 0.45)
    xlim = (min(x.min() - x_margin, -0.25), max(x.max() + x_margin, 0.25))
    ylim = (
        min(y_delta.min() - y_margin, -0.25),
        max(y_delta.max() + y_margin, 0.25),
    )
    ax_b.add_patch(
        Rectangle(
            (xlim[0], ylim[0]),
            0 - xlim[0],
            0 - ylim[0],
            facecolor="#DDF3DE",
            edgecolor="none",
            alpha=0.7,
            zorder=0,
        )
    )
    for row in formal.itertuples(index=False):
        xx = row.relaxation_delta_mean * 1_000
        yy = row.discomfort_delta_mean * 1_000
        ax_b.scatter(
            xx,
            yy,
            s=30 if row.candidate == "modality_expert_simplex" else 22,
            color=METHOD_COLORS[row.candidate],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        offset = (3, 3)
        horizontal_alignment = "left"
        if row.candidate == "modality_expert_simplex":
            offset = (-5, -11)
            horizontal_alignment = "right"
        elif row.candidate == "modality_expert_simplex_foundation_only":
            offset = (5, 6)
        elif row.candidate == "modality_pca2_additive":
            offset = (-4, 4)
            horizontal_alignment = "right"
        ax_b.annotate(
            SHORT_LABELS[row.candidate],
            (xx, yy),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            ha=horizontal_alignment,
        )
    ax_b.axvline(0, color="#767676", ls="--", lw=0.8)
    ax_b.axhline(0, color="#767676", ls="--", lw=0.8)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(*ylim)
    ax_b.set_xlabel(r"Relaxation MAE difference ($\times 10^{-3}$)")
    ax_b.set_ylabel(r"Discomfort MAE difference ($\times 10^{-3}$)")
    ax_b.set_title("Target trade-off (lower left is better)")
    ax_b.grid(color="#EAEAEA", lw=0.5)
    panel_label(ax_b, "b")

    norm = symmetric_norm(heatmap, floor=1.0)
    image = ax_c.imshow(heatmap, cmap=DELTA_CMAP, norm=norm, aspect="auto")
    ax_c.set_xticks(range(len(participants)))
    ax_c.set_xticklabels(participants)
    ax_c.set_yticks(range(len(forest_order)))
    ax_c.set_yticklabels([METHOD_LABELS[method] for method in forest_order])
    ax_c.set_title("Participant-level macro heterogeneity after seed averaging")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = heatmap[i, j]
            color = "white" if abs(value) > 0.58 * max(abs(norm.vmin), norm.vmax) else "#272727"
            ax_c.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
            )
    cbar = fig.colorbar(image, ax=ax_c, fraction=0.022, pad=0.018)
    cbar.set_label(r"Macro MAE difference ($\times 10^{-3}$)")
    cbar.ax.tick_params(labelsize=6)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    panel_label(ax_c, "c", x=-0.075, y=1.04)

    fig.suptitle(
        "Constrained frozen-feature fusion does not establish a Condition-only gain",
        x=0.5,
        y=0.995,
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Negative differences indicate lower MAE. Intervals are 10,000 participant-cluster bootstrap resamples; n = 9 participants.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    figure_paths = save_bundle(fig, output_dir / "figure1_formal_outcomes")
    return figure_paths, source_paths


def make_feature_figure(
    artifact_root: Path, output_dir: Path
) -> tuple[list[Path], list[Path]]:
    audit_dir = artifact_root / "audit"
    evaluation_dir = artifact_root / "evaluation"
    modality = read_csv(
        audit_dir / "modality_audit.csv",
        [
            "modality",
            "dimension",
            "entropy_effective_rank",
            "pc_count_95",
            "participant_feature_r2_descriptive",
            "condition_feature_r2_descriptive",
        ],
    )
    joint = read_csv(
        audit_dir / "joint_pca8_fold_audit.csv",
        [
            "joint_input_dimension",
            "ecg_dimension_with_presence",
            "ecg_loading_mass_share",
            "eye_dimension_with_presence",
            "eye_loading_mass_share",
            "head_dimension_with_presence",
            "head_loading_mass_share",
            "video_dimension_with_presence",
            "video_loading_mass_share",
        ],
    )
    target_probe = read_csv(
        audit_dir / "fixed_target_information_probe.csv",
        ["modality", "target", "delta_vs_condition"],
    )
    weights = read_csv(
        evaluation_dir / "expert_weight_summary.csv",
        ["candidate", "target", "modality", "mean_weight", "std_weight"],
    )
    removals = read_csv(
        evaluation_dir / "runner_validation_modality_ablation_summary.csv",
        [
            "candidate",
            "outcome",
            "modality",
            "mean_delta_without_minus_full",
            "std_delta_without_minus_full",
            "fold_seed_evaluations",
        ],
    )

    modality_order = ["eeg", "ecg", "eye", "head", "video"]
    no_eeg_order = ["ecg", "eye", "head", "video"]
    modality = modality.set_index("modality").loc[modality_order].reset_index()
    block_rows = []
    joint_dim = float(joint["joint_input_dimension"].mean())
    for name in no_eeg_order:
        block_rows.append(
            {
                "modality": name,
                "coordinate_share": float(joint[f"{name}_dimension_with_presence"].mean())
                / joint_dim,
                "joint_pca8_loading_mass_share": float(
                    joint[f"{name}_loading_mass_share"].mean()
                ),
            }
        )
    blocks = pd.DataFrame(block_rows)
    primary_weights = weights[weights["candidate"].eq("modality_expert_simplex")].copy()
    primary_removals = removals[
        removals["candidate"].eq("modality_expert_simplex")
    ].copy()
    if len(primary_weights) != 8 or len(primary_removals) != 8:
        raise ValueError("Primary expert diagnostics are incomplete")

    source_paths = [
        output_dir / "figure2_modality_audit_source.csv",
        output_dir / "figure2_joint_pca_block_source.csv",
        output_dir / "figure2_target_probe_source.csv",
        output_dir / "figure2_expert_diagnostics_source.csv",
    ]
    modality.to_csv(source_paths[0], index=False)
    blocks.to_csv(source_paths[1], index=False)
    target_probe.to_csv(source_paths[2], index=False)
    primary_weights.merge(
        primary_removals,
        left_on=["candidate", "target", "modality"],
        right_on=["candidate", "outcome", "modality"],
        how="outer",
        validate="one_to_one",
    ).to_csv(source_paths[3], index=False)

    fig = plt.figure(figsize=(7.20, 7.85))
    grid = fig.add_gridspec(3, 2, hspace=0.48, wspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, 0])
    subgrid = grid[2, 1].subgridspec(2, 1, hspace=0.16, height_ratios=[1, 1])
    ax_f1 = fig.add_subplot(subgrid[0, 0])
    ax_f2 = fig.add_subplot(subgrid[1, 0], sharex=ax_f1)

    reverse = modality.iloc[::-1]
    yy = np.arange(len(reverse))
    ax_a.barh(
        yy,
        reverse["dimension"],
        color=[MODALITY_COLORS[name] for name in reverse["modality"]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_a.set_xscale("log")
    ax_a.set_yticks(yy)
    ax_a.set_yticklabels(reverse["modality"].str.upper())
    ax_a.set_xlabel("Original feature dimension (log scale)")
    ax_a.set_title("Frozen feature widths")
    ax_a.grid(axis="x", color="#E6E6E6", lw=0.5)
    for yi, value in zip(yy, reverse["dimension"]):
        ax_a.text(value * 1.05, yi, str(int(value)), va="center", fontsize=6)
    panel_label(ax_a, "a")

    ax_b.hlines(
        yy,
        reverse["entropy_effective_rank"],
        reverse["pc_count_95"],
        color="#CFCFCF",
        lw=1.2,
    )
    ax_b.scatter(
        reverse["entropy_effective_rank"],
        yy,
        color="#3775BA",
        s=22,
        label="Entropy effective rank",
        zorder=3,
    )
    ax_b.scatter(
        reverse["pc_count_95"],
        yy,
        marker="x",
        color="#B64342",
        s=24,
        linewidth=1.1,
        label="PCs for 95% variance",
        zorder=3,
    )
    ax_b.set_yticks(yy)
    ax_b.set_yticklabels(reverse["modality"].str.upper())
    ax_b.set_xlabel("Components")
    ax_b.set_title("Low but unequal effective dimensionality")
    ax_b.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_b.legend(loc="lower right")
    panel_label(ax_b, "b")

    for row in modality.itertuples(index=False):
        ax_c.scatter(
            row.condition_feature_r2_descriptive,
            row.participant_feature_r2_descriptive,
            s=30,
            color=MODALITY_COLORS[row.modality],
            edgecolor="black",
            linewidth=0.4,
        )
        label_offsets = {
            "eeg": (-3, 6),
            "ecg": (4, -10),
            "eye": (4, 3),
            "head": (4, 3),
            "video": (4, 3),
        }
        ax_c.annotate(
            row.modality.upper(),
            (
                row.condition_feature_r2_descriptive,
                row.participant_feature_r2_descriptive,
            ),
            xytext=label_offsets[row.modality],
            textcoords="offset points",
            fontsize=6,
        )
    limit = max(
        modality["participant_feature_r2_descriptive"].max(),
        modality["condition_feature_r2_descriptive"].max(),
    ) * 1.1
    ax_c.plot([0, limit], [0, limit], color="#A8A8A8", ls="--", lw=0.8)
    ax_c.set_xlim(0, max(0.30, modality["condition_feature_r2_descriptive"].max() * 1.15))
    ax_c.set_ylim(0, max(0.70, modality["participant_feature_r2_descriptive"].max() * 1.08))
    ax_c.set_xlabel("Condition-explained feature variance")
    ax_c.set_ylabel("Participant-explained feature variance")
    ax_c.set_title("Participant structure dominates most blocks")
    ax_c.grid(color="#EAEAEA", lw=0.5)
    panel_label(ax_c, "c", x=-0.12, y=1.10)

    x_block = np.arange(len(blocks))
    width = 0.34
    ax_d.bar(
        x_block - width / 2,
        blocks["coordinate_share"] * 100,
        width,
        color="#D8D8D8",
        edgecolor="black",
        linewidth=0.4,
        label="Coordinate share",
    )
    ax_d.bar(
        x_block + width / 2,
        blocks["joint_pca8_loading_mass_share"] * 100,
        width,
        color=[MODALITY_COLORS[name] for name in blocks["modality"]],
        edgecolor="black",
        linewidth=0.4,
        label="PCA8 loading mass",
    )
    ax_d.set_xticks(x_block)
    ax_d.set_xticklabels(blocks["modality"].str.upper())
    ax_d.set_ylabel("Share (%)")
    large_mass = blocks.loc[
        blocks["modality"].isin(["ecg", "video"]), "joint_pca8_loading_mass_share"
    ].sum()
    ax_d.set_title(
        f"Joint PCA8 is width-weighted\nECG + video loading mass = {large_mass:.1%}"
    )
    ax_d.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax_d.legend(loc="upper right")
    panel_label(ax_d, "d")

    probe_matrix = (
        target_probe.pivot(index="modality", columns="target", values="delta_vs_condition")
        .loc[modality_order, ["relaxation", "discomfort"]]
        .to_numpy()
        * 1_000
    )
    probe_norm = symmetric_norm(probe_matrix, floor=1.0)
    image = ax_e.imshow(probe_matrix, cmap=DELTA_CMAP, norm=probe_norm, aspect="auto")
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(["Relaxation", "Discomfort"])
    ax_e.set_yticks(range(len(modality_order)))
    ax_e.set_yticklabels([name.upper() for name in modality_order])
    ax_e.set_title("Fixed fold-safe target probes")
    for i in range(probe_matrix.shape[0]):
        for j in range(probe_matrix.shape[1]):
            value = probe_matrix[i, j]
            color = (
                "white"
                if abs(value) > 0.58 * max(abs(probe_norm.vmin), probe_norm.vmax)
                else "#272727"
            )
            ax_e.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(image, ax=ax_e, fraction=0.045, pad=0.035)
    cbar.set_label(r"MAE difference ($\times 10^{-3}$)")
    cbar.ax.tick_params(labelsize=6)
    for spine in ax_e.spines.values():
        spine.set_visible(False)
    panel_label(ax_e, "e")

    x_mod = np.arange(len(no_eeg_order))
    bar_width = 0.34
    for target_index, target in enumerate(["relaxation", "discomfort"]):
        subset = (
            primary_weights[primary_weights["target"].eq(target)]
            .set_index("modality")
            .loc[no_eeg_order]
        )
        ax_f1.bar(
            x_mod + (target_index - 0.5) * bar_width,
            subset["mean_weight"],
            bar_width,
            color=TARGET_COLORS[target],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.35,
            label=target.capitalize(),
        )
    ax_f1.set_ylabel("Mean weight")
    ax_f1.set_ylim(0, 0.62)
    ax_f1.set_title("Expert weights and validation removal checks")
    ax_f1.legend(ncol=2, loc="upper right")
    ax_f1.tick_params(axis="x", labelbottom=False)
    ax_f1.grid(axis="y", color="#EAEAEA", lw=0.45)
    panel_label(ax_f1, "f")

    for target_index, target in enumerate(["relaxation", "discomfort"]):
        subset = (
            primary_removals[primary_removals["outcome"].eq(target)]
            .set_index("modality")
            .loc[no_eeg_order]
        )
        offset = (target_index - 0.5) * 0.16
        ax_f2.errorbar(
            x_mod + offset,
            subset["mean_delta_without_minus_full"] * 1_000,
            yerr=subset["std_delta_without_minus_full"] * 1_000,
            fmt=TARGET_MARKERS[target],
            color=TARGET_COLORS[target],
            markersize=3.7,
            elinewidth=0.75,
            capsize=1.8,
        )
    ax_f2.axhline(0, color="#767676", ls="--", lw=0.8)
    ax_f2.set_xticks(x_mod)
    ax_f2.set_xticklabels([name.upper() for name in no_eeg_order])
    ax_f2.set_ylabel(r"Removal Δ ($\times 10^{-3}$)")
    ax_f2.set_xlabel("Positive: removal worsened validation MAE")
    ax_f2.grid(axis="y", color="#EAEAEA", lw=0.45)

    fig.suptitle(
        "Feature audit supports protected blocks, but modality evidence remains descriptive",
        x=0.5,
        y=0.995,
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.004,
        "Panels c–f are descriptive diagnostics. Expert weights and removal deltas are not causal modality effects.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    figure_paths = save_bundle(fig, output_dir / "figure2_feature_and_modality_audit")
    return figure_paths, source_paths


def make_redundancy_figure(
    audit_dir: Path, output_dir: Path
) -> tuple[list[Path], list[Path]]:
    redundancy = read_csv(
        audit_dir / "cross_modal_redundancy_balanced_pca8_cka.csv",
        [
            "modality_a",
            "modality_b",
            "linear_cka_balanced_pca8",
            "linear_cka_balanced_pca8_after_participant_condition_residualization",
            "status",
        ],
    )
    source_path = output_dir / "figure3_cross_modal_redundancy_source.csv"
    redundancy.to_csv(source_path, index=False)
    modalities = ["eeg", "ecg", "eye", "head", "video"]

    def matrix(column: str) -> np.ndarray:
        values = np.full((len(modalities), len(modalities)), np.nan, dtype=float)
        positions = {name: index for index, name in enumerate(modalities)}
        for row in redundancy.itertuples(index=False):
            left = positions[row.modality_a]
            right = positions[row.modality_b]
            values[left, right] = float(getattr(row, column))
            values[right, left] = float(getattr(row, column))
        return values

    raw = matrix("linear_cka_balanced_pca8")
    residual = matrix(
        "linear_cka_balanced_pca8_after_participant_condition_residualization"
    )
    vmax = max(float(np.nanmax(raw)), float(np.nanmax(residual)), 0.55)
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.15), constrained_layout=True)
    images = []
    for ax, values, title, label in zip(
        axes,
        [raw, residual],
        [
            "Balanced PCA8 similarity",
            "After participant + condition residualization",
        ],
        ["a", "b"],
        strict=True,
    ):
        image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax)
        images.append(image)
        ax.set_xticks(range(len(modalities)))
        ax.set_xticklabels([name.upper() for name in modalities])
        ax.set_yticks(range(len(modalities)))
        ax.set_yticklabels([name.upper() for name in modalities])
        ax.set_title(title)
        for i in range(len(modalities)):
            for j in range(len(modalities)):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 0.55 * vmax else "#272727",
                    )
                elif i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#767676")
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, label, x=-0.11, y=1.04)
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.026, pad=0.025)
    cbar.set_label("Linear CKA")
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        "Cross-modal redundancy changes after removing cohort structure",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Equal-width eight-PC representations; descriptive post hoc supplement, not a selection criterion.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    figure_paths = save_bundle(fig, output_dir / "figure3_cross_modal_redundancy")
    return figure_paths, [source_path]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (artifact_root / "figures").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    figure1, source1 = make_outcome_figure(artifact_root / "evaluation", output_dir)
    figure2, source2 = make_feature_figure(artifact_root, output_dir)
    figure3, source3 = make_redundancy_figure(artifact_root / "audit", output_dir)
    outputs = figure1 + figure2 + figure3 + source1 + source2 + source3

    qa_path = output_dir / "figure_qa_notes.md"
    qa_path.write_text(
        "\n".join(
            [
                "# Figure QA notes",
                "",
                "- Backend: Python 3 / matplotlib only for drawing, previews, and exports.",
                "- Archetype: quantitative grid with formal effect estimates as the hero evidence.",
                "- Final width: 7.20 inches (approximately 183 mm, double column).",
                "- Exports: editable SVG, TrueType-text PDF, 600-dpi TIFF, and 300-dpi PNG preview.",
                "- Figure 1 n: nine participant clusters after averaging three seeds; metric is participant-macro MAE.",
                "- Figure 1 intervals/tests: 10,000 participant-cluster bootstrap resamples, exact 512 sign flips, Holm family of 18 tests.",
                "- Figure 2 diagnostics: descriptive only; no causal modality claim is encoded.",
                "- Figure 3 is a post hoc reproducibility supplement and did not influence the locked method panel.",
                "- Difference heatmaps use blue for lower MAE and red for higher MAE; method-comparison panels retain one fixed color per method.",
                "- Source-data CSVs are stored beside the exports and all generated files are hashed in figure_manifest.json.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(qa_path)
    manifest_path = output_dir / "figure_manifest.json"
    manifest = {
        "status": "passed",
        "backend": "python_matplotlib",
        "figure_contract": "no_confirmed_condition_only_gain",
        "generated_files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "figures": len(figure1) + len(figure2) + len(figure3),
                "source_tables": len(source1) + len(source2) + len(source3),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
