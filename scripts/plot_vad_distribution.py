"""Plot VAD label distribution for EgoEmotion raw dataset.

Outputs a 2x3 panel figure:
  (a) Valence histogram
  (b) Arousal histogram
  (c) Dominance histogram
  (d) Valence x Arousal joint heatmap (circumplex view)
  (e) Per-video V vs A scatter (bubble size = Dominance)
  (f) Per-participant V/A/D mean (rater bias strips)
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = "data/egoemotion_raw"
OUT = "figures/vad_analysis/vad_distribution.png"


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*/Session_*.csv"))):
        pid = os.path.basename(os.path.dirname(f))
        sess = os.path.basename(f).split("_")[1]
        d = pd.read_csv(f)
        d["participant"] = pid
        d["session"] = sess
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load()
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    dims = ["Valence", "Arousal", "Dominance"]
    colors = {"Valence": "#2E86AB", "Arousal": "#E63946", "Dominance": "#6A994E"}

    # (a-c) marginal histograms
    for i, dim in enumerate(dims):
        ax = fig.add_subplot(gs[0, i])
        vc = df[dim].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color=colors[dim], edgecolor="black", linewidth=0.6)
        mu, sd = df[dim].mean(), df[dim].std()
        ax.axvline(mu, color="k", linestyle="--", linewidth=1, label=f"μ={mu:.2f}\nσ={sd:.2f}")
        ax.set_xlabel(f"{dim} (1–7 Likert)")
        ax.set_ylabel("Count")
        ax.set_title(f"({chr(97 + i)}) {dim} distribution")
        ax.set_xticks(range(1, 8))
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        for x, y in zip(vc.index, vc.values):
            ax.text(x, y + 1, f"{y / len(df) * 100:.1f}%", ha="center", fontsize=8)

    # (d) V x A joint heatmap
    ax = fig.add_subplot(gs[1, 0])
    ct = pd.crosstab(df["Valence"], df["Arousal"]).reindex(
        index=range(1, 8), columns=range(1, 8), fill_value=0
    )
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        cbar_kws={"label": "Count"},
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.invert_yaxis()
    ax.set_title("(d) Valence × Arousal joint (circumplex)")
    ax.set_xlabel("Arousal")
    ax.set_ylabel("Valence")

    # (e) per-video mean V/A scatter, bubble = Dominance
    ax = fig.add_subplot(gs[1, 1])
    g = df.groupby(["Video Name", "Video Emotion"], as_index=False)[
        ["Valence", "Arousal", "Dominance"]
    ].mean()
    scat = ax.scatter(
        g["Valence"],
        g["Arousal"],
        s=(g["Dominance"] - 2.5) * 120 + 60,
        c=g["Valence"],
        cmap="RdYlBu",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    for _, r in g.iterrows():
        ax.annotate(
            f"{r['Video Name']}\n({r['Video Emotion']})",
            (r["Valence"], r["Arousal"]),
            fontsize=7,
            ha="center",
            va="center",
            xytext=(0, -18),
            textcoords="offset points",
        )
    ax.axvline(4, color="grey", linewidth=0.6, linestyle=":")
    ax.axhline(4, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xlabel("Valence (mean)")
    ax.set_ylabel("Arousal (mean)")
    ax.set_title("(e) Per-video V/A (size ∝ Dominance)")
    plt.colorbar(scat, ax=ax, shrink=0.8, label="Valence")

    # (f) per-participant rater bias strips
    ax = fig.add_subplot(gs[1, 2])
    pg = (
        df.groupby("participant")[dims]
        .mean()
        .reset_index()
        .melt(id_vars="participant", var_name="Dimension", value_name="Mean rating")
    )
    sns.stripplot(
        data=pg,
        x="Dimension",
        y="Mean rating",
        hue="Dimension",
        palette=colors,
        size=6,
        alpha=0.7,
        jitter=0.18,
        ax=ax,
        legend=False,
    )
    sns.boxplot(
        data=pg,
        x="Dimension",
        y="Mean rating",
        boxprops=dict(facecolor="none", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        width=0.45,
        ax=ax,
        fliersize=0,
    )
    ax.set_ylim(1, 7)
    ax.set_title("(f) Per-participant rater bias (n=28)")
    ax.set_xlabel("")
    ax.set_ylabel("Subject mean rating")

    fig.suptitle(
        "EgoEmotion VAD label distribution — 28 participants × 2 sessions × 9 videos (n=455)",
        fontsize=13,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
