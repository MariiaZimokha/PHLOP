"""Generate radar/spider charts of PHLOP reasoning-category accuracy.

Reads ``category_rollup.csv`` (columns: model, physics, condition, Perception,
Dynamics, Temporal, Causal) and renders one polygon per model over the four
reasoning-category axes.

Outputs (written to ``figures/``):
  * ``radar_categories.{pdf,png}``        -- single radar, best condition.
  * ``radar_categories_compare.{pdf,png}`` -- 2-panel no-physics vs with-physics.

Run:  python make_radar.py
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-phlop")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["Perception", "Dynamics", "Temporal", "Causal"]
MODELS = ["SmolVLM", "Qwen2-VL", "InternVL3"]
DEFAULT_CONDITION = "Full"


def _load():
    df = pd.read_csv(HERE / "category_rollup.csv")
    # accuracies are stored as fractions in [0, 1]
    return df


def _angles():
    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    return angles + angles[:1]  # close the loop


def _plot_panel(ax, df, physics, condition, title):
    angles = _angles()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="gray")
    ax.set_rlabel_position(180 / len(CATEGORIES))

    for model in MODELS:
        row = df[
            (df["model"] == model)
            & (df["physics"] == physics)
            & (df["condition"] == condition)
        ]
        if row.empty:
            continue
        values = row[CATEGORIES].iloc[0].astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=model)
        ax.fill(angles, values, alpha=0.10)

    ax.set_title(title, fontsize=11, pad=14)


def make_single(df, physics="With physics hint", condition=DEFAULT_CONDITION):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"polar": True})
    _plot_panel(ax, df, physics, condition, f"{condition} curriculum ({physics})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=9, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"radar_categories.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_compare(df, condition=DEFAULT_CONDITION):
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.6), subplot_kw={"polar": True}
    )
    _plot_panel(axes[0], df, "No physics hint", condition, "No physics hint")
    _plot_panel(axes[1], df, "With physics hint", condition, "With physics hint")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(f"Reasoning-category accuracy ({condition} curriculum)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"radar_categories_compare.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    df = _load()
    make_single(df)
    make_compare(df)
    print(f"Wrote radar figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
