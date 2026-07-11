"""Publication-style radar plots for PHLOP fine-tuning experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parent
MODELS = {
    "Qwen2-VL-2B": "qwen2vl_finetuned_metrics_static.json",
    "InternVL3-2B": "internvl3_finetuned_metrics_static.json",
    "SmolVLM": "smolvlm_finetuned_metrics_static.json",
}
DIFFICULTIES = ["easy", "medium", "hard", "very_hard"]
AXIS_LABELS = ["Overall", "Easy", "Medium", "Hard", "Very hard"]

RUN_LABELS = {
    "base": "Baseline",
    "easy": "Easy",
    "easy_medium": "Easy + medium",
    "hard": "Hard",
    "full": "All tasks",
}
RUN_ORDER = list(RUN_LABELS)
COLORS = {
    "base": "#9CA3AF",
    "easy": "#60A5FA",
    "easy_medium": "#8B5CF6",
    "hard": "#F59E0B",
    "full": "#059669",
}

# The dataset assigns each question family to one fixed difficulty.
QUESTION_DIFFICULTY = {
    "apparent_contradiction": "medium",
    "collision_presence": "easy",
    "direction_reversal": "medium",
    "energy_analysis_taxonomy": "hard",
    "event_sequence": "medium",
    "fastest_object": "medium",
    "four_hop_causation": "very_hard",
    "friction_coefficient_comparison": "medium",
    "friction_comparison": "medium",
    "friction_scaling": "hard",
    "indirect_causation": "very_hard",
    "mass_ratio": "medium",
    "momentum_conservation": "hard",
    "newtons_second_law": "medium",
    "object_count": "easy",
    "property_competition": "very_hard",
    "relative_velocity_decision": "medium",
    "relative_velocity_magnitude": "medium",
    "rolling_detection": "easy",
    "shape_distribution": "easy",
    "stationary_duration": "medium",
    "stationary_start_time": "medium",
    "stopped_objects_count": "easy",
    "temporal_consistency": "hard",
    "velocity_scaling": "hard",
}

QUESTION_FAMILIES = {
    "Object & event perception": {
        "collision_presence",
        "object_count",
        "rolling_detection",
        "shape_distribution",
        "stopped_objects_count",
    },
    "Motion & velocity": {
        "stationary_duration",
        "stationary_start_time",
        "fastest_object",
        "direction_reversal",
        "relative_velocity_decision",
        "relative_velocity_magnitude",
        "velocity_scaling",
    },
    "Forces & friction": {
        "friction_comparison",
        "friction_coefficient_comparison",
        "friction_scaling",
        "newtons_second_law",
        "mass_ratio",
    },
    "Momentum & energy": {
        "momentum_conservation",
        "energy_analysis_taxonomy",
    },
    "Temporal & causal reasoning": {
        "apparent_contradiction",
        "event_sequence",
        "temporal_consistency",
        "indirect_causation",
        "four_hop_causation",
        "property_competition",
    },
}


def _recipe(model_name: str) -> str | None:
    """Convert raw checkpoint names into stable recipe identifiers."""
    if model_name == "base":
        return "base"
    for recipe in ("easy_medium", "easy", "hard", "full"):
        if recipe in model_name:
            return recipe
    return None


def _clean(value: str) -> str:
    """Match answers robustly without changing their semantic content."""
    return " ".join((value or "").strip().lower().split())


def load_accuracy(metrics_path: str | Path, use_physics: bool = True) -> dict:
    """Aggregate official evaluator accuracy from question type to difficulty."""
    raw = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["static"]
    physics_key = "with_physics" if "with_physics" in raw else "physics"
    condition = raw[physics_key if use_physics else "no_physics"]
    result = {}
    for raw_name, payload in condition.items():
        recipe = _recipe(raw_name)
        if recipe is None:
            continue
        test = payload["test"]
        weighted_correct = defaultdict(float)
        totals = defaultdict(int)
        for question_type, metrics in test["per_question_type"].items():
            difficulty = QUESTION_DIFFICULTY.get(question_type)
            if difficulty is None:
                continue
            weighted_correct[difficulty] += metrics["accuracy"] * metrics["count"]
            totals[difficulty] += metrics["count"]
        result[recipe] = {"overall": test["answer_accuracy"]}
        result[recipe].update(
            {
                difficulty: weighted_correct[difficulty] / totals[difficulty]
                for difficulty in DIFFICULTIES
            }
        )
    return {recipe: result[recipe] for recipe in RUN_ORDER if recipe in result}


def build_summary(use_physics: bool = True) -> dict:
    return {
        model: load_accuracy(ROOT / filename, use_physics=use_physics)
        for model, filename in MODELS.items()
    }


def plot_radar(
    summary: dict,
    output_stem: str | Path | None = None,
    condition_label: str = "physics-augmented",
):
    """Draw and save a three-model radar figure (PNG, SVG, and PDF)."""
    categories = ["overall", *DIFFICULTIES]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(
        1, len(summary), figsize=(15.5, 5.8), subplot_kw={"projection": "polar"}
    )
    fig.patch.set_facecolor("#F8FAFC")

    for ax, (model, runs) in zip(np.atleast_1d(axes), summary.items()):
        ax.set_facecolor("#FFFFFF")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(AXIS_LABELS, fontsize=10, color="#111827")
        ax.tick_params(axis="x", pad=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["20", "40", "60", "80", "100%"], fontsize=8, color="#64748B"
        )
        ax.set_rlabel_position(18)
        ax.grid(color="#CBD5E1", linewidth=0.8, alpha=0.8)
        ax.spines["polar"].set_color("#CBD5E1")
        ax.set_title(model, y=1.12, fontsize=14, color="#0F172A")

        for recipe in RUN_ORDER:
            if recipe not in runs:
                continue
            values = [runs[recipe][key] for key in categories]
            values += values[:1]
            is_full = recipe == "full"
            ax.plot(
                closed_angles,
                values,
                color=COLORS[recipe],
                linewidth=3.2 if is_full else 1.7,
                alpha=1 if is_full else 0.82,
                marker="o",
                markersize=5 if is_full else 3.5,
                zorder=5 if is_full else 2,
            )
            ax.fill(
                closed_angles,
                values,
                color=COLORS[recipe],
                alpha=0.13 if is_full else 0.025,
                zorder=4 if is_full else 1,
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[r],
            lw=3.2 if r == "full" else 1.8,
            marker="o",
            markersize=5,
            label=RUN_LABELS[r],
        )
        for r in RUN_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        fontsize=10,
    )
    fig.suptitle(
        "Training on all task difficulties yields the broadest gains",
        x=0.5,
        y=1.02,
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.925,
        f"Test accuracy by difficulty · {condition_label} setting · radius is absolute accuracy",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.76, bottom=0.16, wspace=0.28)

    output_stem = (
        Path(output_stem) if output_stem else OUTPUT_DIR / "radar_finetuning_comparison"
    )
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_combined_radar(
    summary: dict,
    output_stem: str | Path | None = None,
    condition_label: str = "physics-augmented",
):
    """Draw one radar using the macro-average across all available VLMs."""
    categories = ["overall", *DIFFICULTIES]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    combined = {}
    for recipe in RUN_ORDER:
        combined[recipe] = {}
        for metric in categories:
            values = [
                runs[recipe][metric]
                for runs in summary.values()
                if recipe in runs and np.isfinite(runs[recipe][metric])
            ]
            combined[recipe][metric] = float(np.mean(values)) if values else np.nan

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 8.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(AXIS_LABELS, fontsize=12, color="#111827")
    ax.tick_params(axis="x", pad=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20", "40", "60", "80", "100%"], fontsize=9, color="#64748B")
    ax.set_rlabel_position(18)
    ax.grid(color="#CBD5E1", linewidth=0.9, alpha=0.85)
    ax.spines["polar"].set_color("#CBD5E1")

    for recipe in RUN_ORDER:
        values = [combined[recipe][key] for key in categories]
        values += values[:1]
        is_full = recipe == "full"
        ax.plot(
            closed_angles,
            values,
            color=COLORS[recipe],
            linewidth=3.8 if is_full else 2.0,
            alpha=1 if is_full else 0.85,
            marker="o",
            markersize=6 if is_full else 4.5,
            zorder=5 if is_full else 2,
        )
        ax.fill(
            closed_angles,
            values,
            color=COLORS[recipe],
            alpha=0.16 if is_full else 0.035,
            zorder=4 if is_full else 1,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[r],
            lw=3.5 if r == "full" else 2,
            marker="o",
            markersize=5,
            label=RUN_LABELS[r],
        )
        for r in RUN_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=10.5,
    )
    fig.suptitle(
        "All-model fine-tuning comparison",
        y=0.985,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.945,
        f"Macro-average across three VLMs · {condition_label} setting · absolute test accuracy",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.09, right=0.91, top=0.87, bottom=0.15)

    output_stem = (
        Path(output_stem) if output_stem else OUTPUT_DIR / "radar_all_models_combined"
    )
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def _annotate_heatmap(ax, matrix, *, percentage_points=False, threshold=None):
    """Add readable values to a heatmap, including signs for changes."""
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                label = "—"
                color = "#475569"
            elif percentage_points:
                label = f"{value * 100:+.1f}"
                color = (
                    "white"
                    if threshold is not None and abs(value) >= threshold
                    else "#0F172A"
                )
            else:
                label = f"{value * 100:.1f}"
                color = (
                    "white"
                    if threshold is not None and value >= threshold
                    else "#0F172A"
                )
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color=color,
            )


def plot_accuracy_heatmap(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot absolute evaluator accuracy for every model, recipe, and difficulty."""
    categories = ["overall", *DIFFICULTIES]
    conditions = [
        ("Physics augmented", physics_summary),
        ("No physics", no_physics_summary),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.5), constrained_layout=False)
    fig.patch.set_facecolor("#F8FAFC")
    image = None

    for row, model in enumerate(MODELS):
        for col, (condition, summary) in enumerate(conditions):
            ax = axes[row, col]
            matrix = np.array(
                [
                    [summary[model][recipe][metric] for metric in categories]
                    for recipe in RUN_ORDER
                ]
            )
            image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            _annotate_heatmap(ax, matrix, threshold=0.62)
            ax.set_xticks(range(len(categories)), AXIS_LABELS, fontsize=10)
            row_labels = [RUN_LABELS[r] for r in RUN_ORDER] if col == 0 else []
            ax.set_yticks(range(len(RUN_ORDER)), row_labels, fontsize=10)
            ax.tick_params(length=0)
            ax.set_title(
                condition if row == 0 else "", fontsize=13, pad=12, color="#334155"
            )
            if col == 0:
                ax.set_ylabel(
                    model, fontsize=13, fontweight="bold", labelpad=15, color="#0F172A"
                )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(RUN_ORDER), 1), minor=True)
            ax.grid(which="minor", color="#F8FAFC", linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "Fine-tuning accuracy across task difficulty",
        y=0.985,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.952,
        "Official evaluator accuracy (%) · higher is better",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.15, right=0.91, top=0.91, bottom=0.07, hspace=0.34, wspace=0.18
    )
    colorbar_ax = fig.add_axes([0.93, 0.16, 0.015, 0.65])
    cbar = fig.colorbar(image, cax=colorbar_ax)
    cbar.set_ticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1], labels=["0", "20", "40", "60", "80", "100%"]
    )
    cbar.outline.set_visible(False)

    output_stem = (
        Path(output_stem) if output_stem else OUTPUT_DIR / "heatmap_absolute_accuracy"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_improvement_heatmap(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot percentage-point change from each model's own baseline."""
    categories = ["overall", *DIFFICULTIES]
    tuned_recipes = [r for r in RUN_ORDER if r != "base"]
    conditions = [
        ("Physics augmented", physics_summary),
        ("No physics", no_physics_summary),
    ]

    all_deltas = []
    for _, summary in conditions:
        for model in MODELS:
            baseline = np.array([summary[model]["base"][m] for m in categories])
            for recipe in tuned_recipes:
                values = np.array([summary[model][recipe][m] for m in categories])
                all_deltas.extend(values - baseline)
    limit = max(0.1, float(np.nanmax(np.abs(all_deltas))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.2), constrained_layout=False)
    fig.patch.set_facecolor("#F8FAFC")
    image = None
    for row, model in enumerate(MODELS):
        for col, (condition, summary) in enumerate(conditions):
            ax = axes[row, col]
            baseline = np.array([summary[model]["base"][m] for m in categories])
            matrix = np.array(
                [
                    np.array([summary[model][recipe][m] for m in categories]) - baseline
                    for recipe in tuned_recipes
                ]
            )
            image = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
            _annotate_heatmap(
                ax, matrix, percentage_points=True, threshold=limit * 0.58
            )
            ax.set_xticks(range(len(categories)), AXIS_LABELS, fontsize=10)
            row_labels = [RUN_LABELS[r] for r in tuned_recipes] if col == 0 else []
            ax.set_yticks(range(len(tuned_recipes)), row_labels, fontsize=10)
            ax.tick_params(length=0)
            ax.set_title(
                condition if row == 0 else "", fontsize=13, pad=12, color="#334155"
            )
            if col == 0:
                ax.set_ylabel(
                    model, fontsize=13, fontweight="bold", labelpad=15, color="#0F172A"
                )
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(tuned_recipes), 1), minor=True)
            ax.grid(which="minor", color="#F8FAFC", linewidth=3)
            ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "Improvement over each model’s baseline",
        y=0.985,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.948,
        "Accuracy change in percentage points · green indicates improvement",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.15, right=0.91, top=0.90, bottom=0.07, hspace=0.37, wspace=0.18
    )
    colorbar_ax = fig.add_axes([0.93, 0.16, 0.015, 0.65])
    cbar = fig.colorbar(image, cax=colorbar_ax)
    ticks = np.linspace(-limit, limit, 5)
    cbar.set_ticks(ticks, labels=[f"{v * 100:+.0f}" for v in ticks])
    cbar.set_label("Percentage points", fontsize=9)
    cbar.outline.set_visible(False)

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "heatmap_improvement_over_baseline"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_physics_dumbbell(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Compare no-physics and physics-augmented accuracy with dumbbells."""
    categories = ["overall", *DIFFICULTIES]
    recipe_labels = [RUN_LABELS[r] for r in RUN_ORDER]
    no_color = "#64748B"
    physics_color = "#0D9488"

    fig, axes = plt.subplots(3, 5, figsize=(18, 11.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")

    for row, model in enumerate(MODELS):
        for col, metric in enumerate(categories):
            ax = axes[row, col]
            ax.set_facecolor("#FFFFFF")
            y = np.arange(len(RUN_ORDER))
            no_values = np.array(
                [no_physics_summary[model][r][metric] * 100 for r in RUN_ORDER]
            )
            physics_values = np.array(
                [physics_summary[model][r][metric] * 100 for r in RUN_ORDER]
            )

            for ypos, no_value, physics_value in zip(y, no_values, physics_values):
                gain = physics_value >= no_value
                ax.plot(
                    [no_value, physics_value],
                    [ypos, ypos],
                    color="#5EEAD4" if gain else "#FCA5A5",
                    linewidth=3,
                    solid_capstyle="round",
                    zorder=1,
                )
            ax.scatter(
                no_values,
                y,
                s=48,
                color=no_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.scatter(
                physics_values,
                y,
                s=58,
                color=physics_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

            ax.set_xlim(0, 102)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(recipe_labels, fontsize=9.5)
            ax.invert_yaxis()
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
            ax.tick_params(axis="both", length=0, pad=5)
            ax.tick_params(axis="y", labelleft=(col == 0))
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(
                    AXIS_LABELS[col],
                    fontsize=12.5,
                    fontweight="bold",
                    color="#0F172A",
                    pad=12,
                )
            if col == 0:
                ax.set_ylabel(
                    model,
                    fontsize=12.5,
                    fontweight="bold",
                    color="#0F172A",
                    labelpad=20,
                )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=no_color,
            markeredgecolor="white",
            markersize=8,
            label="No physics",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=physics_color,
            markeredgecolor="white",
            markersize=8,
            label="Physics augmented",
        ),
        Line2D([0], [0], color="#5EEAD4", lw=3, label="Gain with physics"),
        Line2D([0], [0], color="#FCA5A5", lw=3, label="Regression with physics"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=10,
    )
    fig.suptitle(
        "Effect of physics augmentation",
        y=0.982,
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.948,
        "Each line connects the same model and tuning recipe across evaluation conditions",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.89, bottom=0.10, hspace=0.35, wspace=0.16
    )

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "physics_augmentation_dumbbell"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def average_across_models(summary: dict) -> dict:
    """Macro-average every recipe/metric across the available model families."""
    categories = ["overall", *DIFFICULTIES]
    return {
        recipe: {
            metric: float(
                np.mean(
                    [
                        runs[recipe][metric]
                        for runs in summary.values()
                        if recipe in runs
                    ]
                )
            )
            for metric in categories
        }
        for recipe in RUN_ORDER
    }


def plot_average_accuracy_heatmap(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot absolute accuracy macro-averaged across all model families."""
    categories = ["overall", *DIFFICULTIES]
    conditions = [
        ("Physics augmented", average_across_models(physics_summary)),
        ("No physics", average_across_models(no_physics_summary)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    fig.patch.set_facecolor("#F8FAFC")
    image = None
    for col, (condition, summary) in enumerate(conditions):
        ax = axes[col]
        matrix = np.array([[summary[r][m] for m in categories] for r in RUN_ORDER])
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        _annotate_heatmap(ax, matrix, threshold=0.62)
        ax.set_xticks(range(len(categories)), AXIS_LABELS, fontsize=10)
        labels = [RUN_LABELS[r] for r in RUN_ORDER] if col == 0 else []
        ax.set_yticks(range(len(RUN_ORDER)), labels, fontsize=10)
        ax.set_title(condition, fontsize=13, fontweight="bold", pad=12, color="#334155")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(RUN_ORDER), 1), minor=True)
        ax.grid(which="minor", color="#F8FAFC", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "All-model average accuracy",
        y=0.98,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.92,
        "Macro-average across three VLMs · official evaluator accuracy (%)",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.13, right=0.90, top=0.80, bottom=0.14, wspace=0.16)
    colorbar_ax = fig.add_axes([0.92, 0.18, 0.016, 0.58])
    cbar = fig.colorbar(image, cax=colorbar_ax)
    cbar.set_ticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1], labels=["0", "20", "40", "60", "80", "100%"]
    )
    cbar.outline.set_visible(False)

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "heatmap_average_all_models_accuracy"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_average_improvement_heatmap(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot average percentage-point improvement over model-specific baselines."""
    categories = ["overall", *DIFFICULTIES]
    tuned = [r for r in RUN_ORDER if r != "base"]
    conditions = [
        ("Physics augmented", average_across_models(physics_summary)),
        ("No physics", average_across_models(no_physics_summary)),
    ]
    matrices = []
    for _, summary in conditions:
        baseline = np.array([summary["base"][m] for m in categories])
        matrices.append(
            np.array(
                [
                    np.array([summary[r][m] for m in categories]) - baseline
                    for r in tuned
                ]
            )
        )
    limit = max(0.1, float(np.nanmax(np.abs(matrices))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    fig.patch.set_facecolor("#F8FAFC")
    image = None
    for col, ((condition, _), matrix) in enumerate(zip(conditions, matrices)):
        ax = axes[col]
        image = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        _annotate_heatmap(ax, matrix, percentage_points=True, threshold=limit * 0.58)
        ax.set_xticks(range(len(categories)), AXIS_LABELS, fontsize=10)
        labels = [RUN_LABELS[r] for r in tuned] if col == 0 else []
        ax.set_yticks(range(len(tuned)), labels, fontsize=10)
        ax.set_title(condition, fontsize=13, fontweight="bold", pad=12, color="#334155")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tuned), 1), minor=True)
        ax.grid(which="minor", color="#F8FAFC", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "All-model average improvement over baseline",
        y=0.98,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.91,
        "Mean accuracy change across three VLMs, in percentage points",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.13, right=0.90, top=0.78, bottom=0.14, wspace=0.16)
    colorbar_ax = fig.add_axes([0.92, 0.18, 0.016, 0.55])
    cbar = fig.colorbar(image, cax=colorbar_ax)
    ticks = np.linspace(-limit, limit, 5)
    cbar.set_ticks(ticks, labels=[f"{v * 100:+.0f}" for v in ticks])
    cbar.set_label("Percentage points", fontsize=9)
    cbar.outline.set_visible(False)

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "heatmap_average_all_models_improvement"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_average_physics_dumbbell(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot the average effect of physics augmentation across all VLMs."""
    categories = ["overall", *DIFFICULTIES]
    physics = average_across_models(physics_summary)
    no_physics = average_across_models(no_physics_summary)
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    y = np.arange(len(RUN_ORDER))

    for col, metric in enumerate(categories):
        ax = axes[col]
        ax.set_facecolor("#FFFFFF")
        no_values = np.array([no_physics[r][metric] * 100 for r in RUN_ORDER])
        physics_values = np.array([physics[r][metric] * 100 for r in RUN_ORDER])
        for ypos, no_value, physics_value in zip(y, no_values, physics_values):
            ax.plot(
                [no_value, physics_value],
                [ypos, ypos],
                color="#5EEAD4" if physics_value >= no_value else "#FCA5A5",
                linewidth=3.5,
                solid_capstyle="round",
                zorder=1,
            )
        ax.scatter(no_values, y, s=55, color="#64748B", edgecolor="white", zorder=3)
        ax.scatter(
            physics_values, y, s=65, color="#0D9488", edgecolor="white", zorder=4
        )
        ax.set_xlim(0, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([RUN_LABELS[r] for r in RUN_ORDER], fontsize=10)
        ax.tick_params(axis="y", labelleft=(col == 0))
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.tick_params(length=0)
        ax.set_title(AXIS_LABELS[col], fontsize=12.5, fontweight="bold", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#64748B",
            markeredgecolor="white",
            markersize=8,
            label="No physics",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#0D9488",
            markeredgecolor="white",
            markersize=8,
            label="Physics augmented",
        ),
        Line2D([0], [0], color="#5EEAD4", lw=3, label="Gain with physics"),
        Line2D([0], [0], color="#FCA5A5", lw=3, label="Regression with physics"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=10,
    )
    fig.suptitle(
        "Average effect of physics augmentation",
        y=0.98,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.91,
        "Macro-average across Qwen2-VL-2B, InternVL3-2B, and SmolVLM",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.20, wspace=0.17)

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "physics_augmentation_dumbbell_average_all_models"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def load_question_family_accuracy(metrics_path: str | Path, use_physics: bool) -> dict:
    """Return count-weighted evaluator accuracy by conceptual question family."""
    raw = json.loads(Path(metrics_path).read_text(encoding="utf-8"))["static"]
    physics_key = "with_physics" if "with_physics" in raw else "physics"
    condition = raw[physics_key if use_physics else "no_physics"]
    selected = {}
    for raw_name, payload in condition.items():
        recipe = _recipe(raw_name)
        if recipe not in {"base", "full"}:
            continue
        question_metrics = payload["test"]["per_question_type"]
        selected[recipe] = {}
        for family, question_types in QUESTION_FAMILIES.items():
            weighted_correct = 0.0
            total = 0
            for question_type in question_types:
                metrics = question_metrics.get(question_type)
                if metrics is None:
                    continue
                weighted_correct += metrics["accuracy"] * metrics["count"]
                total += metrics["count"]
            selected[recipe][family] = weighted_correct / total if total else np.nan
    return selected


def plot_question_family_improvement(output_stem: str | Path | None = None):
    """Plot all-task tuning gains over baseline for conceptual QA families."""
    families = list(QUESTION_FAMILIES)
    condition_data = {}
    for condition_label, use_physics in (
        ("Physics augmented", True),
        ("No physics", False),
    ):
        condition_data[condition_label] = {}
        for model, filename in MODELS.items():
            values = load_question_family_accuracy(ROOT / filename, use_physics)
            condition_data[condition_label][model] = (
                np.array(
                    [
                        values["full"][family] - values["base"][family]
                        for family in families
                    ]
                )
                * 100
            )

    all_values = np.concatenate(
        [values for models in condition_data.values() for values in models.values()]
    )
    low = min(-5, float(np.nanmin(all_values)) - 5)
    high = max(5, float(np.nanmax(all_values)) + 9)
    model_colors = {
        "Qwen2-VL-2B": "#3B82F6",
        "InternVL3-2B": "#8B5CF6",
        "SmolVLM": "#F59E0B",
        "All-model average": "#059669",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    base_y = np.arange(len(families))
    bar_height = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_height

    for col, (condition, models) in enumerate(condition_data.items()):
        ax = axes[col]
        ax.set_facecolor("#FFFFFF")
        display = dict(models)
        display["All-model average"] = np.mean(list(models.values()), axis=0)
        for offset, (label, values) in zip(offsets, display.items()):
            bars = ax.barh(
                base_y + offset,
                values,
                height=bar_height * 0.88,
                color=model_colors[label],
                label=label,
                alpha=0.95,
            )
            for bar, value in zip(bars, values):
                x = value + (1.0 if value >= 0 else -1.0)
                ax.text(
                    x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:+.1f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8,
                    color="#0F172A",
                )
        ax.axvline(0, color="#334155", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_yticks(base_y)
        ax.set_yticklabels(families, fontsize=10.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlabel("Accuracy improvement (percentage points)", fontsize=10)
        ax.set_title(condition, fontsize=14, fontweight="bold", pad=13, color="#334155")
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=10,
    )
    fig.suptitle(
        "Where all-task tuning improves physics reasoning",
        y=0.985,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.945,
        "Count-weighted question-family accuracy · all-task checkpoint minus model baseline",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.86, bottom=0.14, wspace=0.12)

    output_stem = (
        Path(output_stem) if output_stem else OUTPUT_DIR / "question_family_improvement"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_generalization_matrix(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Plot cross-difficulty transfer, averaged equally across model families."""
    test_metrics = DIFFICULTIES
    tuning_recipes = ["easy", "easy_medium", "hard", "full"]
    conditions = [
        ("Physics augmented", average_across_models(physics_summary)),
        ("No physics", average_across_models(no_physics_summary)),
    ]
    matrices = [
        np.array([[summary[r][m] for m in test_metrics] for r in tuning_recipes])
        for _, summary in conditions
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    fig.patch.set_facecolor("#F8FAFC")
    image = None
    for col, ((condition, _), matrix) in enumerate(zip(conditions, matrices)):
        ax = axes[col]
        image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
        _annotate_heatmap(ax, matrix, threshold=0.62)
        ax.set_xticks(
            range(len(test_metrics)),
            ["Easy", "Medium", "Hard", "Very hard"],
            fontsize=10,
        )
        labels = [RUN_LABELS[r] for r in tuning_recipes] if col == 0 else []
        ax.set_yticks(range(len(tuning_recipes)), labels, fontsize=10)
        ax.set_xlabel("Evaluation difficulty", fontsize=10, labelpad=8)
        if col == 0:
            ax.set_ylabel("Training data", fontsize=10, labelpad=10)
        ax.set_title(
            condition, fontsize=13.5, fontweight="bold", pad=12, color="#334155"
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(test_metrics), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tuning_recipes), 1), minor=True)
        ax.grid(which="minor", color="#F8FAFC", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        "Generalization across task difficulty",
        y=0.985,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.925,
        "All-model macro-average test accuracy (%)",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.14, right=0.89, top=0.80, bottom=0.15, wspace=0.15)
    colorbar_ax = fig.add_axes([0.91, 0.19, 0.016, 0.54])
    cbar = fig.colorbar(image, cax=colorbar_ax)
    cbar.set_ticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1], labels=["0", "20", "40", "60", "80", "100%"]
    )
    cbar.outline.set_visible(False)

    output_stem = (
        Path(output_stem) if output_stem else OUTPUT_DIR / "generalization_matrix"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_all_task_gain_vs_specialized(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Compare all-task tuning with the strongest specialized checkpoint."""
    specialized = ["easy", "easy_medium", "hard"]
    conditions = [
        ("Physics augmented", physics_summary),
        ("No physics", no_physics_summary),
    ]
    colors = {"Qwen2-VL-2B": "#3B82F6", "InternVL3-2B": "#8B5CF6", "SmolVLM": "#F59E0B"}
    gains_by_condition = {}
    for condition, summary in conditions:
        gains_by_condition[condition] = {
            model: np.array(
                [
                    summary[model]["full"][metric]
                    - max(summary[model][recipe][metric] for recipe in specialized)
                    for metric in DIFFICULTIES
                ]
            )
            * 100
            for model in MODELS
        }
    all_gains = np.concatenate(
        [values for models in gains_by_condition.values() for values in models.values()]
    )
    bound = max(8, float(np.nanmax(np.abs(all_gains))) + 5)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    y = np.arange(len(DIFFICULTIES))
    height = 0.22
    offsets = [-height, 0, height]
    for col, (condition, models) in enumerate(gains_by_condition.items()):
        ax = axes[col]
        ax.set_facecolor("#FFFFFF")
        for offset, (model, values) in zip(offsets, models.items()):
            bars = ax.barh(
                y + offset,
                values,
                height=height * 0.85,
                color=colors[model],
                label=model,
            )
            for bar, value in zip(bars, values):
                display_value = 0.0 if abs(value) < 0.05 else value
                ax.text(
                    value + (0.7 if value >= 0 else -0.7),
                    bar.get_y() + bar.get_height() / 2,
                    f"{display_value:+.1f}",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=8.5,
                    color="#0F172A",
                )
        ax.axvline(0, color="#334155", linewidth=1.1)
        ax.set_xlim(-bound, bound)
        ax.set_yticks(y)
        ax.set_yticklabels(["Easy", "Medium", "Hard", "Very hard"], fontsize=10.5)
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(
            condition, fontsize=13.5, fontweight="bold", pad=13, color="#334155"
        )
        ax.set_xlabel("All-task gain (percentage points)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Evaluation difficulty", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=10,
    )
    fig.suptitle(
        "All-task tuning versus the best specialized checkpoint",
        y=0.985,
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.93,
        "Positive values favor all-task training; specialized candidates: easy, easy + medium, and hard",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.16, wspace=0.12)

    output_stem = (
        Path(output_stem)
        if output_stem
        else OUTPUT_DIR / "all_task_gain_vs_best_specialized"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def plot_model_consistency(
    physics_summary: dict,
    no_physics_summary: dict,
    output_stem: str | Path | None = None,
):
    """Show mean, standard deviation, and individual VLM results per recipe."""
    categories = ["overall", *DIFFICULTIES]
    conditions = [
        ("Physics augmented", physics_summary),
        ("No physics", no_physics_summary),
    ]
    model_colors = ["#3B82F6", "#8B5CF6", "#F59E0B"]
    mean_color = "#059669"
    y = np.arange(len(RUN_ORDER))

    fig, axes = plt.subplots(2, 5, figsize=(18, 9.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    for row, (condition, summary) in enumerate(conditions):
        for col, metric in enumerate(categories):
            ax = axes[row, col]
            ax.set_facecolor("#FFFFFF")
            matrix = np.array(
                [
                    [summary[model][recipe][metric] * 100 for model in MODELS]
                    for recipe in RUN_ORDER
                ]
            )
            means = np.mean(matrix, axis=1)
            stds = np.std(matrix, axis=1, ddof=1)

            ax.errorbar(
                means,
                y,
                xerr=stds,
                fmt="o",
                color=mean_color,
                ecolor="#6EE7B7",
                elinewidth=5,
                capsize=5,
                capthick=1.5,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=3,
            )
            jitter = [-0.11, 0, 0.11]
            for model_index, (model, color, dy) in enumerate(
                zip(MODELS, model_colors, jitter)
            ):
                ax.scatter(
                    matrix[:, model_index],
                    y + dy,
                    s=28,
                    color=color,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.9,
                    zorder=4,
                )

            ax.set_xlim(0, 102)
            ax.set_ylim(len(RUN_ORDER) - 0.5, -0.5)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels([RUN_LABELS[r] for r in RUN_ORDER], fontsize=9.5)
            ax.tick_params(axis="y", labelleft=(col == 0))
            ax.tick_params(length=0)
            ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(
                    AXIS_LABELS[col],
                    fontsize=12.5,
                    fontweight="bold",
                    color="#0F172A",
                    pad=12,
                )
            if col == 0:
                ax.set_ylabel(
                    condition,
                    fontsize=12.5,
                    fontweight="bold",
                    color="#334155",
                    labelpad=18,
                )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#6EE7B7",
            lw=5,
            markerfacecolor=mean_color,
            markeredgecolor="white",
            markersize=8,
            label="Mean ± 1 SD",
        ),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                label=model,
            )
            for model, color in zip(MODELS, model_colors)
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        fontsize=10,
    )
    fig.suptitle(
        "Consistency of fine-tuning performance across models",
        y=0.985,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.945,
        "Average accuracy and between-model variation across three VLMs",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(
        left=0.11, right=0.98, top=0.86, bottom=0.12, hspace=0.28, wspace=0.16
    )

    output_stem = Path(output_stem) if output_stem else OUTPUT_DIR / "model_consistency"
    for suffix in (".png", ".pdf"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    return fig


def save_summary(summary: dict, path: str | Path | None = None) -> None:
    path = Path(path) if path else OUTPUT_DIR / "radar_metrics_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Use the no-physics evaluation condition instead of physics-augmented results.",
    )
    args = parser.parse_args()
    use_physics = not args.no_physics
    suffix = "with_physics" if use_physics else "no_physics"

    data = build_summary(use_physics=use_physics)
    save_summary(data, OUTPUT_DIR / f"radar_metrics_summary_{suffix}.json")
    condition_label = "physics-augmented" if use_physics else "no-physics"
    plot_radar(
        data, OUTPUT_DIR / f"radar_finetuning_comparison_{suffix}", condition_label
    )
    plot_combined_radar(
        data, OUTPUT_DIR / f"radar_all_models_combined_{suffix}", condition_label
    )
    physics_data = build_summary(use_physics=True)
    no_physics_data = build_summary(use_physics=False)
    plot_accuracy_heatmap(physics_data, no_physics_data)
    plot_improvement_heatmap(physics_data, no_physics_data)
    plot_physics_dumbbell(physics_data, no_physics_data)
    plot_average_accuracy_heatmap(physics_data, no_physics_data)
    plot_average_improvement_heatmap(physics_data, no_physics_data)
    plot_average_physics_dumbbell(physics_data, no_physics_data)
    plot_question_family_improvement()
    plot_generalization_matrix(physics_data, no_physics_data)
    plot_all_task_gain_vs_specialized(physics_data, no_physics_data)
    plot_model_consistency(physics_data, no_physics_data)
    print(f"Created per-model and all-model {suffix} radar plots in PNG, SVG, and PDF")
    print("Created absolute-accuracy and baseline-improvement heatmaps in PNG and PDF")
    print("Created physics-augmentation dumbbell chart in PNG and PDF")
    print("Created all-model average heatmaps and dumbbell chart in PNG and PDF")
    print("Created question-family improvement chart in PNG and PDF")
    print("Created generalization matrix and all-task gain chart in PNG and PDF")
    print("Created model-consistency chart in PNG and PDF")
