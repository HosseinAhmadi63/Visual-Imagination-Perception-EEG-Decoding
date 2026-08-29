from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from mne.viz import plot_topomap

from vip_eeg.config import project_path, stage_root
from vip_eeg.data.topomaps import make_info
from vip_eeg.utils import read_json, write_json


def save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def protocol_figure(config: dict[str, Any], output: Path) -> None:
    segments = [
        ("Cue", 1.0, "#2E4057"),
        ("Mask", 0.5, "#9AA6B2"),
        ("Perception", 3.0, "#E67E22"),
        ("Mask", 0.5, "#9AA6B2"),
        ("Imagination", 4.0, "#7D3C98"),
    ]
    figure, axes = plt.subplots(figsize=(11, 2.5))
    start = 0.0
    for label, duration, color in segments:
        axes.barh(0, duration, left=start, height=0.55, color=color, edgecolor="black")
        axes.text(
            start + duration / 2,
            0,
            f"{label}\n{duration:g} s",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        start += duration
    axes.set_xlim(0, start)
    axes.set_ylim(-0.65, 0.65)
    axes.set_yticks([])
    axes.set_xlabel("Time from pictorial trial cue (s)")
    axes.set_title("Pictorial trial protocol repeated five times per block")
    axes.spines[["left", "right", "top"]].set_visible(False)
    save_figure(figure, output)


def load_archive(config: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    path = project_path(config, config["project"]["epoch_root"]) / f"{key}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def scalp_value(archive: dict[str, np.ndarray], task: str, time_seconds: float) -> np.ndarray:
    sampling_rate = float(archive["sampling_rate"])
    index = int(round(time_seconds * sampling_rate))
    return archive[task][:, index].astype(np.float64) * 1e6


def example_topomaps(config: dict[str, Any], output: Path) -> None:
    archive = load_archive(config, "18_1")
    info = make_info(
        archive["channel_names"], archive["positions"], float(archive["sampling_rate"])
    )
    panels = [
        ("imagination", 0.400),
        ("imagination", 0.420),
        ("perception", 0.400),
        ("perception", 0.415),
    ]
    values = [scalp_value(archive, task, time) for task, time in panels]
    limit = max(float(np.max(np.abs(value))) for value in values)
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    image = None
    for axes_item, (task, time), data in zip(axes.flat, panels, values, strict=True):
        image, _ = plot_topomap(
            data,
            info,
            axes=axes_item,
            show=False,
            cmap=config["topomaps"]["cmap"],
            sensors=True,
            contours=int(config["topomaps"]["contours"]),
            res=int(config["topomaps"]["interpolation_resolution"]),
            extrapolate=config["topomaps"]["extrapolate"],
            outlines=config["topomaps"]["outlines"],
            border=config["topomaps"]["border"],
            vlim=(-limit, limit),
        )
        axes_item.set_title(f"{task.capitalize()} {time:.3f} s")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Amplitude (µV)")
    figure.suptitle("Selected topomaps for recording 18_1")
    save_figure(figure, output)


def difference_topomaps(config: dict[str, Any], output: Path) -> None:
    archive = load_archive(config, "18_1")
    info = make_info(
        archive["channel_names"], archive["positions"], float(archive["sampling_rate"])
    )
    settings = config["statistics"]["difference_times"]
    panels = []
    for task in ("imagination", "perception"):
        first, second = [float(value) for value in settings[task]]
        difference = scalp_value(archive, task, second) - scalp_value(archive, task, first)
        panels.append((task, first, second, difference))
    limit = max(float(np.max(np.abs(item[3]))) for item in panels)
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    image = None
    for axes_item, (task, first, second, data) in zip(axes, panels, strict=True):
        image, _ = plot_topomap(
            data,
            info,
            axes=axes_item,
            show=False,
            cmap=config["topomaps"]["cmap"],
            sensors=True,
            contours=int(config["topomaps"]["contours"]),
            res=int(config["topomaps"]["interpolation_resolution"]),
            extrapolate=config["topomaps"]["extrapolate"],
            outlines=config["topomaps"]["outlines"],
            border=config["topomaps"]["border"],
            vlim=(-limit, limit),
        )
        axes_item.set_title(f"{task.capitalize()} ({second:.3f} - {first:.3f}) s")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Amplitude change (µV)")
    figure.suptitle("Difference topomaps for recording 18_1")
    save_figure(figure, output)


def random_forest_roc(source: Path, output: Path) -> None:
    curve = pd.read_csv(source / "roc_curve.csv")
    summary = read_json(source / "summary.json")
    auc_value = summary["holdout"]["roc_auc"]
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    axes.plot(
        curve["false_positive_rate"],
        curve["true_positive_rate"],
        color="#E67E22",
        linewidth=2.2,
        label=f"RF ROC (AUC = {auc_value:.2f})",
    )
    axes.plot([0, 1], [0, 1], linestyle="--", color="#2E86C1", label="Chance")
    axes.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Random-forest ROC curve",
    )
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right")
    save_figure(figure, output)


def permutation_histogram(source: Path, output: Path) -> None:
    scores = pd.read_csv(source / "permutation_scores.csv")
    summary = read_json(source / "summary.json")
    observed = summary["permutation"]["observed_cross_validated_accuracy"]
    p_value = summary["permutation"]["p_value"]
    figure, axes = plt.subplots(figsize=(7.0, 5.2))
    axes.hist(scores["accuracy"], bins=20, density=True, color="#5DADE2", alpha=0.85)
    axes.axvline(observed, color="#C0392B", linewidth=2.2, label=f"Observed = {observed:.3f}")
    axes.set(
        xlabel="Accuracy",
        ylabel="Density",
        title=f"Label-permutation distribution (p = {p_value:.4f})",
    )
    axes.legend(loc="upper right")
    axes.grid(axis="y", alpha=0.25)
    save_figure(figure, output)


def clustering_figure(source: Path, output: Path) -> None:
    table = pd.read_csv(source / "pca_clusters.csv")
    summary = read_json(source / "summary.json")
    silhouette = summary["pca"]["silhouette_score"]
    figure, axes = plt.subplots(figsize=(7.0, 5.8))
    colors = np.where(table["cluster"].to_numpy() == 0, "#7B2CBF", "#FF3B30")
    axes.scatter(table["pca_1"], table["pca_2"], c=colors, s=8, alpha=0.72)
    axes.set(
        xlabel="PCA dimension 1",
        ylabel="PCA dimension 2",
        title=f"Agglomerative clustering in PCA space\nSilhouette score = {silhouette:.4f}",
    )
    axes.grid(alpha=0.2)
    save_figure(figure, output)


def cnn_roc(source: Path, output: Path) -> None:
    curve = pd.read_csv(source / "aggregate_roc.csv")
    summary = read_json(source / "summary.json")
    auc_value = summary["aggregate_metrics"]["roc_auc"]
    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    axes.plot(
        curve["false_positive_rate"],
        curve["true_positive_rate"],
        color="#E67E22",
        linewidth=2.2,
        label=f"Aggregated ROC (AUC = {auc_value:.2f})",
    )
    axes.plot([0, 1], [0, 1], linestyle="--", color="#2E86C1", label="Chance")
    axes.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="Aggregated recording-fold ROC curve",
    )
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right")
    save_figure(figure, output)


def make_paper_figures(config: dict[str, Any]) -> dict[str, Any]:
    output = project_path(config, config["project"]["paper_generated_root"]) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    protocol_path = output / "figure_1_protocol.png"
    protocol_figure(config, protocol_path)
    generated.append(str(protocol_path))
    archive = project_path(config, config["project"]["epoch_root"]) / "18_1.npz"
    if archive.exists():
        example_path = output / "figure_2_selected_topomaps.png"
        difference_path = output / "figure_4_difference_topomaps.png"
        example_topomaps(config, example_path)
        difference_topomaps(config, difference_path)
        generated.extend([str(example_path), str(difference_path)])
    random_forest = stage_root(config, "random_forest")
    if (random_forest / "summary.json").exists():
        paths = [
            output / "figure_5_random_forest_roc.png",
            output / "figure_6_permutation_test.png",
            output / "figure_7_pca_clustering.png",
        ]
        random_forest_roc(random_forest, paths[0])
        permutation_histogram(random_forest, paths[1])
        clustering_figure(random_forest, paths[2])
        generated.extend(str(path) for path in paths)
    cnn = stage_root(config, "cnn", config["evaluation"]["paper_grouping"])
    if (cnn / "summary.json").exists():
        path = output / "figure_8_aggregated_cnn_roc.png"
        cnn_roc(cnn, path)
        generated.append(str(path))
    summary = {"figures": generated, "count": len(generated)}
    write_json(output.parent / "figure_manifest.json", summary)
    return summary
