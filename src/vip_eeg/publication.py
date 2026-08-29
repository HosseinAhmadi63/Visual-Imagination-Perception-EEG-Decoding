import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from vip_eeg.config import project_path, stage_root
from vip_eeg.plotting import make_paper_figures
from vip_eeg.utils import read_json, write_json


def source_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = project_path(config, config["project"]["paper_source_root"])
    return (
        root / "table_ii_state_of_the_art.csv",
        root / "table_i_random_forest.csv",
        root / "paper_targets.yaml",
    )


def verify_paper_sources(config: dict[str, Any]) -> dict[str, Any]:
    table_ii_path, table_i_path, targets_path = source_paths(config)
    table_ii = pd.read_csv(table_ii_path)
    table_i = pd.read_csv(table_i_path)
    with targets_path.open("r", encoding="utf-8") as stream:
        targets = yaml.safe_load(stream)
    selection_path = project_path(config, config["dataset"]["trial_selection_reference"])
    selection = pd.read_csv(selection_path, dtype={"subject_session": str})
    subject_columns = [item["subject_session"] for item in config["dataset"]["recordings"]]
    missing = set(subject_columns + ["method", "reported_average_percent"]) - set(table_ii.columns)
    if missing:
        raise ValueError(f"Table II source is missing columns: {sorted(missing)}")
    calculated = table_ii[subject_columns].mean(axis=1)
    comparison = table_ii[["method", "reported_average_percent"]].copy()
    comparison["calculated_average_percent"] = calculated
    comparison["difference_percent"] = (
        comparison["calculated_average_percent"] - comparison["reported_average_percent"]
    )
    study = comparison.loc[comparison["method"] == "This Study"].iloc[0]
    if round(float(study["calculated_average_percent"]), 1) != 95.1:
        raise ValueError("The transcribed study accuracies do not reproduce the paper's 95.1%")
    expected_keys = [item["subject_session"] for item in config["dataset"]["recordings"]]
    if selection["subject_session"].tolist() != expected_keys:
        raise ValueError("Deterministic trial-selection records do not match the paper cohort")
    if len(table_i) != 4 or int(targets["dataset"]["topomap_images"]) != 6000:
        raise ValueError("Paper source records are incomplete")
    return {
        "table_ii_rows": int(len(table_ii)),
        "table_i_rows": int(len(table_i)),
        "trial_selection_rows": int(len(selection)),
        "study_calculated_average_percent": float(study["calculated_average_percent"]),
        "study_reported_average_percent": float(study["reported_average_percent"]),
        "lmda_net_calculated_average_percent": float(
            comparison.loc[comparison["method"] == "LMDA-Net", "calculated_average_percent"].iloc[0]
        ),
        "lmda_net_reported_average_percent": float(
            comparison.loc[comparison["method"] == "LMDA-Net", "reported_average_percent"].iloc[0]
        ),
        "targets": targets,
        "averages": comparison.to_dict(orient="records"),
    }


def reproduce_paper_analysis(config: dict[str, Any], figures: bool = True) -> dict[str, Any]:
    generated = project_path(config, config["project"]["paper_generated_root"])
    generated.mkdir(parents=True, exist_ok=True)
    source_summary = verify_paper_sources(config)
    pd.DataFrame(source_summary["averages"]).to_csv(
        generated / "generated_table_ii_computed_averages.csv", index=False
    )
    result_records = {}
    random_forest = stage_root(config, "random_forest")
    if (random_forest / "summary.json").exists():
        result_records["random_forest"] = read_json(random_forest / "summary.json")
        for filename in [
            "classification_report.csv",
            "confusion_matrix.csv",
            "roc_curve.csv",
            "permutation_scores.csv",
            "pca_clusters.csv",
        ]:
            source = random_forest / filename
            if source.exists():
                shutil.copy2(source, generated / f"generated_{filename}")
    cnn = stage_root(config, "cnn", config["evaluation"]["paper_grouping"])
    if (cnn / "summary.json").exists():
        result_records["cnn"] = read_json(cnn / "summary.json")
        for filename in [
            "fold_metrics.csv",
            "paper_accuracy_comparison.csv",
            "aggregate_roc.csv",
        ]:
            source = cnn / filename
            if source.exists():
                shutil.copy2(source, generated / f"generated_{filename}")
    statistics = stage_root(config, "statistics")
    if (statistics / "summary.json").exists():
        result_records["statistics"] = read_json(statistics / "summary.json")
        source = statistics / "channel_cluster_statistics.csv"
        if source.exists():
            shutil.copy2(source, generated / "generated_channel_cluster_statistics.csv")
    figure_summary = make_paper_figures(config) if figures else {"figures": [], "count": 0}
    summary = {
        "paper_sources": source_summary,
        "completed_result_stages": sorted(result_records),
        "results": result_records,
        "figures": figure_summary,
    }
    write_json(generated / "paper_consistency.json", summary)
    return summary
