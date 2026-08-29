import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vip_eeg.config import project_path, stage_root
from vip_eeg.data.manifest import TopomapDataset, load_manifest, validate_manifest
from vip_eeg.evaluation.metrics import binary_metrics
from vip_eeg.models import build_model
from vip_eeg.utils import read_json, set_global_seed, write_csv, write_json


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    config: dict[str, Any],
    frame: pd.DataFrame,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    settings = config["cnn"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TopomapDataset(frame, project_path(config, "."))
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(settings["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def training_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch
        total_samples += batch
    return total_loss / total_samples


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    probabilities_all = []
    with torch.inference_mode():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            probabilities = torch.sigmoid(logits)
            batch = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch
            total_samples += batch
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())
    return (
        total_loss / total_samples,
        np.concatenate(labels_all).astype(np.int64),
        np.concatenate(probabilities_all).astype(np.float64),
    )


def train_fold(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output: Path,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    settings = config["cnn"]
    set_global_seed(seed)
    train_loader = make_loader(config, train_frame, True, seed, device)
    validation_loader = make_loader(config, validation_frame, False, seed, device)
    test_loader = make_loader(config, test_frame, False, seed, device)
    model = build_model(config).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["lr_plateau_factor"]),
        patience=int(settings["lr_plateau_patience"]),
        min_lr=float(settings["minimum_learning_rate"]),
    )
    checkpoint = output / "best_model.pt"
    history = []
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        train_loss = training_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, validation_labels, validation_probabilities = evaluate_epoch(
            model, validation_loader, loss_function, device
        )
        validation_metrics = binary_metrics(
            validation_labels,
            validation_probabilities,
            float(settings["decision_threshold"]),
        )
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "learning_rate": current_learning_rate,
            }
        )
        improvement = best_validation - validation_loss
        if improvement > float(settings["early_stopping_min_delta"]):
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "seed": seed,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
        scheduler.step(validation_loss)
        logging.info(
            "Epoch %d/%d train_loss=%.6f validation_loss=%.6f validation_accuracy=%.4f",
            epoch,
            int(settings["max_epochs"]),
            train_loss,
            validation_loss,
            validation_metrics["accuracy"],
        )
        if stale_epochs >= int(settings["early_stopping_patience"]):
            break
    write_csv(pd.DataFrame(history), output / "history.csv", index=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    test_loss, test_labels, test_probabilities = evaluate_epoch(
        model, test_loader, loss_function, device
    )
    threshold = float(settings["decision_threshold"])
    predictions = (test_probabilities >= threshold).astype(np.int64)
    metrics = binary_metrics(test_labels, test_probabilities, threshold)
    metrics.update(
        {
            "test_loss": test_loss,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation),
            "epochs_completed": int(len(history)),
            "training_images": int(len(train_frame)),
            "validation_images": int(len(validation_frame)),
            "test_images": int(len(test_frame)),
            "seed": int(seed),
            "device": str(device),
        }
    )
    predictions_frame = test_frame.copy().reset_index(drop=True)
    predictions_frame["probability_imagination"] = test_probabilities
    predictions_frame["prediction"] = predictions
    predictions_frame["correct"] = predictions == test_labels
    return metrics, predictions_frame


def grouping_values(config: dict[str, Any], grouping: str) -> list[str]:
    recordings = config["dataset"]["recordings"]
    if grouping == "subject_session":
        return [str(item["subject_session"]) for item in recordings]
    if grouping == "subject":
        return list(dict.fromkeys(str(item["subject"]) for item in recordings))
    raise ValueError("Grouping must be subject_session or subject")


def paper_reference(config: dict[str, Any]) -> dict[str, float]:
    path = project_path(
        config,
        Path(config["project"]["paper_source_root"]) / "table_ii_state_of_the_art.csv",
    )
    table = pd.read_csv(path)
    row = table.loc[table["method"] == "This Study"].iloc[0]
    return {
        item["subject_session"]: float(row[item["subject_session"]]) / 100.0
        for item in config["dataset"]["recordings"]
    }


def run_cnn_loso(
    config: dict[str, Any],
    grouping: str | None = None,
    folds: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    grouping = grouping or config["evaluation"]["paper_grouping"]
    output = stage_root(config, "cnn", grouping)
    summary_path = output / "summary.json"
    ordered_groups = grouping_values(config, grouping)
    if summary_path.exists() and not force and not folds:
        completed = read_json(summary_path)
        valid_summary = (
            completed.get("scope") == "complete_grouping"
            and completed.get("included_groups") == ordered_groups
            and int(completed.get("folds", -1)) == len(ordered_groups)
        )
        if valid_summary:
            logging.info("Reusing completed CNN evaluation from %s", output)
            return completed
        logging.warning("Ignoring an incomplete CNN summary in %s", output)
        summary_path.unlink(missing_ok=True)
    if force and not folds:
        summary_path.unlink(missing_ok=True)
    seed = int(config["project"]["random_seed"])
    device = resolve_device(config["cnn"]["device"])
    manifest = load_manifest(config).reset_index().rename(columns={"index": "manifest_index"})
    validate_manifest(config, manifest)
    group_column = grouping
    manifest["subject"] = manifest["subject"].astype(str)
    manifest["subject_session"] = manifest["subject_session"].astype(str)
    selected_groups = (
        ordered_groups if not folds else [value for value in ordered_groups if value in folds]
    )
    unknown = set(folds or []) - set(ordered_groups)
    if unknown:
        raise ValueError(f"Unknown folds for {grouping}: {sorted(unknown)}")
    if folds:
        selection_name = "__".join(selected_groups)
        aggregate_output = output / "selected_folds" / selection_name
        aggregate_output.mkdir(parents=True, exist_ok=True)
        if force:
            (aggregate_output / "summary.json").unlink(missing_ok=True)
    else:
        aggregate_output = output
    fold_metrics = []
    prediction_frames = []
    for held_out in selected_groups:
        fold_index = ordered_groups.index(held_out)
        fold_seed = seed + fold_index
        fold_name = f"{grouping}_{held_out}"
        fold_output = (
            aggregate_output / "folds" / fold_name if folds else output / "folds" / fold_name
        )
        fold_output.mkdir(parents=True, exist_ok=True)
        metrics_path = fold_output / "metrics.json"
        predictions_path = fold_output / "predictions.csv"
        test_frame = manifest[manifest[group_column] == held_out].copy()
        development = manifest[manifest[group_column] != held_out].copy()
        if test_frame.empty or development.empty:
            raise ValueError(f"Fold {fold_name} has an empty development or test set")
        reusable = metrics_path.exists() and predictions_path.exists() and not force
        if reusable:
            try:
                metrics = read_json(metrics_path)
                predictions = pd.read_csv(
                    predictions_path,
                    dtype={"subject_session": str, "subject": str, "fold": str},
                )
                required_prediction_columns = {
                    "fold",
                    "grouping",
                    "path",
                    "label",
                    "probability_imagination",
                    "prediction",
                }
                reusable = (
                    int(metrics.get("seed", -1)) == fold_seed
                    and metrics.get("grouping") == grouping
                    and str(metrics.get("fold")) == held_out
                    and int(metrics.get("test_images", -1)) == len(test_frame)
                    and required_prediction_columns.issubset(predictions.columns)
                    and len(predictions) == len(test_frame)
                    and predictions["fold"].astype(str).eq(held_out).all()
                    and predictions["grouping"].eq(grouping).all()
                    and predictions["path"].tolist() == test_frame["path"].tolist()
                )
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                reusable = False
        if not reusable:
            metrics_path.unlink(missing_ok=True)
            train_frame, validation_frame = train_test_split(
                development,
                test_size=float(config["cnn"]["validation_fraction"]),
                random_state=fold_seed,
                shuffle=True,
                stratify=development["label"],
            )
            logging.info(
                "Training %s on %d train, %d validation, %d test images using %s",
                fold_name,
                len(train_frame),
                len(validation_frame),
                len(test_frame),
                device,
            )
            metrics, predictions = train_fold(
                config,
                train_frame,
                validation_frame,
                test_frame,
                fold_output,
                fold_seed,
                device,
            )
            metrics["fold"] = held_out
            metrics["grouping"] = grouping
            metrics["test_subject_sessions"] = sorted(
                test_frame["subject_session"].unique().tolist()
            )
            predictions.insert(0, "fold", held_out)
            predictions.insert(1, "grouping", grouping)
            write_csv(predictions, predictions_path, index=False)
            write_json(metrics_path, metrics)
        else:
            logging.info("Reusing fold %s", fold_name)
        fold_metrics.append(metrics)
        prediction_frames.append(predictions)
    metrics_table = pd.DataFrame(fold_metrics)
    write_csv(metrics_table, aggregate_output / "fold_metrics.csv", index=False)
    aggregate_predictions = pd.concat(prediction_frames, ignore_index=True)
    write_csv(
        aggregate_predictions,
        aggregate_output / "aggregate_predictions.csv",
        index=False,
    )
    labels = aggregate_predictions["label"].to_numpy(dtype=np.int64)
    probabilities = aggregate_predictions["probability_imagination"].to_numpy(dtype=np.float64)
    aggregate_metrics = binary_metrics(
        labels, probabilities, float(config["cnn"]["decision_threshold"])
    )
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    roc_table = pd.DataFrame(
        {
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
            "threshold": thresholds,
        }
    )
    write_csv(roc_table, aggregate_output / "aggregate_roc.csv", index=False)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mean_fold = {name: float(metrics_table[name].mean()) for name in metric_names}
    standard_deviation = {
        name: float(metrics_table[name].std(ddof=1)) if len(metrics_table) > 1 else None
        for name in metric_names
    }
    if grouping == "subject_session":
        reference = paper_reference(config)
        comparison = metrics_table[["fold", "accuracy"]].copy()
        comparison["paper_accuracy"] = comparison["fold"].map(reference)
        comparison["difference"] = comparison["accuracy"] - comparison["paper_accuracy"]
        write_csv(comparison, aggregate_output / "paper_accuracy_comparison.csv", index=False)
    summary = {
        "grouping": grouping,
        "folds": int(len(metrics_table)),
        "scope": "selected_folds" if folds else "complete_grouping",
        "included_groups": selected_groups,
        "device": str(device),
        "mean_fold_metrics": mean_fold,
        "fold_metric_standard_deviation": standard_deviation,
        "aggregate_metrics": aggregate_metrics,
        "total_predictions": int(len(aggregate_predictions)),
    }
    destination = aggregate_output / "summary.json" if folds else summary_path
    write_json(destination, summary)
    return summary
