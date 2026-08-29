import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vip_eeg.config import config_hash
from vip_eeg.evaluation import cnn_loso


def test_selected_fold_is_isolated_and_uses_canonical_seed(tmp_path: Path, monkeypatch):
    config = {
        "_project_root": str(tmp_path),
        "project": {"random_seed": 42, "results_root": "results"},
        "dataset": {
            "recordings": [
                {"subject": 1, "session": 1, "subject_session": "1_1"},
                {"subject": 2, "session": 1, "subject_session": "2_1"},
            ]
        },
        "cnn": {
            "device": "cpu",
            "validation_fraction": 0.2,
            "decision_threshold": 0.5,
        },
        "evaluation": {"paper_grouping": "subject"},
    }
    rows = []
    for subject in (1, 2):
        for index in range(10):
            label = index % 2
            rows.append(
                {
                    "path": f"subject_{subject}_{index}.png",
                    "subject": str(subject),
                    "session": 1,
                    "subject_session": f"{subject}_1",
                    "task": "imagination" if label else "perception",
                    "label": label,
                    "frame_index": index,
                    "time_seconds": float(index),
                }
            )
    manifest = pd.DataFrame(rows)

    def fake_train_fold(
        config_value,
        train_frame,
        validation_frame,
        test_frame,
        output,
        seed,
        device,
    ):
        labels = test_frame["label"].to_numpy(dtype=np.int64)
        metrics = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "samples": len(test_frame),
            "perception_samples": int(np.sum(labels == 0)),
            "imagination_samples": int(np.sum(labels == 1)),
            "test_loss": 0.0,
            "best_epoch": 1,
            "best_validation_loss": 0.0,
            "epochs_completed": 1,
            "training_images": len(train_frame),
            "validation_images": len(validation_frame),
            "test_images": len(test_frame),
            "seed": seed,
            "device": str(device),
        }
        predictions = test_frame.copy().reset_index(drop=True)
        predictions["probability_imagination"] = labels.astype(np.float64)
        predictions["prediction"] = labels
        predictions["correct"] = True
        return metrics, predictions

    monkeypatch.setattr(cnn_loso, "load_manifest", lambda _: manifest.copy())
    monkeypatch.setattr(cnn_loso, "validate_manifest", lambda *_: {})
    monkeypatch.setattr(cnn_loso, "train_fold", fake_train_fold)
    monkeypatch.setattr(cnn_loso, "resolve_device", lambda _: torch.device("cpu"))
    summary = cnn_loso.run_cnn_loso(config, grouping="subject", folds=["2"])
    run = tmp_path / "results" / "runs" / config_hash(config) / "cnn_subject"
    selected = run / "selected_folds" / "2"
    assert summary["scope"] == "selected_folds"
    assert summary["folds"] == 1
    assert not (run / "summary.json").exists()
    assert (selected / "summary.json").exists()
    with (selected / "folds" / "subject_2" / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    assert metrics["seed"] == 43
