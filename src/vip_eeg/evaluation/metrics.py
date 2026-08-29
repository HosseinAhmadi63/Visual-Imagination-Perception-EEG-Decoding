from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if predictions is None:
        predictions = (probabilities >= threshold).astype(np.int64)
    else:
        predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != probabilities.shape or labels.shape != predictions.shape:
        raise ValueError("Labels, probabilities, and predictions must have identical shapes")
    values = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "samples": int(len(labels)),
        "perception_samples": int(np.sum(labels == 0)),
        "imagination_samples": int(np.sum(labels == 1)),
    }
    return values
