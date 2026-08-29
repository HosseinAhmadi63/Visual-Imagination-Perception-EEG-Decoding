import logging
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, permutation_test_score, train_test_split

from vip_eeg.config import project_path, stage_root
from vip_eeg.data.manifest import load_manifest, validate_manifest
from vip_eeg.evaluation.metrics import binary_metrics
from vip_eeg.utils import set_global_seed, write_json


def load_flattened_images(config: dict[str, Any], manifest: pd.DataFrame) -> np.ndarray:
    root = project_path(config, ".")
    image_size = int(config["topomaps"]["image_size"])
    features = image_size * image_size * 3
    values = np.empty((len(manifest), features), dtype=np.uint8)
    for index, relative in enumerate(manifest["path"]):
        with Image.open(root / relative) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            values[index] = np.asarray(image, dtype=np.uint8).reshape(-1)
        if (index + 1) % 500 == 0 or index + 1 == len(manifest):
            logging.info("Loaded %d/%d topomaps", index + 1, len(manifest))
    return values


def build_random_forest(settings: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(settings["n_estimators"]),
        criterion=settings["criterion"],
        max_depth=settings["max_depth"],
        max_features=settings["max_features"],
        min_samples_split=int(settings["min_samples_split"]),
        bootstrap=bool(settings["bootstrap"]),
        n_jobs=int(settings["n_jobs"]),
        random_state=seed,
    )


def run_random_forest(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output = stage_root(config, "random_forest")
    summary_path = output / "summary.json"
    if summary_path.exists() and not force:
        from vip_eeg.utils import read_json

        logging.info("Reusing completed random-forest analysis from %s", output)
        return read_json(summary_path)
    if force:
        summary_path.unlink(missing_ok=True)
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    settings = config["random_forest"]
    manifest = load_manifest(config)
    validate_manifest(config, manifest)
    labels = manifest["label"].to_numpy(dtype=np.int64)
    features = load_flattened_images(config, manifest)
    indices = np.arange(len(manifest), dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=float(settings["test_fraction"]),
        random_state=seed,
        shuffle=True,
        stratify=labels if settings["stratified_split"] else None,
    )
    classifier = build_random_forest(settings, seed)
    logging.info("Fitting random forest on %d images", len(train_indices))
    classifier.fit(features[train_indices], labels[train_indices])
    probabilities = classifier.predict_proba(features[test_indices])[:, 1]
    predictions = classifier.predict(features[test_indices]).astype(np.int64)
    metrics = binary_metrics(labels[test_indices], probabilities, predictions=predictions)
    report = pd.DataFrame(
        classification_report(
            labels[test_indices],
            predictions,
            labels=[0, 1],
            target_names=["Perception", "Imagination"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(output / "classification_report.csv", index_label="class")
    matrix = confusion_matrix(labels[test_indices], predictions, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["true_perception", "true_imagination"],
        columns=["predicted_perception", "predicted_imagination"],
    ).to_csv(output / "confusion_matrix.csv", index_label="class")
    false_positive, true_positive, thresholds = roc_curve(labels[test_indices], probabilities)
    pd.DataFrame(
        {
            "false_positive_rate": false_positive,
            "true_positive_rate": true_positive,
            "threshold": thresholds,
        }
    ).to_csv(output / "roc_curve.csv", index=False)
    prediction_table = manifest.iloc[test_indices].copy()
    prediction_table.insert(0, "manifest_index", test_indices)
    prediction_table["probability_imagination"] = probabilities
    prediction_table["prediction"] = predictions
    prediction_table.to_csv(output / "test_predictions.csv", index=False)
    cross_validation = StratifiedKFold(
        n_splits=int(settings["permutation_folds"]), shuffle=True, random_state=seed
    )
    permutation_classifier = build_random_forest(settings, seed)
    logging.info("Running %d label permutations", int(settings["permutation_runs"]))
    observed_cv, permutation_scores, permutation_p_value = permutation_test_score(
        permutation_classifier,
        features,
        labels,
        scoring="accuracy",
        cv=cross_validation,
        n_permutations=int(settings["permutation_runs"]),
        n_jobs=int(settings["n_jobs"]),
        random_state=seed,
    )
    pd.DataFrame(
        {"permutation": np.arange(1, len(permutation_scores) + 1), "accuracy": permutation_scores}
    ).to_csv(output / "permutation_scores.csv", index=False)
    logging.info("Computing two-dimensional PCA")
    pca_features = features.astype(np.float32)
    pca = PCA(
        n_components=int(settings["pca_components"]),
        svd_solver=settings["pca_solver"],
        random_state=seed,
    )
    coordinates = pca.fit_transform(pca_features)
    clustering = AgglomerativeClustering(
        n_clusters=int(settings["clusters"]), linkage=settings["clustering_linkage"]
    )
    cluster_labels = clustering.fit_predict(coordinates)
    silhouette = float(silhouette_score(coordinates, cluster_labels))
    cluster_table = manifest[
        ["path", "subject", "session", "subject_session", "task", "label", "frame_index"]
    ].copy()
    cluster_table["pca_1"] = coordinates[:, 0]
    cluster_table["pca_2"] = coordinates[:, 1]
    cluster_table["cluster"] = cluster_labels
    cluster_table.to_csv(output / "pca_clusters.csv", index=False)
    summary = {
        "holdout": metrics,
        "training_images": int(len(train_indices)),
        "test_images": int(len(test_indices)),
        "feature_shape": [int(value) for value in features.shape],
        "permutation": {
            "observed_cross_validated_accuracy": float(observed_cv),
            "p_value": float(permutation_p_value),
            "runs": int(settings["permutation_runs"]),
            "folds": int(settings["permutation_folds"]),
        },
        "pca": {
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "silhouette_score": silhouette,
            "clusters": int(settings["clusters"]),
        },
    }
    write_json(summary_path, summary)
    return summary
