import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from vip_eeg.config import project_path, stage_root
from vip_eeg.data.topomaps import make_info
from vip_eeg.utils import write_json


def archive_path(config: dict[str, Any], subject_session: str) -> Path:
    root = project_path(config, config["project"]["epoch_root"])
    path = root / f"{subject_session}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Epoch archive not found: {path}")
    return path


def sample_at(epoch: np.ndarray, sampling_rate: float, time_seconds: float) -> np.ndarray:
    index = int(round(time_seconds * sampling_rate))
    if index < 0 or index >= epoch.shape[1]:
        raise IndexError(
            f"Time {time_seconds} s maps outside an epoch with {epoch.shape[1]} samples"
        )
    return epoch[:, index]


def cluster_mask(cluster: Any, channels: int) -> np.ndarray:
    if isinstance(cluster, tuple):
        if len(cluster) != 1:
            raise ValueError(f"Expected one-dimensional channel cluster, found {len(cluster)} axes")
        array = np.asarray(cluster[0])
    else:
        array = np.asarray(cluster)
    if array.dtype == bool and array.size == channels:
        return array.reshape(channels)
    mask = np.zeros(channels, dtype=bool)
    indices = np.asarray(array, dtype=np.int64).reshape(-1)
    mask[indices] = True
    return mask


def run_cluster_statistics(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output = stage_root(config, "statistics")
    summary_path = output / "summary.json"
    if summary_path.exists() and not force:
        from vip_eeg.utils import read_json

        logging.info("Reusing completed cluster statistics from %s", output)
        return read_json(summary_path)
    if force:
        summary_path.unlink(missing_ok=True)
    settings = config["statistics"]
    recording_keys = [item["subject_session"] for item in config["dataset"]["recordings"]]
    differences = {"perception": [], "imagination": []}
    reference_names = None
    reference_positions = None
    reference_rate = None
    for key in recording_keys:
        with np.load(archive_path(config, key), allow_pickle=False) as archive:
            names = archive["channel_names"].astype(str)
            positions = archive["positions"].astype(np.float64)
            sampling_rate = float(archive["sampling_rate"])
            if reference_names is None:
                reference_names = names
                reference_positions = positions
                reference_rate = sampling_rate
            if not np.array_equal(names, reference_names):
                raise ValueError(f"Channel order differs in epoch archive {key}")
            for task in ("perception", "imagination"):
                first, second = [float(value) for value in settings["difference_times"][task]]
                epoch = archive[task]
                difference = sample_at(epoch, sampling_rate, second) - sample_at(
                    epoch, sampling_rate, first
                )
                differences[task].append(difference)
    info = make_info(reference_names, reference_positions, float(reference_rate))
    adjacency, adjacency_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    if adjacency.shape[0] != len(reference_names):
        raise ValueError("EEG adjacency does not match the archive channel count")
    if list(adjacency_names) != reference_names.tolist():
        raise ValueError("EEG adjacency channel order does not match the archives")
    summaries = {}
    channel_rows = []
    saved = {}
    for task in ("perception", "imagination"):
        matrix = np.stack(differences[task]).astype(np.float64)
        statistic, clusters, p_values, null_distribution = mne.stats.permutation_cluster_1samp_test(
            matrix,
            threshold=None,
            n_permutations=int(settings["cluster_permutations"]),
            tail=int(settings["cluster_tail"]),
            adjacency=adjacency,
            out_type="mask",
            seed=int(config["project"]["random_seed"]),
            verbose=False,
        )
        minimum_p = np.ones(len(reference_names), dtype=np.float64)
        cluster_records = []
        for cluster_index, (cluster, p_value) in enumerate(zip(clusters, p_values, strict=True)):
            mask = cluster_mask(cluster, len(reference_names))
            minimum_p[mask] = np.minimum(minimum_p[mask], float(p_value))
            cluster_records.append(
                {
                    "cluster": cluster_index,
                    "p_value": float(p_value),
                    "channels": reference_names[mask].tolist(),
                }
            )
        significant = minimum_p <= float(settings["cluster_alpha"])
        for index, channel in enumerate(reference_names):
            channel_rows.append(
                {
                    "task": task,
                    "channel": str(channel),
                    "t_statistic": float(statistic[index]),
                    "minimum_cluster_p": float(minimum_p[index]),
                    "significant": bool(significant[index]),
                }
            )
        summaries[task] = {
            "clusters": cluster_records,
            "significant_clusters": int(
                np.sum(np.asarray(p_values) <= float(settings["cluster_alpha"]))
            ),
            "significant_channels": reference_names[significant].tolist(),
            "null_distribution_size": int(len(null_distribution)),
        }
        saved[f"{task}_differences"] = matrix.astype(np.float32)
        saved[f"{task}_mean"] = matrix.mean(axis=0).astype(np.float32)
        saved[f"{task}_statistic"] = np.asarray(statistic, dtype=np.float32)
        saved[f"{task}_minimum_p"] = minimum_p.astype(np.float32)
    pd.DataFrame(channel_rows).to_csv(output / "channel_cluster_statistics.csv", index=False)
    np.savez_compressed(
        output / "difference_statistics.npz",
        channel_names=reference_names,
        positions=reference_positions,
        sampling_rate=np.asarray(reference_rate),
        **saved,
    )
    summary = {
        "recordings": len(recording_keys),
        "permutations": int(settings["cluster_permutations"]),
        "alpha": float(settings["cluster_alpha"]),
        "tasks": summaries,
    }
    write_json(summary_path, summary)
    return summary
