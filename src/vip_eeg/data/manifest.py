from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from vip_eeg.config import project_path


def manifest_path(config: dict[str, Any]) -> Path:
    return project_path(config, config["project"]["topomap_root"]) / "manifest.csv"


def load_manifest(config: dict[str, Any]) -> pd.DataFrame:
    path = manifest_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Topomap manifest not found: {path}")
    frame = pd.read_csv(path, dtype={"subject_session": str, "task": str})
    expected_columns = {
        "path",
        "subject",
        "session",
        "subject_session",
        "task",
        "label",
        "frame_index",
        "time_seconds",
    }
    missing = expected_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Topomap manifest is missing columns: {sorted(missing)}")
    return frame


def validate_manifest(config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    root = project_path(config, ".")
    recordings = config["dataset"]["recordings"]
    expected_recordings = [item["subject_session"] for item in recordings]
    expected_by_recording = {
        item["subject_session"]: (int(item["subject"]), int(item["session"])) for item in recordings
    }
    expected_tasks = ["perception", "imagination"]
    expected_total = sum(
        int(config["tasks"][task]["frames"]) for task in expected_tasks for _ in expected_recordings
    )
    failures = []
    if len(frame) != expected_total:
        failures.append(f"Expected {expected_total} manifest rows, found {len(frame)}")
    actual_recordings = set(frame["subject_session"].astype(str))
    unknown_recordings = actual_recordings - set(expected_recordings)
    if unknown_recordings:
        failures.append(f"Unknown recordings: {sorted(unknown_recordings)}")
    missing_recordings = set(expected_recordings) - actual_recordings
    if missing_recordings:
        failures.append(f"Missing recordings: {sorted(missing_recordings)}")
    actual_tasks = set(frame["task"].astype(str))
    if actual_tasks != set(expected_tasks):
        failures.append(f"Tasks must be {expected_tasks}, found {sorted(actual_tasks)}")
    duplicate_columns = ["subject_session", "task", "frame_index"]
    duplicates = frame.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, duplicate_columns].head(10).to_dict(orient="records")
        failures.append(f"Duplicate recording/task/frame rows: {examples}")
    if frame["path"].duplicated().any():
        failures.append("Every manifest path must be unique")
    counts = frame.groupby(["subject_session", "task"]).size()
    for recording in expected_recordings:
        expected_subject, expected_session = expected_by_recording[recording]
        for task in expected_tasks:
            task_config = config["tasks"][task]
            expected_frames = int(task_config["frames"])
            actual = int(counts.get((recording, task), 0))
            if actual != expected_frames:
                failures.append(f"{recording}/{task}: expected {expected_frames}, found {actual}")
                continue
            subset = frame[(frame["subject_session"] == recording) & (frame["task"] == task)]
            frame_indices = np.sort(subset["frame_index"].to_numpy(dtype=np.int64))
            if not np.array_equal(frame_indices, np.arange(expected_frames, dtype=np.int64)):
                failures.append(
                    f"{recording}/{task}: frame indices must be 0-{expected_frames - 1}"
                )
            ordered = subset.sort_values("frame_index")
            expected_times = np.arange(expected_frames, dtype=np.float64)
            expected_times *= float(task_config["interval_seconds"])
            actual_times = ordered["time_seconds"].to_numpy(dtype=np.float64)
            if not np.allclose(actual_times, expected_times, atol=1e-12, rtol=0.0):
                failures.append(f"{recording}/{task}: time grid does not match the paper interval")
            if not (ordered["label"].to_numpy(dtype=np.int64) == int(task_config["label"])).all():
                failures.append(f"{recording}/{task}: labels do not match the task definition")
            if not (ordered["subject"].to_numpy(dtype=np.int64) == expected_subject).all():
                failures.append(f"{recording}/{task}: subject values are inconsistent")
            if not (ordered["session"].to_numpy(dtype=np.int64) == expected_session).all():
                failures.append(f"{recording}/{task}: session values are inconsistent")
    missing_files = []
    invalid_images = []
    image_size = int(config["topomaps"]["image_size"])
    for value in frame["path"]:
        path = root / value
        if not path.exists():
            missing_files.append(value)
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (image_size, image_size) or image.mode != "RGB":
                    invalid_images.append(f"{value}: mode={image.mode}, size={image.size}")
        except OSError as error:
            invalid_images.append(f"{value}: {error}")
    if failures or missing_files or invalid_images:
        details = failures
        details += [f"Missing image: {value}" for value in missing_files[:20]]
        details += [f"Invalid image: {value}" for value in invalid_images[:20]]
        raise ValueError("\n".join(details))
    return {
        "rows": int(len(frame)),
        "recordings": int(frame["subject_session"].nunique()),
        "tasks": expected_tasks,
        "labels": sorted(int(value) for value in frame["label"].unique()),
    }


class TopomapDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path):
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(self.root / row["path"]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        tensor = torch.from_numpy(array.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label, int(index)
