from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vip_eeg.data.manifest import TopomapDataset, validate_manifest
from vip_eeg.data.topomaps import valid_topomap


def test_topomap_dataset_returns_normalized_chw_tensor(tmp_path: Path):
    image_path = tmp_path / "topomap.png"
    values = np.full((150, 150, 3), 128, dtype=np.uint8)
    Image.fromarray(values, mode="RGB").save(image_path)
    frame = pd.DataFrame([{"path": "topomap.png", "label": 1}])
    dataset = TopomapDataset(frame, tmp_path)
    image, label, index = dataset[0]
    assert image.shape == (3, 150, 150)
    assert abs(float(image.mean()) - 128 / 255) < 1e-6
    assert float(label) == 1.0
    assert index == 0


def test_manifest_validation_rejects_contaminated_rows(tmp_path: Path):
    config = {
        "_project_root": str(tmp_path),
        "dataset": {"recordings": [{"subject": 1, "session": 1, "subject_session": "1_1"}]},
        "tasks": {
            "perception": {"frames": 2, "interval_seconds": 0.1, "label": 0},
            "imagination": {"frames": 2, "interval_seconds": 0.2, "label": 1},
        },
        "topomaps": {"image_size": 4},
    }
    rows = []
    for task, interval, label in (("perception", 0.1, 0), ("imagination", 0.2, 1)):
        for frame_index in range(2):
            path = Path("images") / f"{task}_{frame_index}.png"
            destination = tmp_path / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(destination)
            rows.append(
                {
                    "path": path.as_posix(),
                    "subject": 1,
                    "session": 1,
                    "subject_session": "1_1",
                    "task": task,
                    "label": label,
                    "frame_index": frame_index,
                    "time_seconds": frame_index * interval,
                }
            )
    frame = pd.DataFrame(rows)
    assert validate_manifest(config, frame)["rows"] == 4
    assert valid_topomap(tmp_path / frame.iloc[0]["path"], 4)
    contaminated = frame.copy()
    contaminated.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="labels do not match"):
        validate_manifest(config, contaminated)
