import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from vip_eeg.constants import DEFAULT_CONFIG, PROJECT_ROOT, TASK_ORDER


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    recordings = config["dataset"]["recordings"]
    keys = [item["subject_session"] for item in recordings]
    expected = [
        "3_3",
        "8_3",
        "10_1",
        "11_1",
        "12_1",
        "12_2",
        "13_1",
        "14_1",
        "14_2",
        "15_1",
        "15_2",
        "16_1",
        "17_1",
        "18_1",
        "19_1",
    ]
    if keys != expected:
        raise ValueError(f"The paper cohort must be ordered as {expected}")
    if len(set(keys)) != len(keys):
        raise ValueError("Every subject-session key must be unique")
    labels = [config["tasks"][task]["label"] for task in TASK_ORDER]
    if labels != [0, 1]:
        raise ValueError("The paper text defines Perception=0 and Imagination=1")
    for task in TASK_ORDER:
        task_config = config["tasks"][task]
        if task_config["frames"] != 200:
            raise ValueError(f"{task} must produce exactly 200 topomaps")
        covered = task_config["frames"] * task_config["interval_seconds"]
        if abs(covered - task_config["duration_seconds"]) > 1e-9:
            raise ValueError(f"{task} duration and frame interval do not agree")
    filters = config["cnn"]["filters"]
    if filters != [32, 64, 128, 256, 512]:
        raise ValueError("The paper architecture requires filters 32, 64, 128, 256, 512")
    if config["topomaps"]["image_size"] != 150:
        raise ValueError("The paper architecture requires 150x150 topomaps")


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_root", PROJECT_ROOT)).resolve()


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def config_hash(config: dict[str, Any]) -> str:
    frozen = deepcopy(config)
    frozen.pop("_config_path", None)
    frozen.pop("_project_root", None)
    payload = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_root(config: dict[str, Any]) -> Path:
    base = project_path(config, config["project"]["results_root"])
    return base / "runs" / config_hash(config)


def stage_root(config: dict[str, Any], stage: str, grouping: str | None = None) -> Path:
    name = stage if grouping is None else f"{stage}_{grouping}"
    path = run_root(config) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def selected_recordings(
    config: dict[str, Any], selectors: list[str] | tuple[str, ...] | None = None
) -> list[dict[str, Any]]:
    recordings = config["dataset"]["recordings"]
    if not selectors:
        return recordings
    requested = set(selectors)
    available = {item["subject_session"] for item in recordings}
    missing = requested - available
    if missing:
        raise ValueError(f"Unknown recording selectors: {sorted(missing)}")
    return [item for item in recordings if item["subject_session"] in requested]
