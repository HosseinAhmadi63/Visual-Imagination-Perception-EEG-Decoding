import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from vip_eeg.config import project_path, selected_recordings, stage_root
from vip_eeg.data.events import raw_annotation_events, select_first_pictorial_pair
from vip_eeg.data.manifest import manifest_path, validate_manifest
from vip_eeg.data.openneuro import find_preprocessed_fif
from vip_eeg.data.topomaps import render_topomap, valid_topomap
from vip_eeg.utils import write_csv, write_json


def extract_epoch(
    raw: Any, picks: np.ndarray, start_sample: int, duration_seconds: float
) -> np.ndarray:
    sampling_rate = float(raw.info["sfreq"])
    samples = int(round(duration_seconds * sampling_rate)) + 1
    stop_sample = start_sample + samples
    if start_sample < 0 or stop_sample > raw.n_times:
        raise ValueError(
            f"Requested samples [{start_sample}, {stop_sample}) exceed recording length {raw.n_times}"
        )
    return raw.get_data(picks=picks, start=start_sample, stop=stop_sample).astype(np.float32)


def frame_offsets(task_config: dict[str, Any], sampling_rate: float) -> np.ndarray:
    times = np.arange(int(task_config["frames"]), dtype=np.float64)
    times *= float(task_config["interval_seconds"])
    offsets = np.rint(times * sampling_rate).astype(np.int64)
    if len(np.unique(offsets)) != len(offsets):
        raise ValueError("Frame times collapse onto duplicate EEG samples")
    return offsets


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def prepare_recording(
    config: dict[str, Any], recording: dict[str, Any], force: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_path(config, ".")
    data_root = project_path(config, config["project"]["data_root"])
    epoch_root = project_path(config, config["project"]["epoch_root"])
    topomap_root = project_path(config, config["project"]["topomap_root"])
    subject = int(recording["subject"])
    session = int(recording["session"])
    key = recording["subject_session"]
    source = find_preprocessed_fif(data_root, subject, session)
    logging.info("Reading %s from %s", key, source)
    raw = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    expected_channels = int(config["dataset"]["expected_eeg_channels"])
    if len(picks) != expected_channels:
        raise ValueError(f"{key} has {len(picks)} EEG channels; expected {expected_channels}")
    info = mne.pick_info(raw.info, picks, copy=True)
    positions = np.asarray([channel["loc"][:3] for channel in info["chs"]], dtype=np.float64)
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError(f"{key} contains missing or invalid channel positions")
    events = raw_annotation_events(raw)
    pair = select_first_pictorial_pair(
        events,
        perception_prefix=config["tasks"]["perception"]["event_prefix"],
        imagination_prefix=config["tasks"]["imagination"]["event_prefix"],
    )
    selection_path = project_path(config, config["dataset"]["trial_selection_reference"])
    selection = pd.read_csv(selection_path, dtype={"subject_session": str})
    expected = selection.loc[selection["subject_session"] == key]
    if len(expected) != 1:
        raise ValueError(f"Trial-selection reference must contain exactly one row for {key}")
    expected_row = expected.iloc[0]
    actual = {
        "stimulus": pair.stimulus,
        "perception_sample": pair.perception.sample,
        "imagination_sample": pair.imagination.sample,
    }
    for field, value in actual.items():
        if str(value) != str(expected_row[field]):
            raise ValueError(
                f"Deterministic trial selection for {key} produced {field}={value}, expected {expected_row[field]}"
            )
    event_by_task = {"perception": pair.perception, "imagination": pair.imagination}
    epoch_by_task = {}
    for task in ("perception", "imagination"):
        task_config = config["tasks"][task]
        epoch_by_task[task] = extract_epoch(
            raw,
            picks,
            event_by_task[task].sample,
            float(task_config["duration_seconds"]),
        )
    epoch_root.mkdir(parents=True, exist_ok=True)
    archive = epoch_root / f"{key}.npz"
    np.savez_compressed(
        archive,
        perception=epoch_by_task["perception"],
        imagination=epoch_by_task["imagination"],
        channel_names=np.asarray(info.ch_names, dtype=str),
        positions=positions,
        sampling_rate=np.asarray(float(raw.info["sfreq"])),
        perception_event=np.asarray(pair.perception.description),
        imagination_event=np.asarray(pair.imagination.description),
        perception_onset=np.asarray(pair.perception.onset_seconds),
        imagination_onset=np.asarray(pair.imagination.onset_seconds),
        stimulus=np.asarray(pair.stimulus),
        source_fif=np.asarray(relative_path(source, root)),
    )
    rows = []
    for task in ("perception", "imagination"):
        task_config = config["tasks"][task]
        offsets = frame_offsets(task_config, float(raw.info["sfreq"]))
        times = np.arange(int(task_config["frames"]), dtype=np.float64)
        times *= float(task_config["interval_seconds"])
        values = epoch_by_task[task][:, offsets]
        task_directory = topomap_root / key / task
        for frame_index, time_seconds in enumerate(times):
            filename = f"{task}_sub{subject:02d}_sess{session}_t{frame_index + 1:04d}.png"
            image_path = task_directory / filename
            if force or not valid_topomap(image_path, int(config["topomaps"]["image_size"])):
                render_topomap(
                    values[:, frame_index],
                    info,
                    image_path,
                    config["topomaps"],
                )
            rows.append(
                {
                    "path": relative_path(image_path, root),
                    "subject": subject,
                    "session": session,
                    "subject_session": key,
                    "task": task,
                    "label": int(task_config["label"]),
                    "frame_index": frame_index,
                    "time_seconds": float(time_seconds),
                    "source_fif": relative_path(source, root),
                    "source_event": event_by_task[task].description,
                    "source_onset_seconds": event_by_task[task].onset_seconds,
                    "stimulus": pair.stimulus,
                }
            )
    raw.close()
    summary = {
        "subject_session": key,
        "subject": subject,
        "session": session,
        "source_fif": relative_path(source, root),
        "sampling_rate": float(info["sfreq"]),
        "eeg_channels": len(info.ch_names),
        "stimulus": pair.stimulus,
        "perception_event": pair.perception.description,
        "imagination_event": pair.imagination.description,
        "topomaps": len(rows),
        "archive": relative_path(archive, root),
    }
    return rows, summary


def prepare_topomaps(
    config: dict[str, Any], selectors: list[str] | None = None, force: bool = False
) -> dict[str, Any]:
    recordings = selected_recordings(config, selectors)
    new_rows = []
    recording_summaries = []
    for recording in recordings:
        rows, summary = prepare_recording(config, recording, force)
        new_rows.extend(rows)
        recording_summaries.append(summary)
        logging.info("Prepared %s with %d topomaps", summary["subject_session"], len(rows))
    path = manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated_keys = {item["subject_session"] for item in recordings}
    if path.exists() and len(recordings) < len(config["dataset"]["recordings"]):
        existing = pd.read_csv(path, dtype={"subject_session": str})
        existing = existing[~existing["subject_session"].isin(updated_keys)]
        frame = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        frame = pd.DataFrame(new_rows)
    recording_order = [item["subject_session"] for item in config["dataset"]["recordings"]]
    frame["_recording_order"] = pd.Categorical(
        frame["subject_session"], categories=recording_order, ordered=True
    )
    frame["_task_order"] = pd.Categorical(
        frame["task"], categories=["perception", "imagination"], ordered=True
    )
    frame = frame.sort_values(["_recording_order", "_task_order", "frame_index"])
    frame = frame.drop(columns=["_recording_order", "_task_order"]).reset_index(drop=True)
    write_csv(frame, path, index=False)
    expected_keys = {item["subject_session"] for item in config["dataset"]["recordings"]}
    complete = set(frame["subject_session"].astype(str)) == expected_keys
    validation = validate_manifest(config, frame) if complete else {"rows": int(len(frame))}
    summary = {
        "manifest": str(path),
        "updated_recordings": [item["subject_session"] for item in recordings],
        "complete_paper_cohort": complete,
        "validation": validation,
        "recordings": recording_summaries,
    }
    write_json(stage_root(config, "prepare") / "summary.json", summary)
    return summary
