import logging
from pathlib import Path
from typing import Any

import pandas as pd

from vip_eeg.config import project_path, selected_recordings, stage_root
from vip_eeg.utils import write_json


def expected_fif_name(subject: int, session: int) -> str:
    return f"sub{subject}_sess{session}_50_ica_eeg.fif"


def find_preprocessed_fif(data_root: Path, subject: int, session: int) -> Path:
    filename = expected_fif_name(subject, session)
    matches = sorted(data_root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename} under {data_root}, found {len(matches)}"
        )
    return matches[0]


def validate_download(config: dict[str, Any]) -> dict[str, Any]:
    data_root = project_path(config, config["project"]["data_root"])
    object_manifest_path = project_path(config, config["dataset"]["object_manifest"])
    object_manifest = pd.read_csv(object_manifest_path)
    object_failures = []
    object_records = []
    for row in object_manifest.itertuples(index=False):
        path = data_root / row.path
        if not path.exists():
            object_failures.append(f"Missing OpenNeuro object: {row.path}")
            continue
        size = path.stat().st_size
        if size != int(row.size_bytes):
            object_failures.append(
                f"OpenNeuro object size mismatch for {row.path}: expected {row.size_bytes}, found {size}"
            )
        object_records.append(
            {
                "path": str(path),
                "size_bytes": int(size),
                "sha256": row.sha256,
                "version_id": row.version_id,
            }
        )
    if object_failures:
        raise FileNotFoundError("\n".join(object_failures))
    files = []
    missing = []
    for recording in selected_recordings(config):
        try:
            path = find_preprocessed_fif(
                data_root, int(recording["subject"]), int(recording["session"])
            )
            files.append(
                {
                    "subject_session": recording["subject_session"],
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        except FileNotFoundError as error:
            missing.append(str(error))
    if missing:
        raise FileNotFoundError("\n".join(missing))
    summary = {
        "accession": config["dataset"]["accession"],
        "snapshot": config["dataset"]["snapshot"],
        "data_root": str(data_root),
        "recordings": len(files),
        "objects": len(object_records),
        "total_bytes": int(sum(item["size_bytes"] for item in object_records)),
        "object_manifest": str(object_manifest_path),
        "base_fif_files": files,
    }
    return summary


def download_dataset(config: dict[str, Any]) -> dict[str, Any]:
    from openneuro import download as openneuro_download

    dataset = config["dataset"]
    data_root = project_path(config, config["project"]["data_root"])
    data_root.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Downloading OpenNeuro %s snapshot %s into %s",
        dataset["accession"],
        dataset["snapshot"],
        data_root,
    )
    openneuro_download(
        dataset=dataset["accession"],
        tag=dataset["snapshot"],
        target_dir=data_root,
        include=[dataset["include"]],
        verify_hash=True,
        verify_size=True,
        max_retries=5,
        max_concurrent_downloads=5,
        metadata_timeout=60.0,
    )
    summary = validate_download(config)
    write_json(stage_root(config, "download") / "summary.json", summary)
    logging.info("Validated %d preprocessed recording files", summary["recordings"])
    return summary
