from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg", force=True)

from matplotlib.figure import Figure
from mne.viz import plot_topomap


def valid_topomap(path: str | Path, image_size: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.size == (image_size, image_size)
    except (OSError, ValueError):
        return False


def render_topomap(
    values: np.ndarray,
    info: Any,
    output_path: str | Path,
    settings: dict[str, Any],
    value_limit: float | None = None,
) -> None:
    image_size = int(settings["image_size"])
    limit = float(np.max(np.abs(values))) if value_limit is None else float(value_limit)
    limit = max(limit, np.finfo(float).eps)
    figure = Figure(figsize=(1.5, 1.5), dpi=100, facecolor="white")
    axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    plot_topomap(
        values,
        info,
        axes=axes,
        show=False,
        cmap=settings["cmap"],
        sensors=bool(settings["sensors"]),
        contours=int(settings["contours"]),
        res=int(settings["interpolation_resolution"]),
        image_interp=settings["image_interpolation"],
        extrapolate=settings["extrapolate"],
        outlines=settings["outlines"],
        border=settings["border"],
        vlim=(-limit, limit),
    )
    axes.set_axis_off()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    resampling = Image.Resampling.LANCZOS
    image = image.resize((image_size, image_size), resample=resampling)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".png") as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format="PNG", optimize=True)
        temporary_path.replace(destination)
    finally:
        image.close()
        buffer.close()
        figure.clear()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def make_info(channel_names: np.ndarray, positions: np.ndarray, sampling_rate: float) -> Any:
    import mne

    names = [str(name) for name in channel_names.tolist()]
    channel_positions = {name: positions[index] for index, name in enumerate(names)}
    montage = mne.channels.make_dig_montage(ch_pos=channel_positions, coord_frame="head")
    info = mne.create_info(names, sfreq=float(sampling_rate), ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    return info
