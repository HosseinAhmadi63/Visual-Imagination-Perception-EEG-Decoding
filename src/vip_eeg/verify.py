import importlib.metadata
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from vip_eeg.config import config_hash, stage_root
from vip_eeg.models import build_model
from vip_eeg.publication import verify_paper_sources
from vip_eeg.utils import set_global_seed, write_json


def dependency_versions() -> dict[str, str]:
    distributions = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "mne",
        "matplotlib",
        "Pillow",
        "PyYAML",
        "joblib",
        "torch",
        "openneuro-py",
    ]
    values = {}
    for distribution in distributions:
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = "not-installed"
    return values


def verify_installation(config: dict[str, Any], output: str | Path | None = None) -> dict[str, Any]:
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)
    model = build_model(config)
    model.eval()
    image_size = int(config["topomaps"]["image_size"])
    inputs = torch.randn(2, 3, image_size, image_size)
    with torch.inference_mode():
        intermediate = model.intermediate_shapes(inputs)
        logits = model(inputs)
    expected = [
        (2, 32, 75, 75),
        (2, 64, 37, 37),
        (2, 128, 18, 18),
        (2, 256, 9, 9),
        (2, 512, 4, 4),
    ]
    if intermediate != expected:
        raise ValueError(f"CNN shapes {intermediate} do not match Figure 3 shapes {expected}")
    if tuple(logits.shape) != (2,):
        raise ValueError(f"CNN output shape is {tuple(logits.shape)}, expected (2,)")
    model.train()
    optimizer = Adam(
        model.parameters(),
        lr=float(config["cnn"]["learning_rate"]),
        weight_decay=float(config["cnn"]["weight_decay"]),
    )
    labels = torch.tensor([0.0, 1.0])
    optimizer.zero_grad(set_to_none=True)
    loss = nn.BCEWithLogitsLoss()(model(inputs), labels)
    loss.backward()
    optimizer.step()
    sources = verify_paper_sources(config)
    summary = {
        "config_hash": config_hash(config),
        "cnn_intermediate_shapes": [list(value) for value in intermediate],
        "cnn_output_shape": list(logits.shape),
        "cnn_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "single_training_step_loss": float(loss.detach()),
        "paper_sources": sources,
        "dependencies": dependency_versions(),
        "status": "passed",
    }
    destination = (
        Path(output) if output is not None else stage_root(config, "verify") / "summary.json"
    )
    write_json(destination, summary)
    return summary
