import torch

from vip_eeg.config import load_config
from vip_eeg.models import build_model


def test_figure_three_shapes_and_output():
    config = load_config("configs/paper.yaml")
    model = build_model(config).eval()
    inputs = torch.randn(2, 3, 150, 150)
    with torch.inference_mode():
        shapes = model.intermediate_shapes(inputs)
        outputs = model(inputs)
    assert shapes == [
        (2, 32, 75, 75),
        (2, 64, 37, 37),
        (2, 128, 18, 18),
        (2, 256, 9, 9),
        (2, 512, 4, 4),
    ]
    assert outputs.shape == (2,)
    assert model.feature_shape == (512, 4, 4)
