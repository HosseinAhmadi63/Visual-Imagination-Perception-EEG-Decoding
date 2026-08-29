from vip_eeg.config import load_config
from vip_eeg.publication import verify_paper_sources


def test_transcribed_paper_results_are_internally_checked():
    config = load_config("configs/paper.yaml")
    summary = verify_paper_sources(config)
    assert round(summary["study_calculated_average_percent"], 1) == 95.1
    assert round(summary["lmda_net_calculated_average_percent"], 1) == 91.5
    assert summary["lmda_net_reported_average_percent"] == 91.2
    assert summary["targets"]["dataset"]["topomap_images"] == 6000
