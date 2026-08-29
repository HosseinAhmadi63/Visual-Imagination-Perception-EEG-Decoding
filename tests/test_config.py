from vip_eeg.config import config_hash, load_config


def test_paper_config_has_exact_cohort_and_frame_count():
    config = load_config("configs/paper.yaml")
    recordings = config["dataset"]["recordings"]
    assert len(recordings) == 15
    assert len({item["subject"] for item in recordings}) == 12
    total = len(recordings) * sum(config["tasks"][task]["frames"] for task in config["tasks"])
    assert total == 6000


def test_config_hash_is_stable():
    first = load_config("configs/paper.yaml")
    second = load_config("configs/paper.yaml")
    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 12
