from vip_eeg.data.events import EventRecord, normalize_description, select_first_pictorial_pair


def test_description_normalization():
    value = "0, Perception_image_flower_m###my_stream_name"
    assert normalize_description(value) == "Perception_image_flower_m"


def test_first_matched_pictorial_pair_is_deterministic():
    events = [
        EventRecord(100, 0.1, "Perception_t_flower_1"),
        EventRecord(200, 1.0, "Perception_image_flower_m"),
        EventRecord(300, 4.0, "Imagination_image_penguin_s"),
        EventRecord(400, 4.4, "Imagination_image_flower_m"),
        EventRecord(500, 8.0, "Perception_image_penguin_s"),
        EventRecord(600, 11.4, "Imagination_image_penguin_s"),
    ]
    pair = select_first_pictorial_pair(events)
    assert pair.stimulus == "penguin_s"
    assert pair.perception.sample == 500
    assert pair.imagination.sample == 600
