import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventRecord:
    sample: int
    onset_seconds: float
    description: str


@dataclass(frozen=True)
class PictorialPair:
    perception: EventRecord
    imagination: EventRecord
    stimulus: str


def normalize_description(description: str) -> str:
    value = description.strip().lstrip("\ufeff")
    value = re.sub(r"^\s*0\s*,\s*", "", value)
    value = value.split("###", maxsplit=1)[0]
    return value.strip()


def raw_annotation_events(raw: Any) -> list[EventRecord]:
    import mne

    events, event_ids = mne.events_from_annotations(raw, regexp=None, verbose=False)
    descriptions = {value: key for key, value in event_ids.items()}
    records = []
    for sample, _, event_code in events:
        relative_sample = int(sample - raw.first_samp)
        records.append(
            EventRecord(
                sample=relative_sample,
                onset_seconds=relative_sample / float(raw.info["sfreq"]),
                description=normalize_description(descriptions[int(event_code)]),
            )
        )
    return sorted(records, key=lambda item: item.sample)


def select_first_pictorial_pair(
    events: Iterable[EventRecord],
    perception_prefix: str = "Perception_image_",
    imagination_prefix: str = "Imagination_image_",
    maximum_gap_seconds: float = 4.5,
) -> PictorialPair:
    ordered = sorted(events, key=lambda item: item.sample)
    for index, perception in enumerate(ordered[:-1]):
        if not perception.description.startswith(perception_prefix):
            continue
        stimulus = perception.description[len(perception_prefix) :]
        expected = f"{imagination_prefix}{stimulus}"
        imagination = ordered[index + 1]
        gap = imagination.onset_seconds - perception.onset_seconds
        if imagination.description == expected and 0 < gap <= maximum_gap_seconds:
            return PictorialPair(
                perception=perception,
                imagination=imagination,
                stimulus=stimulus,
            )
    raise ValueError("No matched pictorial Perception/Imagination event pair was found")
