import numpy as np

from vip_eeg.data.prepare import frame_offsets
from vip_eeg.evaluation.statistics import cluster_mask, sample_at


def test_paper_frame_intervals_produce_unique_samples():
    perception = {"frames": 200, "interval_seconds": 0.015}
    imagination = {"frames": 200, "interval_seconds": 0.020}
    assert len(frame_offsets(perception, 1024.0)) == 200
    assert len(frame_offsets(imagination, 1024.0)) == 200
    assert frame_offsets(perception, 1024.0)[-1] == round(2.985 * 1024)
    assert frame_offsets(imagination, 1024.0)[-1] == round(3.980 * 1024)


def test_exact_difference_time_sampling():
    epoch = np.arange(5000, dtype=np.float64)[None, :]
    assert sample_at(epoch, 1024.0, 0.400)[0] == 410
    assert sample_at(epoch, 1024.0, 0.415)[0] == 425
    assert sample_at(epoch, 1024.0, 0.420)[0] == 430


def test_cluster_masks_accept_mne_boolean_and_index_outputs():
    boolean = cluster_mask((np.array([True, False, True]),), 3)
    indexed = cluster_mask((np.array([0, 2]),), 3)
    assert boolean.tolist() == [True, False, True]
    assert indexed.tolist() == [True, False, True]
