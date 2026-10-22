import numpy as np
import pytest
from prepare_support_band import make_band


def test_no_wrap_at_grid_boundary_and_no_refill_large_gap():
    coords = np.array([[0, 8, 8], [12, 8, 8], [0, 8, 8]], dtype=np.int64)
    band, audit = make_band(coords)
    assert not (band[:, 0] == 63).any()
    assert not ((band[:, 0] > 1) & (band[:, 0] < 11)).any()
    assert audit['grid_boundary_clips_band']
    assert audit['input_unique'] == 2
    assert audit['components6_after'] == 2


def test_preserve_source_order_and_warn_about_merged_gap():
    coords = np.array([[8, 8, 8], [10, 8, 8]])
    band, audit = make_band(coords)
    assert np.array_equal(band[:2], coords)
    assert audit['components6_before'] == 2
    assert audit['components6_after'] == 1


@pytest.mark.parametrize('coords', [np.array([[1.5, 2, 3]]), np.array([[-1, 2, 3]]), np.empty((0,3), int)])
def test_invalid_coordinates_are_rejected_not_silently_clamped(coords):
    with pytest.raises(ValueError):
        make_band(coords)
