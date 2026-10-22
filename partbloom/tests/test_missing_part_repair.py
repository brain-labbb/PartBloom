import numpy as np
import pytest

from connected_part_labels import component_labels
from local_geometry_repair import restore_missing_parts


def test_restore_lost_small_part_without_changing_existing_support():
    large = np.array([[x, 4, 4] for x in range(10)])
    small = np.array([[5, 7, 4], [6, 7, 4]])
    result, audit = restore_missing_parts(large, [large, small])
    assert np.array_equal(result[:len(large)], large)
    assert set(map(tuple, result[len(large):])) == set(map(tuple, small))
    assert audit['restored_part_indices'] == [1]
    assert audit['added_voxels'] == 2
    labels, _ = component_labels(result, [large, small])
    assert set(labels) == {0, 1}


def test_no_change_when_all_parts_survive():
    source = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3]])
    result, audit = restore_missing_parts(source, [source[:1], source[2:]])
    assert np.array_equal(result, source)
    assert audit['added_voxels'] == 0


def test_ambiguous_duplicate_parts_are_not_silently_accepted():
    coords = np.array([[1, 2, 3]])
    with pytest.raises(ValueError, match='no support'):
        restore_missing_parts(coords, [coords, coords])
