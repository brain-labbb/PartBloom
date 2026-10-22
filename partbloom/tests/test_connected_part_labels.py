import numpy as np
import pytest
from scipy import ndimage as ndi
from connected_part_labels import connected_labels
from connected_part_labels import component_labels


def test_components_keep_separate_assemblies_and_shared_part():
    coords = np.array([[x, 2, 2] for x in range(10)] + [[x, 20, 2] for x in range(3)])
    parts = [coords[[0,1,10,11,12]], coords[8:10]]
    labels, audit = component_labels(coords, parts)
    assert audit['global_components6'] == 2
    assert (labels[-3:] == 0).all()
    assert labels[0] == 0 and labels[9] == 1
    assert sum(audit['part_sizes']) == len(coords)


def test_single_functional_part():
    coords = np.array([[2,2,2], [3,2,2]])
    labels, _ = component_labels(coords, [coords])
    assert labels.tolist() == [0,0]


def test_connected_partition_preserves_support_and_labels():
    coords = np.array([[x,y,10] for x in range(10,21) for y in range(10,15)])
    parts = [coords[coords[:,0] <= 12], coords[coords[:,0] >= 18]]
    saved = coords.copy()
    labels, audit = connected_labels(coords, parts)
    assert np.array_equal(coords, saved)
    assert set(labels) == {0,1}
    assert labels[coords[:,0] == 10].tolist() == [0]*5
    assert labels[coords[:,0] == 20].tolist() == [1]*5
    for i in [0,1]:
        grid = np.zeros((64,)*3,bool); grid[tuple(coords[labels==i].T)] = True
        assert ndi.label(grid)[1] == 1
    assert audit['all_parts_face_connected']


def test_disconnected_union_rejected():
    coords = np.array([[0,0,0],[10,10,10]])
    with pytest.raises(ValueError, match='global support'):
        connected_labels(coords, [coords[:1],coords[1:]])


def test_row_permutation_does_not_change_connected_partition():
    coords = np.array([[x,y,2] for x in range(12) for y in range(4)])
    parts = [coords[coords[:,0] < 3], coords[coords[:,0] > 8]]
    a,_ = connected_labels(coords,parts)
    permutation = np.random.default_rng(1).permutation(len(coords))
    b,_ = connected_labels(coords[permutation],parts)
    assert dict(zip(map(tuple,coords),a)) == dict(zip(map(tuple,coords[permutation]),b))
