import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_visibility_repair import build_visibility_fill

# CPU stand-in for the single tensor operation being adapted.
torch = np


def example_fill(visblity):
    return torch.nonzero(visblity == 0)


def test_visibility_changes_only_seed_classification():
    visibility=np.array([0.,.001,.009,.01,.2])
    modified=build_visibility_fill(example_fill,.01)
    assert np.array_equal(modified(visibility)[0],[0,1,2])
    assert np.array_equal(example_fill(visibility)[0],[0])
    assert build_visibility_fill(example_fill,0) is example_fill
    with pytest.raises(ValueError): build_visibility_fill(example_fill,.25)
