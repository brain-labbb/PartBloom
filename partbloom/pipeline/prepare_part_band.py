"""Partition existing band support, without changing its coordinates or order."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def assign_parts(coords, parts):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
        raise ValueError('Expected nonempty Nx3 support')
    if not np.issubdtype(coords.dtype, np.integer) or coords.min() < 0 or coords.max() > 63:
        raise ValueError('Expected integer support in [0,63]')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Band support must have unique coordinates')
    if not parts or any(np.asarray(p).ndim != 2 or np.asarray(p).shape[1] != 3 or not len(p) for p in parts):
        raise ValueError('Expected nonempty part coordinate arrays')
    distances = np.column_stack([cKDTree(p).query(coords)[0] for p in parts])
    labels = distances.argmin(axis=1)
    tied = np.sum(np.isclose(distances, distances.min(axis=1)[:, None], atol=1e-12, rtol=0), axis=1) > 1
    if set(labels.tolist()) != set(range(len(parts))):
        raise ValueError('A source part received no support')
    return labels, tied, distances
