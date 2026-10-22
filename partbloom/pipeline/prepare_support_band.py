"""Opt-in whole-object support-band adapter for existing Agent sparse exports.

Consumes surface/support XYZ, not an arbitrary dense solid. Keeps all original
positions and adds their six face neighbors. Does not extract a shell afterward.
This was tested on one chair; it is not a universal decoder-compatibility fix.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from local_geometry_repair import ordered_support


def make_band(coords):
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('Expected integer (N,3) sparse support in XYZ order')
    if not len(coords) or coords.min() < 0 or coords.max() >= 64:
        raise ValueError('Coordinates must be nonempty and within 0..63')
    grid = np.zeros((64,) * 3, bool)
    grid[tuple(coords.T)] = True
    band = ndi.binary_dilation(grid, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    result = ordered_support(band, coords)
    audit = {'input_rows': len(coords), 'input_unique': int(grid.sum()),
        'output_rows': len(result), 'input_positions_retained': bool(band[grid].all()),
        'support_expansion_max_voxels': 1, 'components6_before': int(ndi.label(grid)[1]),
        'components6_after': int(ndi.label(band)[1]),
        'grid_boundary_clips_band': bool(np.any((coords == 0) | (coords == 63))),
        'warning': 'May thicken shape or merge narrow gaps; inspect geometry and image fidelity'}
    return result, audit
