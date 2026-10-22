"""Stable support ordering used by the band adapter."""
import numpy as np


def restore_missing_parts(coords, parts):
    """Restore source voxels only for parts lost by structure refinement."""
    from scipy.spatial import cKDTree
    from prepare_part_band import assign_parts

    distances = np.column_stack([cKDTree(p).query(coords)[0] for p in parts])
    counts = np.bincount(distances.argmin(axis=1), minlength=len(parts))
    missing = np.flatnonzero(counts == 0)
    result = [tuple(map(int, row)) for row in coords]
    seen = set(result)
    for i in missing:
        for row in parts[i]:
            key = tuple(map(int, row))
            if key not in seen:
                result.append(key)
                seen.add(key)
    repaired = np.asarray(result, dtype=np.int64).reshape(-1, 3)
    # Reject overlapping/ambiguous sources that still cannot own any support.
    labels, _, _ = assign_parts(repaired, parts)
    return repaired, {'restored_part_indices': missing.tolist(),
                      'added_voxels': len(repaired) - len(coords),
                      'part_sizes_before': counts.tolist(),
                      'part_sizes_after': np.bincount(labels, minlength=len(parts)).tolist(),
                      'policy': 'restore original source voxels of unsupported parts only'}

def ordered_support(grid, original):
    """Preserve first-occurrence order; append newly occupied coordinates by ZYX."""
    result, seen = [], set()
    for row in original:
        key = tuple(int(v) for v in row)
        if grid[key] and key not in seen:
            result.append(key)
            seen.add(key)
    for row in np.argwhere(grid.transpose(2, 1, 0))[:, ::-1]:
        key = tuple(int(v) for v in row)
        if key not in seen:
            result.append(key)
            seen.add(key)
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)
