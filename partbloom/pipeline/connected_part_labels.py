"""Connected competitive region growing over an unchanged sparse support graph."""
import heapq
import numpy as np
from scipy import ndimage as ndi
from prepare_part_band import assign_parts


def component_labels(coords, parts, preference=4., core_margin=1.):
    """Apply connected growth within each existing global component.

    Separate wheels or other disconnected assemblies retain their geometry;
    this operation only assigns labels and never adds connecting material.
    """
    nearest, _, _ = assign_parts(coords, parts)
    grid = np.zeros((64,)*3, bool)
    grid[tuple(coords.T)] = True
    components, count = ndi.label(grid)
    groups = components[tuple(coords.T)]
    labels = np.full(len(coords), -1, np.int32)
    for group in range(1, count+1):
        rows = np.flatnonzero(groups == group)
        active = np.unique(nearest[rows])
        if len(active) == 1:
            labels[rows] = active[0]
        else:
            local, _ = connected_labels(coords[rows], [parts[i] for i in active], preference, core_margin)
            labels[rows] = active[local]
    return labels, {'global_components6': int(count), 'part_sizes': np.bincount(labels, minlength=len(parts)).tolist(),
        'changed_rows': int(np.sum(labels != nearest)), 'support_order_and_union_unchanged': True,
        'assignment': 'connected competitive growth within each existing global component'}


def connected_labels(coords, parts, preference=4., core_margin=1.):
    nearest, _, distances = assign_parts(coords, parts)
    grid = np.full((64,)*3, -1, dtype=np.int32)
    grid[tuple(coords.T)] = np.arange(len(coords))
    support = grid >= 0
    if ndi.label(support)[1] != 1:
        raise ValueError('Expected one face-connected global support')
    sorted_d = np.sort(distances, axis=1)
    margin = sorted_d[:, 1]-sorted_d[:, 0]
    labels = np.full(len(coords), -1, dtype=np.int32)
    cores = []
    for part in range(len(parts)):
        mask = np.zeros_like(support)
        select = (nearest == part) & (margin >= core_margin)
        mask[tuple(coords[select].T)] = True
        components, count = ndi.label(mask)
        if count:
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            indices = grid[components == sizes.argmax()]
        else:
            candidates = np.flatnonzero(nearest == part)
            indices = candidates[[np.argmax(margin[candidates])]]
        labels[indices] = part
        cores.append(indices)
    steps = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    neighbors = []
    for coord in coords:
        q = coord+steps
        q = q[((q >= 0) & (q < 64)).all(1)]
        indices = grid[tuple(q.T)]
        neighbors.append(indices[indices >= 0])
    penalty = 1+preference*(distances-distances.min(1)[:, None])**2
    heap = []
    best = np.full((len(coords), len(parts)), np.inf)

    def expand(index, part, cost):
        for other in neighbors[index]:
            if labels[other] >= 0:
                continue
            candidate = cost+penalty[other, part]
            if candidate < best[other, part]:
                best[other, part] = candidate
                heapq.heappush(heap, (candidate, int(part), int(other)))

    for part, indices in enumerate(cores):
        for index in indices:
            expand(index, part, 0.)
    while heap:
        cost, part, index = heapq.heappop(heap)
        if labels[index] >= 0 or cost != best[index, part]:
            continue
        labels[index] = part
        expand(index, part, cost)
    if np.any(labels < 0):
        raise RuntimeError('Unassigned support')
    for part in range(len(parts)):
        mask = np.zeros_like(support)
        mask[tuple(coords[labels == part].T)] = True
        if ndi.label(mask)[1] != 1:
            raise RuntimeError('Connected growth invariant violated')
    return labels, {'preference': preference, 'core_margin': core_margin,
        'core_sizes': [len(x) for x in cores], 'changed_rows': int(np.sum(labels != nearest)),
        'changed_fraction': float(np.mean(labels != nearest)),
        'part_sizes': np.bincount(labels).tolist(), 'all_parts_face_connected': True,
        'support_order_and_union_unchanged': True}
