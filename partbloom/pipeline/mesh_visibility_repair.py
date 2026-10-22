"""Optional low-visibility cavity repair for the bundled TRELLIS exporter.

The upstream mincut uses only faces visible from zero sampled cameras as inner
seeds. This adapter allows a small nonzero visibility threshold. All remaining
mincut, cut-size checks and boundary filling are unchanged. It is an inference
postprocess, not a learned semantic detector, and can also remove real recesses.
"""
import inspect
import textwrap


def build_visibility_fill(original, threshold):
    """Build a process-local variant without editing the shared vendor checkout."""
    if not 0 <= threshold < .25:
        raise ValueError('Inner visibility must be in [0, .25)')
    if threshold == 0:
        return original
    source = textwrap.dedent(inspect.getsource(original))
    needle = 'torch.nonzero(visblity == 0)'
    if source.count(needle) != 1:
        raise RuntimeError('Upstream hole-fill implementation changed; adapter needs review')
    implementation = source.replace(needle,'torch.nonzero(visblity < _inner_visibility)')
    # Unwrap torch.no_grad so the function globals refer to the upstream module.
    namespace = dict(inspect.unwrap(original).__globals__, _inner_visibility=threshold)
    exec(compile(implementation,'<low_visibility_hole_fill>','exec'),namespace)
    return namespace[original.__name__]


def install_visibility_repair(threshold=0.):
    from trellis.utils import postprocessing_utils as pp
    original = getattr(pp,'_partbloom_original_fill_holes',pp._fill_holes)
    if threshold == 0:
        pp._fill_holes = original
        return
    replacement = build_visibility_fill(original,threshold)
    pp._partbloom_original_fill_holes = original
    pp._fill_holes = replacement
