"""Official geometry export, with optional cavity repair and no UV/texture."""
import argparse
import json
import runpy
import sys
from pathlib import Path


def install_geometry_export(inner_visibility_threshold=0.):
    import numpy as np
    import trimesh
    from trellis.utils import postprocessing_utils as pp
    from mesh_visibility_repair import install_visibility_repair
    install_visibility_repair(inner_visibility_threshold)

    def to_glb(app_rep, mesh, simplify=.95, fill_holes=True,
               fill_holes_max_size=.04, texture_size=1024, debug=False, verbose=True):
        vertices, faces = pp.postprocess_mesh(
            mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy(),
            simplify=simplify > 0, simplify_ratio=simplify, fill_holes=fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1-simplify)),
            fill_holes_resolution=1024, fill_holes_num_views=1000,
            debug=debug, verbose=verbose)
        vertices = vertices @ np.array([[1,0,0],[0,0,-1],[0,1,0]])
        return trimesh.Trimesh(vertices, faces, process=False)
    pp.to_glb = to_glb
