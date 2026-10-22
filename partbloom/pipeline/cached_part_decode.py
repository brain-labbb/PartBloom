"""Decode inherited parts from frozen SLat, with optional joint feature context."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mesh-inner-visibility',type=float,default=.01)
    parser.add_argument('--retry-failed', action='store_true', help='Archive a failed pre-export status, then retry')
    args = parser.parse_args()
    out = args.out.resolve()
    status_file = out/'decode_status.json'
    if args.retry_failed and status_file.exists() and not (out/'objs').exists():
        status = json.loads(status_file.read_text())
        if status['status'] != 'failed':
            raise ValueError('Only failed pre-export runs can be retried')
        archive = out/'decode_attempt_failed.json'
        if archive.exists():
            raise FileExistsError('Failure archive already exists')
        status_file.rename(archive)
    if (out/'decode_status.json').exists() or (out/'objs').exists():
        raise FileExistsError('Refusing to overwrite an existing decode')
    config = json.loads((out/'experiment.json').read_text())
    source = Path(config['latent_source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config['latent_sha256']
    assert hashlib.sha256((out/'allind.npy').read_bytes()).hexdigest() == config['support_sha256']
    ownership = np.load(out/'part_labels.npy')
    part_count = len(json.loads((out/'part_index.json').read_text()))
    expected = np.load(out/'allind.npy')
    from resources import PIPELINE, VENDOR
    sys.path.insert(0, str(VENDOR/'PhysX-Omni'))
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.modules.sparse import SparseTensor
    from geometry_decode import install_geometry_export
    install_geometry_export(args.mesh_inner_visibility)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    assert np.array_equal(saved['coords'][:, 1:].numpy(), expected)

    def to_cpu(mesh):
        for key in ['vertices','faces','vertex_attrs','face_normal','tsdf_v','tsdf_s','reg_loss']:
            value = getattr(mesh, key, None)
            if torch.is_tensor(value):
                setattr(mesh, key, value.detach().cpu())
        if not mesh.success:
            raise RuntimeError('Empty part mesh')
        return mesh

    @torch.no_grad()
    def run(self, coords, image, **kwargs):
        assert np.array_equal(coords[:, 1:].cpu().numpy(), expected)
        slat = SparseTensor(coords=saved['coords'].cuda(), feats=saved['feats'].cuda())
        decoder = self.models['slat_decoder_mesh']
        labels = torch.as_tensor(ownership, device='cuda', dtype=torch.long)
        if config['mode'] == 'parts':
            meshes = []
            for i in range(part_count):
                selected = labels == i
                mesh = decoder(SparseTensor(coords=slat.coords[selected], feats=slat.feats[selected]))[0]
                meshes.append(to_cpu(mesh))
        elif config['mode'] == 'joint':
            original = decoder.to_representation
            audit = {}
            def extract_parts(features):
                # SparseSubdivide creates a 4x4x4 child block per input cell.
                assert decoder.mesh_extractor.res == 256
                grid = torch.full((64,64,64), -1, dtype=torch.long, device='cuda')
                grid[slat.coords[:,1].long(),slat.coords[:,2].long(),slat.coords[:,3].long()] = labels
                parent = torch.div(features.coords[:,1:].long(), 4, rounding_mode='floor')
                owners = grid[parent[:,0], parent[:,1], parent[:,2]]
                assert torch.all(owners >= 0), 'High-resolution cell outside source support'
                if config.get('fine_ownership_repair'):
                    from scipy.spatial import cKDTree
                    from scipy import ndimage as ndi
                    repair = config['fine_ownership_repair']
                    fine_coords = features.coords[:,1:].cpu().numpy()
                    centers = (fine_coords+.5)/256-.5
                    new_owners = owners.cpu().numpy().copy()
                    initial_owners = new_owners.copy()
                    changes = []
                    for patch in repair['proposal']:
                        donor, recipient = patch['donor'], patch['recipient']
                        candidates = np.flatnonzero(new_owners == donor)
                        distance = cKDTree(patch['fragment_vertices']).query(centers[candidates])[0]
                        selected = candidates[distance <= repair['radius_fine_cells']/256]
                        new_owners[selected] = recipient
                        changes.append({'donor':donor,'recipient':recipient,'fine_cells':len(selected)})
                    for part in range(part_count):
                        mask = np.zeros((256,)*3,bool)
                        mask[tuple(fine_coords[new_owners == part].T)] = True
                        if ndi.label(mask)[1] != 1:
                            raise RuntimeError('Fine ownership correction disconnected a part support')
                    owners = torch.as_tensor(new_owners,device='cuda',dtype=torch.long)
                    np.save(out/'fine_ownership_changed_indices.npy',np.flatnonzero(new_owners != initial_owners))
                    audit['fine_repair'] = {'transfers':changes,
                        'changed_fine_cells':int(np.sum(new_owners != initial_owners)),
                        'all_fine_part_supports_connected':True,'no_geometry_features_deleted':True}
                audit.update(high_resolution_cells=len(owners), counts=torch.bincount(owners).cpu().tolist(),
                             disjoint_ownership=True, no_cells_discarded=True)
                result = []
                for i in range(part_count):
                    selected = owners == i
                    # mesh_extractor adds SDF bias in-place: each part gets a fresh tensor.
                    owned = SparseTensor(coords=features.coords[selected], feats=features.feats[selected].clone())
                    result.append(to_cpu(decoder.mesh_extractor(owned, training=False)))
                return result
            decoder.to_representation = extract_parts
            try:
                meshes = decoder(slat)
            finally:
                decoder.to_representation = original
            (out/'feature_ownership_audit.json').write_text(json.dumps(audit, indent=2))
        else:
            raise ValueError(config['mode'])
        (out/'replay_protocol.json').write_text(json.dumps({**config,
            'SLat_sampling_calls': 0, 'SS_sampling_calls': 0, 'API_calls': 0,
            'geometry_network_weights': 'unchanged TRELLIS 1 mesh decoder',
            'geometry_postprocess': 'official settings with optional low-visibility inner seeds',
            'mesh_inner_visibility':args.mesh_inner_visibility,
            'texture': 'omitted; gaussian placeholder ignored by geometry-only exporter'}, indent=2))
        return [{'mesh': [mesh], 'gaussian': [None]} for mesh in meshes]

    TrellisImageTo3DPipeline.run_decoder = run
    sys.argv = ['official_entry.py','decode','--out',str(out)]
    runpy.run_path(str(PIPELINE/'official_entry.py'), run_name='__main__')
    assert len(list((out/'objs').glob('*/*.obj'))) == part_count
    print('CACHED_PART_DECODE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
