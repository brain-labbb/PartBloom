"""Image-guided band refinement and joint-feature part mesh extraction for any asset."""
import argparse
import gc
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

from resources import PIPELINE, VENDOR, WEIGHTS_ROOT
WEIGHTS = WEIGHTS_ROOT/'TRELLIS-image-large/ckpts'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--noise-time', type=float, default=.65)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--mesh-inner-visibility', type=float, default=.01)
    parser.add_argument('--decode-mode', choices=['joint','parts'], default='parts')
    parser.add_argument('--resume-latent', action='store_true')
    parser.add_argument('--restore-missing-parts', action='store_true')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if args.resume_latent:
        latent=out/'shared_slat.pt'
        (out/'experiment.json').write_text(json.dumps({'mode':args.decode_mode,'latent_source':str(latent),
            'latent_sha256':hashlib.sha256(latent.read_bytes()).hexdigest(),
            'support_sha256':hashlib.sha256((out/'allind.npy').read_bytes()).hexdigest(),
            'labels':'connected growth within existing global components'},indent=2))
        os.environ['AGENTVOXEL_SAVE_MESH_STAGES']='1'
        sys.argv=['cached_part_decode.py','--out',str(out),'--retry-failed',
            '--mesh-inner-visibility',str(args.mesh_inner_visibility)]
        runpy.run_path(str(PIPELINE/'cached_part_decode.py'),run_name='__main__')
        return
    if not 0 < args.noise_time < 1 or args.steps < 1:
        raise ValueError('Expected 0 < noise-time < 1 and positive steps')
    index = json.loads((source/'part_index.json').read_text())
    parts = [np.load(source/f'ind_{i}.npy') for i in range(len(index))]
    from prepare_support_band import make_band
    from connected_part_labels import component_labels
    from local_geometry_repair import ordered_support
    band, band_audit = make_band(np.load(source/'allind.npy'))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source/'cond_img.png', out/'cond_img.png')
    for name in ['part_graph.json', 'basic_info.txt', 'prompt_protocol.json']:
        if (source/name).exists():
            shutil.copy2(source/name, out/name)
    np.save(out/'band_support.npy', band)
    np.save(out/'allind.npy', band)
    np.save(out/'ind_0.npy', band)
    protocol = {'source': str(source), 'source_part_count': len(parts),
        'pipeline': 'one-cell band -> partial image-guided SS refinement -> shared SLat -> selected mesh decoding context -> part meshes',
        'decode_mode': args.decode_mode,
        'noise_time': args.noise_time, 'steps': args.steps, 'seed_structure': 1, 'seed_slat': 1,
        'restore_missing_parts': args.restore_missing_parts,
        'cfg_strength': 5., 'cfg_interval': [args.noise_time/2, args.noise_time],
        'mesh_inner_visibility': args.mesh_inner_visibility, 'band_audit': band_audit,
        'native_structure_generation': False, 'texture': False}
    (out/'pipeline_protocol.json').write_text(json.dumps(protocol, indent=2))
    sys.path.insert(0, str(VENDOR/'PhysX-Omni'))
    import torch
    from trellis import models
    from trellis.modules.sparse import SparseTensor
    from trellis.pipelines import TrellisImageTo3DPipeline
    from geometry_decode import install_geometry_export
    install_geometry_export(args.mesh_inner_visibility)

    @torch.no_grad()
    def run(self, coords, image, **kwargs):
        encoder = models.from_pretrained(str(WEIGHTS/'ss_enc_conv3d_16l8_fp16')).eval().cuda()
        grid = np.zeros((64,)*3, np.float32)
        grid[tuple(band.T)] = 1
        z = encoder(torch.from_numpy(grid[None,None]).cuda(), sample_posterior=False)
        del encoder
        gc.collect(); torch.cuda.empty_cache()
        cond = self.get_cond([self.preprocess_image(image)])
        torch.manual_seed(1)
        sampler = self.sparse_structure_sampler
        t = args.noise_time
        sample = (1-t)*z + (sampler.sigma_min+(1-sampler.sigma_min)*t)*torch.randn_like(z)
        timeline = np.linspace(t, 0, args.steps+1)
        for now, then in zip(timeline[:-1], timeline[1:]):
            sample = sampler.sample_once(self.models['sparse_structure_flow_model'], sample,
                float(now), float(then), **cond, cfg_strength=5., cfg_interval=(t/2,t)).pred_x_prev
        logits = self.models['sparse_structure_decoder'](sample)
        support = (logits[0,0] > 0).cpu().numpy()
        array = ordered_support(support, band)
        if args.restore_missing_parts:
            from local_geometry_repair import restore_missing_parts
            array, repair = restore_missing_parts(array, parts)
            (out/'support_repair.json').write_text(json.dumps(repair, indent=2))
        np.save(out/'allind.npy', array)
        labels, audit = component_labels(array, parts)
        np.save(out/'part_labels.npy', labels)
        (out/'partition.json').write_text(json.dumps(audit, indent=2))
        for i, entry in enumerate(index):
            owned = array[labels == i]
            np.save(out/f'ind_{i}.npy', owned)
            entry['coords'] = len(owned)
        (out/'part_index.json').write_text(json.dumps(index, indent=2))
        coords = torch.from_numpy(np.column_stack([np.zeros(len(array), dtype=np.int32), array])).int().cuda()
        torch.manual_seed(1)
        slat = self.sample_slat(cond, coords)
        torch.save({'coords': slat.coords.cpu(), 'feats': slat.feats.cpu()}, out/'shared_slat.pt')
        decoder = self.models['slat_decoder_mesh']
        if args.decode_mode == 'parts':
            owners=torch.as_tensor(labels,device='cuda',dtype=torch.long)
            meshes=[]
            for i in range(len(parts)):
                selected=owners==i
                mesh=decoder(SparseTensor(coords=slat.coords[selected],feats=slat.feats[selected]))[0]
                if not mesh.success:
                    raise RuntimeError('Empty mesh for '+str(index[i]['part']))
                for key in ['vertices','faces','vertex_attrs','face_normal','tsdf_v','tsdf_s','reg_loss']:
                    value=getattr(mesh,key,None)
                    if torch.is_tensor(value):setattr(mesh,key,value.detach().cpu())
                meshes.append(mesh)
            return [{'mesh':[mesh],'gaussian':[None]} for mesh in meshes]
        original = decoder.to_representation
        def extract(features):
            owner_grid = torch.full((64,)*3, -1, dtype=torch.long, device='cuda')
            owner_grid[coords[:,1].long(),coords[:,2].long(),coords[:,3].long()] = torch.as_tensor(labels, device='cuda', dtype=torch.long)
            assert decoder.mesh_extractor.res == 256
            parent = torch.div(features.coords[:,1:].long(), 4, rounding_mode='floor')
            owners = owner_grid[parent[:,0],parent[:,1],parent[:,2]]
            if torch.any(owners < 0):
                raise RuntimeError('Fine feature outside inherited support')
            meshes = []
            for i in range(len(parts)):
                mask = owners == i
                mesh = decoder.mesh_extractor(SparseTensor(coords=features.coords[mask],
                    feats=features.feats[mask].clone()), training=False)
                if not mesh.success:
                    raise RuntimeError('Empty mesh for '+str(index[i]['part']))
                for key in ['vertices','faces','vertex_attrs','face_normal','tsdf_v','tsdf_s','reg_loss']:
                    value = getattr(mesh, key, None)
                    if torch.is_tensor(value):
                        setattr(mesh, key, value.detach().cpu())
                meshes.append(mesh)
            (out/'feature_ownership_audit.json').write_text(json.dumps({
                'high_resolution_cells': len(owners), 'counts': torch.bincount(owners).cpu().tolist(),
                'no_cells_discarded': True, 'disjoint_ownership': True}, indent=2))
            return meshes
        decoder.to_representation = extract
        try:
            meshes = decoder(slat)
        finally:
            decoder.to_representation = original
        return [{'mesh': [mesh], 'gaussian': [None]} for mesh in meshes]

    TrellisImageTo3DPipeline.run_decoder = run
    os.environ['AGENTVOXEL_SAVE_MESH_STAGES'] = '1'
    sys.argv = ['official_entry.py', 'decode', '--out', str(out)]
    runpy.run_path(str(PIPELINE/'official_entry.py'), run_name='__main__')
    if len(list((out/'objs').glob('*/*.obj'))) != len(parts):
        raise RuntimeError('Export count differs from planned parts')


if __name__ == '__main__':
    main()
