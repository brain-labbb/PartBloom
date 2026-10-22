"""Restore TRELLIS native Gaussian appearance decoding and texture baking.

This adapter reuses upstream render_multiview and bake_texture on saved final
part meshes; it does not implement a new texture-generation algorithm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def resolve_source(source):
    source = Path(source).resolve()
    if not (source / 'part_index.json').exists() and (source / 'refined').is_dir():
        source /= 'refined'
    latent = source / 'shared_slat.pt'
    if not latent.exists() and (source / 'experiment.json').is_file():
        experiment = json.loads((source / 'experiment.json').read_text())
        latent = Path(experiment['latent_source'])
        if not latent.is_absolute():
            latent = source / latent
        if sha(latent) != experiment['latent_sha256']:
            raise ValueError('Cached decode latent hash differs from its provenance')
    if not latent.is_file():
        raise FileNotFoundError(f'Saved shared SLat required: {latent}')
    entries = json.loads((source / 'part_index.json').read_text())
    if not entries or len({e['index'] for e in entries}) != len(entries):
        raise ValueError('Part indices must be nonempty and unique')
    paths = {e['index']: source / 'objs' / str(e['index']) / f"{e['index']}.obj" for e in entries}
    for path in [source / 'part_labels.npy', source / 'allind.npy', *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return source, latent.resolve(), entries, paths


def artifacts_valid(out, record):
    return bool(record.get('artifacts')) and all(
        (out / name).is_file() and sha(out / name) == digest
        for name, digest in record['artifacts'].items())


def canonical(vertices, faces):
    import numpy as np
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1).reshape(-1, 9)
    return triangles[np.lexsort(tuple(triangles[:, i] for i in range(8, -1, -1)))]


def export(source, out, texture_size=1024, resume=False):
    from resources import VENDOR, WEIGHTS_ROOT
    source, latent, entries, paths = resolve_source(source)
    model_config = WEIGHTS_ROOT / 'TRELLIS-image-large/pipeline.json'
    model = json.loads(model_config.read_text())['args']['models']['slat_decoder_gs']
    signature = {'version': 1, 'source': str(source), 'latent_sha256': sha(latent),
                 'labels_sha256': sha(source / 'part_labels.npy'),
                 'support_sha256': sha(source / 'allind.npy'),
                 'parts_sha256': sha(source / 'part_index.json'),
                 'meshes': {str(i): sha(p) for i, p in paths.items()},
                 'weights': str(WEIGHTS_ROOT), 'model_config_sha256': sha(model_config),
                 'gaussian_decoder': model, 'texture_size': texture_size,
                 'views': 100, 'resolution': 1024, 'bake_steps': 2500, 'bake_seed': 1}
    out = Path(out).resolve()
    status_path = out / 'texture_status.json'
    if out.exists():
        if not resume or not status_path.exists():
            raise FileExistsError('Choose a new texture output directory, or --resume a matching export')
        previous = json.loads(status_path.read_text())
        if previous.get('signature') != signature:
            raise ValueError('Texture source/settings differ from saved export; choose a new output directory')
        if previous.get('status') == 'complete' and artifacts_valid(out, previous):
            print(f'Texture export already complete: {out}', flush=True)
            return
    else:
        out.mkdir(parents=True)
    started = time.time()
    status = {'status': 'running', 'signature': signature, 'parts': [],
              'API_calls': 0, 'SS_sampling_calls': 0, 'SLat_sampling_calls': 0,
              'mesh_decoder_calls': 0, 'mesh_postprocessing': False,
              'coordinates': 'Source Z-up; exported GLB standard Y-up (x,z,-y)',
              'geometry_policy': 'Existing triangles preserved; only UV seam vertex duplication',
              'appearance': 'Per-part Gaussian decoder from frozen shared SLat; baked base color'}
    write_json(status_path, status)
    try:
        os.environ.setdefault('SPCONV_ALGO', 'native')
        os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
        sys.path.insert(0, str(VENDOR / 'PhysX-Omni'))
        import gc
        import numpy as np
        import torch
        import trimesh
        import xatlas
        from PIL import Image
        from trellis import models
        from trellis.modules.sparse import SparseTensor
        from trellis.utils.postprocessing_utils import bake_texture
        from trellis.utils.render_utils import render_multiview
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        saved = torch.load(latent, map_location='cpu', weights_only=True)
        labels = np.load(source / 'part_labels.npy')
        if len(labels) != len(saved['coords']) or not np.array_equal(saved['coords'][:, 1:].numpy(), np.load(source / 'allind.npy')):
            raise ValueError('SLat coordinates and part labels do not match mesh source')
        if set(np.unique(labels)) != set(paths):
            raise ValueError('Every planned part must have latent support; unknown labels are invalid')
        decoder = None  # Load only when a missing part actually needs baking.
        scene = trimesh.Scene()
        artifacts = {}
        for entry in entries:
            i = entry['index']
            name = entry['part']
            part_out = out / 'parts' / str(i)
            part_out.mkdir(parents=True, exist_ok=True)
            audit = part_out / 'audit.json'
            if resume and audit.exists():
                record = json.loads(audit.read_text())
                if record.get('signature') == signature and artifacts_valid(out, record):
                    scene.add_geometry(trimesh.load(part_out / 'textured.glb', force='mesh', process=False),
                                       node_name=str(i), geom_name=str(i))
                    artifacts.update(record['artifacts'])
                    status['parts'].append(record)
                    continue
            part_started = time.time()
            mesh = trimesh.load(paths[i], force='mesh', process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if not len(faces):
                raise ValueError(f'Empty mesh: {paths[i]}')
            print(f'TEXTURE_PART_START {i} {name}', flush=True)
            if decoder is None:
                decoder = models.from_pretrained(str(WEIGHTS_ROOT / 'TRELLIS-image-large' / model)).eval().cuda()
            selected = torch.from_numpy(labels == i)
            torch.manual_seed(1 + i)
            np.random.seed(1 + i)
            with torch.no_grad():
                slat = SparseTensor(coords=saved['coords'][selected].cuda(), feats=saved['feats'][selected].cuda())
                gaussian = decoder(slat)[0]
                observations, extrinsics, intrinsics = render_multiview(gaussian, resolution=1024, nviews=100)
            masks = [np.any(obs > 0, axis=-1) for obs in observations]
            extrinsics = [e.cpu().numpy() for e in extrinsics]
            intrinsics = [e.cpu().numpy() for e in intrinsics]
            del gaussian, slat
            mapping, uv_faces, uvs = xatlas.parametrize(vertices, faces)
            uv_vertices = vertices[mapping]
            uv_faces = np.asarray(uv_faces, dtype=np.int32)
            if not np.array_equal(canonical(uv_vertices, uv_faces), canonical(vertices, faces)):
                raise RuntimeError('UV unwrapping changed triangle geometry')
            texture = bake_texture(uv_vertices, uv_faces, np.asarray(uvs, dtype=np.float32),
                                   observations, masks, extrinsics, intrinsics,
                                   texture_size=texture_size, mode='opt', lambda_tv=.01, verbose=False)
            image = Image.fromarray(texture)
            image.save(part_out / 'base_color.png')
            material = trimesh.visual.material.PBRMaterial(name=name, roughnessFactor=1., metallicFactor=0.,
                baseColorTexture=image, baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8))
            textured = trimesh.Trimesh(uv_vertices @ rotation, uv_faces, process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
            glb = part_out / 'textured.glb'
            textured.export(glb)
            restored = trimesh.load(glb, force='mesh', process=False)
            if not np.allclose(canonical(restored.vertices @ rotation.T, restored.faces),
                               canonical(vertices, faces), rtol=0, atol=1e-7):
                raise RuntimeError('GLB roundtrip changed triangle geometry')
            scene.add_geometry(textured, node_name=str(i), geom_name=str(i))
            part_artifacts = {str(p.relative_to(out)): sha(p) for p in [glb, part_out / 'base_color.png']}
            record = {'index': i, 'part': name, 'signature': signature,
                      'source_faces': len(faces), 'textured_faces': len(uv_faces),
                      'source_vertices': len(vertices), 'uv_vertices': len(uv_vertices),
                      'triangle_geometry_equivalent': True, 'glb_roundtrip_geometry_equivalent': True,
                      'artifacts': part_artifacts, 'seconds': time.time() - part_started}
            write_json(audit, record)
            artifacts.update(part_artifacts)
            status['parts'].append(record)
            write_json(status_path, status)
            del observations, masks, extrinsics, intrinsics, texture
            gc.collect()
            torch.cuda.empty_cache()
            print(f'TEXTURE_PART_COMPLETE {i}', flush=True)
        assembly = out / 'assembly_textured.glb'
        scene.export(assembly)
        artifacts[assembly.name] = sha(assembly)
        if sha(latent) != signature['latent_sha256']:
            raise RuntimeError('Source latent changed during export')
        status.update(status='complete', artifacts=artifacts, seconds=time.time() - started,
                      max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        status.update(status='failed', seconds=time.time() - started, traceback=traceback.format_exc())
        raise
    finally:
        write_json(status_path, status)
    print(f'TEXTURE_COMPLETE {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--texture-size', type=int, choices=[512, 1024, 2048], default=1024)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    export(args.source, args.out, args.texture_size, args.resume)


if __name__ == '__main__':
    main()
