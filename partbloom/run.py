"""Unified entry point; paths supplied on the CLI are relative to the caller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent


def load_config(path=None):
    config = json.loads((PACKAGE / 'configs/default.json').read_text())
    if path:
        config.update(json.loads(Path(path).read_text()))
    for key in ('max_api_steps', 'repair_max_api_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('python_gpu', 'vendor', 'weights', 'blender'):
        value = Path(config[key]).expanduser()
        if key == 'blender' and len(value.parts) == 1:
            continue  # a bare command name is looked up on PATH
        # Preserve the virtualenv executable path: resolving its symlink bypasses
        # pyvenv.cfg and silently selects the parent interpreter's packages.
        config[key] = str((PROJECT / value).absolute() if not value.is_absolute() else value)
    return config


def environment(config, gpu=None):
    env = os.environ.copy()
    env.update(PARTBLOOM_VENDOR=config['vendor'], PARTBLOOM_WEIGHTS=config['weights'],
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
               PYTHONPATH=os.pathsep.join([str(PACKAGE), str(PACKAGE / 'pipeline'), env.get('PYTHONPATH', '')]))
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    return env


def invoke(script, args, config, env, gpu=False):
    interpreter = config['python_gpu'] if gpu else sys.executable
    subprocess.run([interpreter, str(PACKAGE / 'pipeline' / script), *map(str, args)],
                   env=env, check=True)


def completed(folder, stage):
    status = folder / f'{stage}_status.json'
    if not status.is_file() or json.loads(status.read_text()).get('status') != 'complete':
        return False
    if stage == 'agent':
        return all((folder / name).is_file() for name in
                   ('allind.npy', 'part_index.json', 'cond_img.png', 'dense_parts.npz'))
    if stage == 'texture':
        from pipeline.texture_export import artifacts_valid
        return artifacts_valid(folder, json.loads(status.read_text()))
    return bool(list((folder / 'objs').glob('*/*.obj')))


def fresh(folder):
    if folder.exists():
        raise FileExistsError(f'Output already exists: {folder}. Choose a new folder.')


def coarse(image, out, api_config, config, env):
    fresh(out)
    protocol_args = ['--protocol', config['protocol']] if config.get('protocol') else []
    invoke('agent_runner.py', ['--image', image, '--out', out, '--api-config', api_config,
           '--model', config['model'], '--reasoning-effort', config['reasoning_effort'],
           '--repairs', config['repairs'], '--max-api-steps', config['max_api_steps'],
           '--repair-max-api-steps', config['repair_max_api_steps'], *protocol_args], config, env)


def refine(source, out, config, env):
    fresh(out)
    repair_args = ['--restore-missing-parts'] if config.get('restore_missing_parts', False) else []
    invoke('guided_part_pipeline.py', ['--source', source, '--out', out,
           '--decode-mode', config['decode_mode'], '--noise-time', config['noise_time'],
           '--steps', config['steps'], '--mesh-inner-visibility', config['mesh_inner_visibility'], *repair_args],
           config, env, gpu=True)


def render(source, out, config, env, view):
    fresh(out)
    subprocess.run([config['blender'], '--background', '--threads', '4', '--python',
                    str(PACKAGE / 'tools/render_mesh.py'), '--', '--input', str(source),
                    '--output', str(out), '--view', view, '--part-colors'], env=env, check=True)


def texture(source, out, config, env, resume=False):
    args = ['--source', source, '--out', out, '--texture-size', config['texture_size']]
    if resume:
        args.append('--resume')
    invoke('texture_export.py', args, config, env, gpu=True)
    if not (out / 'preview.png').exists():
        subprocess.run([config['blender'], '--background', '--threads', '4', '--python',
                        str(PACKAGE / 'tools/render_textured.py'), '--',
                        '--input', str(out / 'assembly_textured.glb'),
                        '--output', str(out / 'preview.png')], env=env, check=True)


def resume_manifest_matches(saved, current):
    # Older geometry-only runs predate restored native texture export.
    saved = dict(saved)
    saved['config'] = dict(saved['config'])
    for name in ('texture', 'texture_size'):
        saved['config'].setdefault(name, current['config'][name])
    return saved == current


def main(argv=None):
    parser = argparse.ArgumentParser(description='Image → Agent voxel parts → guided TRELLIS1 → part meshes')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'coarse', 'refine', 'redecode', 'texture', 'render', 'doctor'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, help='JSON overrides; resource paths relative to project root')
        p.add_argument('--gpu', help='CUDA device index or UUID; otherwise inherit CUDA_VISIBLE_DEVICES')
        if name != 'doctor':
            p.add_argument('--out', required=True, type=Path)
        if name in ('run', 'coarse'):
            p.add_argument('--image', required=True, type=Path)
            p.add_argument('--api-config', required=True, type=Path)
            p.add_argument('--model')
            p.add_argument('--protocol', choices=['responses', 'generate_content', 'anthropic', 'chat_completions'])
            p.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'])
        if name in ('refine', 'redecode', 'texture', 'render'):
            p.add_argument('--source', required=True, type=Path)
        if name in ('run', 'refine', 'redecode'):
            p.add_argument('--decode-mode', choices=['parts', 'joint'])
        if name in ('run', 'refine', 'redecode'):
            p.add_argument('--texture', action=argparse.BooleanOptionalAction, default=None,
                           help='Restore TRELLIS native texture export (default: enabled)')
        if name in ('run', 'refine', 'redecode', 'texture'):
            p.add_argument('--texture-size', type=int, choices=[512, 1024, 2048])
        if name in ('run', 'texture'):
            p.add_argument('--resume', action='store_true', help='Skip completed stages of this identical run')
        if name == 'render':
            p.add_argument('--view', choices=['oblique', 'front', 'side', 'top'], default='oblique')
    a = parser.parse_args(argv)
    config = load_config(a.config)
    for name in ('model', 'reasoning_effort', 'decode_mode', 'protocol', 'texture', 'texture_size'):
        if getattr(a, name, None) is not None:
            config[name] = getattr(a, name)
    env = environment(config, a.gpu)
    if a.command == 'doctor':
        paths = {key: Path(config[key]) for key in ('python_gpu', 'vendor', 'weights', 'blender')}
        paths.update(trellis=Path(config['vendor']) / 'PhysX-Omni/trellis',
                     decoder=Path(config['vendor']) / 'PhysX-Omni/decoder_each.py',
                     dinov2=Path(config['vendor']) / 'dinov2/hubconf.py',
                     trellis_weights=Path(config['weights']) / 'TRELLIS-image-large/pipeline.json',
                     dino_weights=Path(config['weights']) / 'dinov2/dinov2_vitl14_reg4_pretrain.pth')
        if config['texture'] and paths['trellis_weights'].exists():
            native = json.loads(paths['trellis_weights'].read_text())
            gs = Path(config['weights']) / 'TRELLIS-image-large' / native['args']['models']['slat_decoder_gs']
            paths.update(gaussian_config=gs.with_suffix('.json'), gaussian_weights=gs.with_suffix('.safetensors'))
        if len(paths['blender'].parts) == 1:  # bare command name, found on PATH
            paths['blender'] = Path(shutil.which(paths['blender']) or paths['blender'])
        result = {key: {'path': str(path), 'exists': path.exists()} for key, path in paths.items()}
        print(json.dumps(result, indent=2))
        return 0 if all(item['exists'] for item in result.values()) else 1
    out = a.out.resolve()
    if a.command in ('run', 'coarse'):
        image = a.image.resolve()
        if not image.is_file() or not a.api_config.is_file():
            parser.error('Input image and external API config must exist')
    if a.command == 'coarse':
        coarse(image, out, a.api_config.resolve(), config, env)
    elif a.command == 'refine':
        refine(a.source.resolve(), out, config, env)
        if config['texture']:
            texture(out, out / 'textured', config, env)
    elif a.command == 'redecode':
        fresh(out)
        invoke('prepare_cached_decode.py', ['--source', a.source.resolve(), '--out', out,
               '--mode', config['decode_mode']], config, env)
        experiment_path = out / 'experiment.json'
        experiment = json.loads(experiment_path.read_text())
        experiment['mesh_inner_visibility'] = config['mesh_inner_visibility']
        experiment_path.write_text(json.dumps(experiment, indent=2))
        env['PARTBLOOM_SAVE_MESH_STAGES'] = '1'
        invoke('cached_part_decode.py', ['--out', out, '--mesh-inner-visibility',
               config['mesh_inner_visibility']], config, env, gpu=True)
        if config['texture']:
            texture(out, out / 'textured', config, env)
    elif a.command == 'texture':
        texture(a.source.resolve(), out, config, env, a.resume)
    elif a.command == 'render':
        render(a.source.resolve(), out, config, env, a.view)
    elif a.command == 'run':
        manifest = {'image': str(image), 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                    'config': config, 'prompt_version': 'functional_motion_voxel_tools_v3'}
        if out.exists():
            if not a.resume or not (out / 'run.json').is_file():
                raise FileExistsError('Use a new output folder, or --resume for a previously started run')
            if not resume_manifest_matches(json.loads((out / 'run.json').read_text()), manifest):
                raise ValueError('Resume image/config differs from the saved run')
        else:
            out.mkdir(parents=True)
            (out / 'run.json').write_text(json.dumps(manifest, indent=2))
        if not completed(out / 'coarse', 'agent'):
            # Composite RGBA on white, matching the earlier two-asset experiment.
            from PIL import Image
            rgba = Image.open(image).convert('RGBA')
            Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB').save(out / 'input.png')
            coarse(out / 'input.png', out / 'coarse', a.api_config.resolve(), config, env)
        if not completed(out / 'refined', 'decode'):
            refine(out / 'coarse', out / 'refined', config, env)
        if not (out / 'parts.png').is_file():
            render(out / 'refined/objs', out / 'parts.png', config, env, 'oblique')
        if config['texture']:
            texture(out / 'refined', out / 'textured', config, env, a.resume)
        print(f'Complete: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
