import json
from pathlib import Path

import pytest
import run


def test_defaults_and_resource_paths():
    config = run.load_config()
    assert config['decode_mode'] == 'parts'
    assert config['noise_time'] == .65 and config['steps'] == 12
    assert all(Path(config[key]).is_absolute() for key in ('vendor', 'weights', 'python_gpu'))
    assert config['python_gpu'].endswith('.venv/bin/python')


def test_redecode_forwards_mode_and_threshold(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    out = tmp_path / 'decoded'
    calls = []

    def invoke(script, args, config, env, gpu=False):
        calls.append((script, args, gpu))
        if script == 'prepare_cached_decode.py':
            out.mkdir()
            (out / 'experiment.json').write_text('{}')

    monkeypatch.setattr(run, 'invoke', invoke)
    config_file = tmp_path / 'config.json'
    config_file.write_text(json.dumps({'mesh_inner_visibility': .005}))
    run.main(['redecode', '--source', str(source), '--out', str(out),
              '--decode-mode', 'joint', '--config', str(config_file), '--no-texture'])
    assert calls[0][1][-1] == 'joint'
    assert calls[1][0] == 'cached_part_decode.py' and calls[1][2]
    assert calls[1][1][-1] == .005
    assert json.loads((out / 'experiment.json').read_text())['mesh_inner_visibility'] == .005
    with pytest.raises(FileExistsError):
        run.main(['redecode', '--source', str(source), '--out', str(out)])


def test_completed_requires_artifacts(tmp_path):
    (tmp_path / 'agent_status.json').write_text('{"status": "complete"}')
    assert not run.completed(tmp_path, 'agent')
    for name in ('allind.npy', 'part_index.json', 'cond_img.png', 'dense_parts.npz'):
        (tmp_path / name).touch()
    assert run.completed(tmp_path, 'agent')


def test_resume_skips_finished_stages_and_rejects_changed_config(tmp_path, monkeypatch):
    image = tmp_path / 'image.png'
    image.write_bytes(b'cached input; no image decoding should run')
    credentials = tmp_path / 'external.txt'
    credentials.touch()
    out = tmp_path / 'run'
    out.mkdir()
    manifest = {'image': str(image), 'image_sha256': run.hashlib.sha256(image.read_bytes()).hexdigest(),
                'config': run.load_config(), 'prompt_version': 'functional_motion_voxel_tools_v3'}
    (out / 'run.json').write_text(json.dumps(manifest))
    (out / 'parts.png').touch()
    monkeypatch.setattr(run, 'completed', lambda *args: True)
    monkeypatch.setattr(run, 'invoke', lambda *args, **kwargs: pytest.fail('Completed geometry stage repeated'))
    texture_calls = []
    monkeypatch.setattr(run, 'texture', lambda *args: texture_calls.append(args))
    args = ['run', '--image', str(image), '--api-config', str(credentials), '--out', str(out), '--resume']
    assert run.main(args) == 0
    assert texture_calls[0][0] == out / 'refined' and texture_calls[0][-1] is True
    with pytest.raises(ValueError, match='differs'):
        run.main([*args, '--decode-mode', 'joint'])


def test_native_texture_is_default_and_can_be_disabled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, 'refine', lambda *args: None)
    monkeypatch.setattr(run, 'texture', lambda *args: calls.append(args))
    args = ['refine', '--source', str(tmp_path / 'source'), '--out', str(tmp_path / 'output')]
    run.main(args)
    assert len(calls) == 1 and calls[0][1] == tmp_path / 'output/textured'
    run.main([*args, '--no-texture'])
    assert len(calls) == 1


def test_legacy_manifest_can_restore_textures_without_repeating_geometry():
    current = {'config': run.load_config(), 'image': 'unchanged'}
    old = {'config': {k: v for k, v in current['config'].items() if k not in ('texture', 'texture_size')},
           'image': 'unchanged'}
    assert run.resume_manifest_matches(old, current)
    old['config']['steps'] = 99
    assert not run.resume_manifest_matches(old, current)
