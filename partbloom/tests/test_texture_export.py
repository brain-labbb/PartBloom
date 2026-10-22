import json
from pathlib import Path
import numpy as np
import pytest
import resources
from texture_export import artifacts_valid, canonical, export, resolve_source, sha


def source_fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'shared_slat.pt').write_bytes(b'frozen latent')
    (source / 'part_index.json').write_text('[{"index":0,"part":"body"}]')
    np.save(source / 'allind.npy', [[1, 2, 3]])
    np.save(source / 'part_labels.npy', [0])
    (source / 'objs/0').mkdir(parents=True)
    (source / 'objs/0/0.obj').write_text('v 0 0 0\n')
    return source


def test_source_supports_cached_decode_and_rejects_changed_latent(tmp_path):
    source = source_fixture(tmp_path)
    latent = source / 'shared_slat.pt'
    moved = tmp_path / 'saved.pt'
    latent.rename(moved)
    (source / 'experiment.json').write_text(json.dumps({'latent_source': str(moved), 'latent_sha256': sha(moved)}))
    assert resolve_source(source)[1] == moved
    moved.write_bytes(b'different')
    with pytest.raises(ValueError, match='hash differs'):
        resolve_source(source)


def test_artifact_integrity_and_triangle_geometry(tmp_path):
    path = tmp_path / 'part.glb'
    path.write_bytes(b'valid')
    record = {'artifacts': {'part.glb': sha(path)}}
    assert artifacts_valid(tmp_path, record)
    path.write_bytes(b'corrupt')
    assert not artifacts_valid(tmp_path, record)
    path.unlink()
    assert not artifacts_valid(tmp_path, record)
    v = np.array([[0,0,0], [1,0,0], [0,1,0]], np.float32)
    assert np.array_equal(canonical(v, [[0,1,2]]), canonical(v.astype(float), [[2,0,1]]))
    assert not np.array_equal(canonical(v, [[0,1,2]]), canonical(v + 1, [[0,1,2]]))


def test_resume_checks_source_before_gpu_loading(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    weights = tmp_path / 'weights'
    config = weights / 'TRELLIS-image-large/pipeline.json'
    config.parent.mkdir(parents=True)
    config.write_text('{"args":{"models":{"slat_decoder_gs":"ckpts/gs"}}}')
    monkeypatch.setattr(resources, 'WEIGHTS_ROOT', weights)
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'texture_status.json').write_text('{"signature":{},"status":"complete"}')
    with pytest.raises(ValueError, match='differ'):
        export(source, out, resume=True)
    with pytest.raises(FileExistsError):
        export(source, out)
