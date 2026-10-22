import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'tools/resume_agent_from_logs.py'
spec=importlib.util.spec_from_file_location('agent_replay',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_replay_requires_identical_request_and_model(tmp_path,monkeypatch):
    config=tmp_path/'api.json'
    config.write_text(json.dumps({'protocol':'responses','base_url':'https://example.test/v1','api_key':'test-key'}))
    source=tmp_path/'original';source.mkdir()
    client=module.replay_client(source)(config,'test-model',tmp_path/'resumed')
    inputs=[{'role':'user','content':'same task'}]
    payload=client.adapter.payload(inputs,None,'test-model','high')
    (source/'step_request.json').write_text(json.dumps(payload))
    response={'model':'test-model','output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':'done'}]}]}
    (source/'step_response.json').write_text(json.dumps(response))
    monkeypatch.setattr('api_client.requests.Session.request',lambda *a,**kw:pytest.fail('Replay made an API call'))
    assert client.call(inputs,'step')==response
    assert client.calls[0]['replayed'] and client.calls[0]['attempts']==0
    with pytest.raises(ValueError,match='request differs'):
        client.call([{'role':'user','content':'changed task'}],'step')
    response['model']='different-model'
    (source/'step_response.json').write_text(json.dumps(response))
    with pytest.raises(ValueError,match='model differs'):
        client.call(inputs,'step')
