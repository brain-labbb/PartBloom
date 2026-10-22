import copy
import json

import pytest

from protocols import get_adapter
from protocols.anthropic import normalize_response


def test_full_tool_round_trip_with_thinking_and_images():
    adapter = get_adapter('anthropic')
    inputs = [{'role': 'system', 'content': 'System'},
              {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Build'},
               {'type': 'input_image', 'image_url': 'data:image/webp;base64,aGVsbG8='}]}]
    tools = [{'type': 'function', 'name': 'build', 'description': 'Build geometry',
              'parameters': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}}]
    parts = [{'type': 'thinking', 'thinking': 'Plan', 'signature': 'opaque-signature'},
             {'type': 'redacted_thinking', 'data': 'opaque-data'},
             {'type': 'text', 'text': 'Building'},
             {'type': 'tool_use', 'id': 'toolu_1', 'name': 'build', 'input': {'x': 2}},
             {'type': 'tool_use', 'id': 'toolu_2', 'name': 'build', 'input': {'x': 3}}]
    raw = {'type': 'message', 'model': 'provider-model', 'stop_reason': 'tool_use', 'content': parts}
    response = adapter.normalize(raw, 'requested-model', 'turn1')
    inputs.extend(response['output'])
    for call_id in ('toolu_1', 'toolu_2'):
        inputs.append({'type': 'function_call_output', 'call_id': call_id, 'output': '{"ok": true}'})
    inputs.append({'role': 'user', 'content': 'Continue'})
    original = copy.deepcopy(inputs)
    payload = adapter.payload(inputs, tools, 'requested-model', 'medium')
    assert inputs == original
    assert payload['thinking'] == {'type': 'adaptive'}
    assert payload['output_config'] == {'effort': 'medium'}
    assert payload['max_tokens'] == 16384
    assert payload['system'] == [{'type': 'text', 'text': 'System'}]
    assert payload['tools'][0]['input_schema'] == tools[0]['parameters']
    assert payload['messages'][0]['content'][1]['source']['media_type'] == 'image/webp'
    assert payload['messages'][1] == {'role': 'assistant', 'content': parts}
    assert len(payload['messages']) == 3
    assert [p['type'] for p in payload['messages'][2]['content']] == ['tool_result', 'tool_result', 'text']
    assert payload['messages'][2]['content'][0]['tool_use_id'] == 'toolu_1'
    assert response['provider_model'] == 'provider-model'
    assert json.loads(response['output'][1]['arguments']) == {'x': 2}


@pytest.mark.parametrize('stop', ['max_tokens', 'pause_turn', 'refusal', None])
def test_reject_incomplete_response_even_if_tool_is_present(stop):
    with pytest.raises(RuntimeError, match='Anthropic completion'):
        normalize_response({'stop_reason': stop, 'content': [
            {'type': 'tool_use', 'id': 'a', 'name': 'build', 'input': {}}]}, 'm', 't')


def test_reject_empty_and_missing_tool():
    with pytest.raises(RuntimeError, match='empty content'):
        normalize_response({'stop_reason': 'end_turn', 'content': []}, 'm', 't')
    with pytest.raises(RuntimeError, match='without a tool call'):
        normalize_response({'stop_reason': 'tool_use', 'content': [{'type': 'text', 'text': 'Oops'}]}, 'm', 't')


@pytest.mark.parametrize('base', ['https://example.test', 'https://example.test/v1'])
def test_messages_endpoint_and_auth(tmp_path, base):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def json(self): return {'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn'}
    class Session:
        def post(self, endpoint, **kwargs):
            assert endpoint == 'https://example.test/v1/messages'
            assert kwargs['headers'] == {'x-api-key': 'secret', 'Authorization': 'Bearer secret', 'anthropic-version': '2023-06-01'}
            return Response()
    trace = tmp_path / 'trace.json'
    raw = get_adapter('anthropic').send(Session(), base, 'secret', 'm', {}, trace)
    assert json.loads(trace.read_text()) == raw
