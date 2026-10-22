from protocols.generate_content import make_payload, normalize_response


def test_vision_and_signed_tool_continuation():
    raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'functionCall': {'name': 'paint', 'args': {'x': 4}}, 'thoughtSignature': 'opaque-signature'}]}}]}
    result = normalize_response(raw, 'gemini-4-argon-medium', 'step0')
    inputs = [{'role': 'developer', 'content': 'Build voxels.'}, {'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'Target'},
        {'type': 'input_image', 'image_url': 'data:image/png;base64,YWJj'}]}]
    inputs += result['output']
    inputs += [{'type': 'function_call_output', 'call_id': 'step0_0', 'output': '{"ok":true}'}]
    payload = make_payload(inputs)
    assert payload['contents'][0]['parts'][1]['inlineData'] == {'mimeType': 'image/png', 'data': 'YWJj'}
    assert payload['contents'][1]['parts'] == raw['candidates'][0]['content']['parts']
    assert payload['contents'][2]['parts'] == [{'functionResponse': {'name': 'paint', 'response': {'ok': True}}}]
    assert payload['systemInstruction']['parts'][0]['text'] == 'Build voxels.'


import json
import pytest
from api_client import Client, response_text
from protocols.generate_content import collect_stream, sse_events


def frames(events):
    yield b': heartbeat'
    yield b''
    for event in events:
        yield ('data: ' + json.dumps(event)).encode()
        yield b''


def test_stream_text_json_and_empty_signature_replay(tmp_path):
    parts = [{'text': '{"na'}, {'text': 'me":"pig"}'}, {'text': '', 'thoughtSignature': 'last-signature'}]
    events = [{'candidates': [{'index': 0, 'content': {'parts': [p]}}]} for p in parts]
    events += [{'candidates': [{'index': 0, 'finishReason': 'STOP'}]},
               {'usageMetadata': {'totalTokenCount': 15}, 'modelVersion': 'provider-model'}]
    raw = collect_stream(sse_events(frames(events)), tmp_path / 'stream.jsonl')
    response = normalize_response(raw, 'requested-model', 'step')
    assert json.loads(response_text(response)) == {'name': 'pig'}
    assert response['usage']['totalTokenCount'] == 15
    assert response['provider_model'] == 'provider-model'
    payload = make_payload(response['output'])
    assert payload['contents'][0]['parts'] == parts
    assert len((tmp_path / 'stream.jsonl').read_text().splitlines()) == len(events)


def test_stream_client_config_and_signed_tool_replay(tmp_path, monkeypatch):
    config = tmp_path / 'api.json'
    config.write_text(json.dumps({'protocol': 'generate_content', 'base_url': 'https://test/v1beta',
                                  'api_key': 'secret', 'stream': True}))
    part = {'functionCall': {'name': 'paint', 'args': {'x': 4}}, 'thoughtSignature': 'signed-call'}
    seen = {}
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            return frames([{'candidates': [{'content': {'parts': [part]}}]},
                           {'candidates': [{'finishReason': 'STOP'}]}])
    def post(self, url, **kwargs):
        seen.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr('api_client.requests.Session.post', post)
    client = Client(config, 'gemini-model', tmp_path / 'logs', reasoning_effort='medium')
    response = client.call([{'role': 'user', 'content': 'Paint'}], 'tool')
    assert seen['url'] == 'https://test/v1beta/models/gemini-model:streamGenerateContent?alt=sse'
    assert seen['stream'] is True and 'stream' not in seen['json']
    assert client.calls[0]['stream'] is True
    call = response['output'][1]
    continued = make_payload(response['output'] + [{'type': 'function_call_output',
        'call_id': call['call_id'], 'output': '{"ok":true}'}])
    assert continued['contents'][0]['parts'] == [part]
    assert continued['contents'][1]['parts'][0]['functionResponse']['name'] == 'paint'


@pytest.mark.parametrize('events', [[], [{'candidates': [{'content': {'parts': [{'text': 'unfinished'}]}}]}]])
def test_stream_rejects_truncation(tmp_path, events):
    with pytest.raises(RuntimeError, match='finishReason'):
        collect_stream(sse_events(frames(events)), tmp_path / 'trace')


def test_stream_error_and_partial_tool_are_not_executed(tmp_path):
    with pytest.raises(RuntimeError, match='stream error'):
        collect_stream(iter([{'error': {'message': 'failure'}}]), tmp_path / 'error')
    with pytest.raises(RuntimeError, match='Partial function-call'):
        collect_stream(iter([{'candidates': [{'content': {'parts': [
            {'functionCall': {'name': 'paint', 'partialArgs': []}}]}}]}]), tmp_path / 'partial')
    raw = collect_stream(iter([{'candidates': [{'content': {'parts': [{'text': 'partial'}]},
                                               'finishReason': 'MAX_TOKENS'}]}]), tmp_path / 'limited')
    with pytest.raises(RuntimeError, match='MAX_TOKENS'):
        normalize_response(raw, 'model', 'step')


def test_sse_multiline_and_final_frame():
    assert list(sse_events([b': keepalive', b'event: message', b'data: {', b'data: "a": 1}', b'',
                            b'data: {"b":2}'])) == [{'a': 1}, {'b': 2}]
    assert list(sse_events([b'data: [DONE]', b''])) == []


def test_stream_config_rejects_string_boolean(tmp_path):
    config = tmp_path / 'api.json'
    config.write_text(json.dumps({'protocol': 'generate_content', 'base_url': 'https://test',
                                 'api_key': 'secret', 'stream': 'true'}))
    with pytest.raises(ValueError, match='boolean'):
        Client(config, 'model', tmp_path / 'logs')
