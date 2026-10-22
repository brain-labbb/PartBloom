import copy
import json

import pytest

from protocols import get_adapter


def test_tool_roundtrip_preserves_reasoning_content():
    adapter = get_adapter('chat_completions')
    message = {'role': 'assistant', 'content': None, 'reasoning_content': 'opaque reasoning',
               'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
                   'name': 'build', 'arguments': '{"x": 2}'}},
                   {'id': 'call_2', 'type': 'function', 'function': {'name': 'build', 'arguments': '{}'}}]}
    raw = {'model': 'returned', 'choices': [{'finish_reason': 'tool_calls', 'message': message}], 'usage': {'total_tokens': 42}}
    response = adapter.normalize(raw, 'requested', 't1')
    inputs = [{'role': 'developer', 'content': 'System'}, {'role': 'user', 'content': [
        {'type': 'input_text', 'text': 'Build'}, {'type': 'input_image', 'image_url': 'data:image/png;base64,aGVsbG8='}]}]
    inputs.extend(response['output'])
    inputs.extend([{'type': 'function_call_output', 'call_id': c['id'], 'output': '{"ok":true}'} for c in message['tool_calls']])
    original = copy.deepcopy(inputs)
    tools = [{'type': 'function', 'name': 'build', 'parameters': {'type': 'object'}}]
    payload = adapter.payload(inputs, tools, 'requested', 'medium')
    assert inputs == original
    assert payload['messages'][0] == {'role': 'system', 'content': 'System'}
    assert payload['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png')
    assert payload['messages'][2] == message
    assert len(payload['messages']) == 5
    assert payload['messages'][3] == {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok":true}'}
    assert payload['tools'] == [{'type': 'function', 'function': {'name': 'build', 'parameters': {'type': 'object'}}}]
    assert payload['reasoning_effort'] == 'medium'
    assert payload['max_tokens'] == 16384
    assert response['provider_model'] == 'returned'
    assert response['model'] == 'requested'
    assert response['output'][1]['arguments'] == '{"x": 2}'
    assert response['usage'] == raw['usage']


@pytest.mark.parametrize('finish', ['length', 'content_filter', None])
def test_reject_truncated_tool_arguments(finish):
    raw = {'choices': [{'finish_reason': finish, 'message': {'role': 'assistant', 'tool_calls': [
        {'id': 'a', 'type': 'function', 'function': {'name': 'build', 'arguments': '{"x":'}}]}}]}
    with pytest.raises(RuntimeError, match='completion'):
        get_adapter('chat_completions').normalize(raw, 'm', 't')


def test_text_completion_and_empty_response():
    adapter = get_adapter('chat_completions')
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'done'}}]}
    assert adapter.normalize(raw, 'm', 't')['output'][0]['content'] == [{'type': 'output_text', 'text': 'done'}]
    with pytest.raises(RuntimeError, match='no choices'):
        adapter.normalize({}, 'm', 't')
    raw['choices'][0]['message']['content'] = None
    with pytest.raises(RuntimeError, match='empty content'):
        adapter.normalize(raw, 'm', 't')


@pytest.mark.parametrize('base', ['https://example.test', 'https://example.test/v1'])
def test_endpoint(tmp_path, base):
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def json(self): return {'choices': []}
    class Session:
        def post(self, url, **kwargs):
            assert url == 'https://example.test/v1/chat/completions'
            assert kwargs['headers'] == {'Authorization': 'Bearer secret'}
            return Response()
    trace = tmp_path / 'trace.json'
    raw = get_adapter('chat_completions').send(Session(), base, 'secret', 'm', {}, trace)
    assert json.loads(trace.read_text()) == raw
