import json
import pytest
from api_client import Client, load_connection


@pytest.mark.parametrize('protocol', ['responses', 'generate_content'])
def test_unified_client_uses_protocol_not_model(tmp_path, monkeypatch, protocol):
    config = tmp_path / 'api.json'
    config.write_text(json.dumps({'protocol': protocol, 'base_url': 'https://gateway.test/v1beta' if protocol == 'generate_content' else 'https://gateway.test/v1', 'api_key': 'secret'}))
    seen = {}
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_lines(self):
            yield b'data: ' + json.dumps({'type': 'response.completed', 'response': {'model': 'custom-model', 'output': [{'type': 'function_call', 'call_id': 'a', 'name': 'paint', 'arguments': '{"x":4}'}], 'usage': {'total_tokens': 12}}}).encode()
        def json(self):
            return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'functionCall': {'name': 'paint', 'args': {'x': 4}}, 'thoughtSignature': 'signature'}]}}], 'usageMetadata': {'totalTokenCount': 12}}
    def post(session, url, **kwargs):
        seen.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr('api_client.requests.Session.post', post)
    client = Client(config, 'custom-model', tmp_path / 'logs')
    result = client.call([{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,YWJj'}]}], 'step')
    call = next(item for item in result['output'] if item['type'] == 'function_call')
    assert call['name'] == 'paint' and json.loads(call['arguments']) == {'x': 4}
    assert client.calls[0]['protocol'] == protocol
    assert seen['url'].endswith('/responses' if protocol == 'responses' else '/v1beta/models/custom-model:generateContent')
    assert '/v1beta/v1beta' not in seen['url']
    if protocol == 'generate_content':
        assert seen['json']['generationConfig']['thinkingConfig']['thinkingLevel'] == 'high'
    assert 'secret' not in ''.join(p.read_text() for p in (tmp_path / 'logs').iterdir())


def test_config_explicit_protocol_and_override(tmp_path, monkeypatch):
    config = tmp_path / 'api.txt'
    config.write_text('base_url = "https://gateway.test/v1"\n"OPENAI_API_KEY": "secret"')
    with pytest.raises(json.JSONDecodeError):
        load_connection(config)
    config.write_text('GOOGLE_GEMINI_BASE_URL = "https://gateway.test"\nGEMINI_API_KEY = "secret"')
    with pytest.raises(json.JSONDecodeError):
        load_connection(config)
    monkeypatch.setenv('MY_SERVICE_KEY', 'env-secret')
    config.write_text(json.dumps({'protocol': 'responses', 'base_url': 'https://other.test/v1', 'api_key_env': 'MY_SERVICE_KEY'}))
    assert load_connection(config)[2] == 'env-secret'
    assert load_connection(config, 'generate_content')[0] == 'generate_content'
    with pytest.raises(ValueError, match='Unsupported protocol'):
        load_connection(config, 'unknown')
    config.write_text(json.dumps({'base_url': 'https://other.test/v1', 'api_key': 'secret'}))
    with pytest.raises(ValueError, match='Unsupported protocol'):
        load_connection(config)


def test_retry_errors_redact_credentials(tmp_path, monkeypatch):
    config = tmp_path / 'api.json'
    config.write_text(json.dumps({'protocol': 'responses', 'base_url': 'https://gateway.test', 'api_key': 'sensitive-key'}))
    def fail(*args, **kwargs):
        raise RuntimeError('failure sensitive-key')
    monkeypatch.setattr('api_client.requests.Session.post', fail)
    monkeypatch.setattr('api_client.time.sleep', lambda _: None)
    client = Client(config, 'custom-model', tmp_path / 'logs')
    with pytest.raises(RuntimeError, match='REDACTED'):
        client.call([], 'step')
    assert len(list((tmp_path / 'logs').glob('*_error.txt'))) == 3
    assert 'sensitive-key' not in ''.join(p.read_text() for p in (tmp_path / 'logs').iterdir())
