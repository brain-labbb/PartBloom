"""Anthropic Messages adapter with lossless thinking and tool-use replay."""
import copy
import json


def content_blocks(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    blocks = []
    for value in content:
        if value['type'] in ('input_text', 'output_text'):
            blocks.append({'type': 'text', 'text': value['text']})
        elif value['type'] == 'input_image':
            url = value['image_url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ValueError('Anthropic adapter requires base64 data image URLs')
            header, data = url.split(',', 1)
            blocks.append({'type': 'image', 'source': {'type': 'base64',
                'media_type': header[5:].split(';')[0], 'data': data}})
        else:
            raise ValueError('Unsupported input content ' + value['type'])
    return blocks


def make_payload(inputs, tools=None):
    messages, system = [], []
    for item in inputs:
        kind = item.get('type')
        if kind == 'function_call':
            if item.get('anthropic_replayed'):
                continue
            role, blocks = 'assistant', [{'type': 'tool_use', 'id': item['call_id'],
                'name': item['name'], 'input': json.loads(item['arguments'])}]
        elif kind == 'function_call_output':
            role, blocks = 'user', [{'type': 'tool_result', 'tool_use_id': item['call_id'],
                'content': item['output']}]
        elif kind == 'reasoning':
            continue
        elif item.get('role') in ('system', 'developer'):
            system.extend(content_blocks(item['content']))
            continue
        else:
            role = item['role']
            blocks = (copy.deepcopy(item['anthropic_parts']) if 'anthropic_parts' in item
                      else content_blocks(item['content']))
        if not blocks:
            continue
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'].extend(blocks)
        else:
            messages.append({'role': role, 'content': blocks})
    payload = {'messages': messages}
    if system:
        payload['system'] = system
    if tools:
        payload['tools'] = [{'name': t['name'], 'description': t.get('description', ''),
                             'input_schema': t['parameters']} for t in tools]
    return payload


def normalize_response(raw, model, label):
    if raw.get('type') == 'error':
        raise RuntimeError('Anthropic error: ' + json.dumps(raw))
    if raw.get('stop_reason') not in ('end_turn', 'tool_use', 'stop_sequence'):
        raise RuntimeError('Anthropic completion: ' + str(raw.get('stop_reason')))
    parts = raw.get('content', [])
    if not parts:
        raise RuntimeError('Anthropic returned empty content')
    output = [{'type': 'message', 'role': 'assistant', 'anthropic_parts': copy.deepcopy(parts),
        'content': [{'type': 'output_text', 'text': p['text']} for p in parts if p['type'] == 'text']}]
    for part in parts:
        if part['type'] == 'tool_use':
            if not isinstance(part.get('input'), dict):
                raise RuntimeError('Anthropic tool input must be an object')
            output.append({'type': 'function_call', 'call_id': part['id'], 'name': part['name'],
                           'arguments': json.dumps(part['input']), 'anthropic_replayed': True})
    if raw['stop_reason'] == 'tool_use' and len(output) == 1:
        raise RuntimeError('Anthropic tool_use completion without a tool call')
    return {'model': model, 'provider_model': raw.get('model'), 'output': output,
            'usage': raw.get('usage')}


class Adapter:
    name = 'anthropic'

    def payload(self, inputs, tools, model, effort):
        payload = make_payload(inputs, tools)
        payload.update(model=model, max_tokens=16384, thinking={'type': 'adaptive'},
                       output_config={'effort': effort})
        return payload

    def send(self, session, base, key, model, payload, trace):
        endpoint = base.rstrip('/')
        if not endpoint.endswith('/v1'):
            endpoint += '/v1'
        with session.post(endpoint + '/messages', json=payload,
                          headers={'x-api-key': key, 'Authorization': 'Bearer ' + key,
                                   'anthropic-version': '2023-06-01'},
                          timeout=(30, 300)) as response:
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}: ' + response.text[:1500])
            raw = response.json()
        trace.write_text(json.dumps(raw, indent=2))
        return raw

    def normalize(self, raw, model, label):
        return normalize_response(raw, model, label)
