"""Chat Completions adapter with reasoning_content and tool-call replay."""
import copy
import json


def content_blocks(content):
    if isinstance(content, str):
        return content
    blocks = []
    for value in content:
        if value['type'] in ('input_text', 'output_text'):
            blocks.append({'type': 'text', 'text': value['text']})
        elif value['type'] == 'input_image':
            blocks.append({'type': 'image_url', 'image_url': {'url': value['image_url']}})
        else:
            raise ValueError('Unsupported input content ' + value['type'])
    return blocks


def make_payload(inputs, tools=None):
    messages = []
    for item in inputs:
        kind = item.get('type')
        if kind == 'function_call':
            if item.get('chat_replayed'):
                continue
            call = {'id': item['call_id'], 'type': 'function',
                    'function': {'name': item['name'], 'arguments': item['arguments']}}
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1].setdefault('tool_calls', []).append(call)
            else:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [call]})
        elif kind == 'function_call_output':
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': item['output']})
        elif kind == 'reasoning':
            continue
        elif 'chat_message' in item:
            messages.append(copy.deepcopy(item['chat_message']))
        else:
            role = 'system' if item['role'] == 'developer' else item['role']
            messages.append({'role': role, 'content': content_blocks(item['content'])})
    payload = {'messages': messages}
    if tools:
        payload['tools'] = [{'type': 'function', 'function': {
            key: value for key, value in tool.items() if key in ('name', 'description', 'parameters')}}
            for tool in tools]
    return payload


def normalize_response(raw, model, label):
    if raw.get('error'):
        raise RuntimeError('Chat Completions error: ' + json.dumps(raw['error']))
    choices = raw.get('choices', [])
    if not choices:
        raise RuntimeError('Chat Completions returned no choices')
    choice = choices[0]
    if choice.get('finish_reason') not in ('stop', 'tool_calls'):
        raise RuntimeError('Chat Completions completion: ' + str(choice.get('finish_reason')))
    message = choice.get('message', {})
    calls = message.get('tool_calls') or []
    content = message.get('content')
    if not content and not calls:
        raise RuntimeError('Chat Completions returned empty content')
    if choice['finish_reason'] == 'tool_calls' and not calls:
        raise RuntimeError('Chat Completions tool_calls completion without a tool call')
    # Preserve opaque reasoning and complete tool calls; omit response-only metadata.
    replay = {key: copy.deepcopy(value) for key, value in message.items()
              if key in ('role', 'content', 'reasoning_content', 'tool_calls')}
    replay.setdefault('role', 'assistant')
    if content is None:
        text_content = []
    elif isinstance(content, str):
        text_content = [{'type': 'output_text', 'text': content}]
    else:
        text_content = [{'type': 'output_text', 'text': p['text']} for p in content if p['type'] == 'text']
    output = [{'type': 'message', 'role': 'assistant', 'content': text_content, 'chat_message': replay}]
    for call in calls:
        if call.get('type') != 'function':
            raise RuntimeError('Unsupported Chat Completions tool type')
        function = call['function']
        arguments = function['arguments']
        if not isinstance(json.loads(arguments), dict):
            raise RuntimeError('Chat Completions tool input must be an object')
        output.append({'type': 'function_call', 'call_id': call['id'], 'name': function['name'],
                       'arguments': arguments, 'chat_replayed': True})
    return {'model': model, 'provider_model': raw.get('model'), 'output': output, 'usage': raw.get('usage')}


class Adapter:
    name = 'chat_completions'

    def payload(self, inputs, tools, model, effort):
        payload = make_payload(inputs, tools)
        payload.update(model=model, max_tokens=16384, reasoning_effort=effort, stream=False)
        return payload

    def send(self, session, base, key, model, payload, trace):
        endpoint = base.rstrip('/')
        if not endpoint.endswith('/v1'):
            endpoint += '/v1'
        with session.post(endpoint + '/chat/completions', json=payload,
                          headers={'Authorization': 'Bearer ' + key}, timeout=(30, 300)) as response:
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}: ' + response.text[:1500])
            raw = response.json()
        trace.write_text(json.dumps(raw, indent=2))
        return raw

    def normalize(self, raw, model, label):
        return normalize_response(raw, model, label)
