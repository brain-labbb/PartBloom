"""Translate the shared voxel-agent conversation to Gemini native tool calls.

Replay the complete provider model parts, including thought signatures, unchanged.
Thinking level is sent explicitly in generationConfig.
"""
import copy
import json


def make_payload(inputs, tools=None):
    contents, system, names = [], [], {}
    for item in inputs:
        kind = item.get('type')
        if kind == 'function_call':
            names[item['call_id']] = item['name']
            if item.get('gemini_replayed'):
                continue
            role, parts = 'model', [{'functionCall': {'name': item['name'],
                'args': json.loads(item['arguments'])}}]
        elif kind == 'function_call_output':
            role, parts = 'user', [{'functionResponse': {'name': names[item['call_id']],
                'response': json.loads(item['output'])}}]
        elif kind == 'reasoning':
            continue
        elif item.get('role') in ('system', 'developer'):
            system.append(item['content'])
            continue
        else:
            role = 'model' if item.get('role') == 'assistant' else 'user'
            if 'gemini_parts' in item:
                parts = copy.deepcopy(item['gemini_parts'])
            else:
                content = item['content']
                if isinstance(content, str):
                    content = [{'type': 'input_text', 'text': content}]
                parts = []
                for value in content:
                    if value['type'] in ('input_text', 'output_text'):
                        parts.append({'text': value['text']})
                    elif value['type'] == 'input_image':
                        header, data = value['image_url'].split(',', 1)
                        parts.append({'inlineData': {'mimeType': header[5:].split(';')[0], 'data': data}})
                    else:
                        raise ValueError('Unsupported input content '+value['type'])
        if contents and contents[-1]['role'] == role:
            contents[-1]['parts'].extend(parts)
        else:
            contents.append({'role': role, 'parts': parts})
    payload = {'contents': contents}
    if system:
        payload['systemInstruction'] = {'parts': [{'text': '\n\n'.join(system)}]}
    if tools:
        payload['tools'] = [{'functionDeclarations': [
            {key: value for key, value in tool.items() if key in ('name', 'description', 'parameters')}
            for tool in tools]}]
    return payload


def sse_events(lines):
    """Decode SSE frames, including comments, multiline data and a final unclosed frame."""
    data = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if not line:
            if data:
                raw = '\n'.join(data)
                data = []
                if raw == '[DONE]':
                    return
                yield json.loads(raw)
        elif line.startswith('data:'):
            value = line[5:]
            data.append(value[1:] if value.startswith(' ') else value)
    if data:
        raw = '\n'.join(data)
        if raw != '[DONE]':
            yield json.loads(raw)


def collect_stream(events, trace):
    # Retain all provider parts unchanged for signed conversation replay.
    # Visible text is joined separately when normalizing token-sized chunks.
    result = {'candidates': [], '_streamed': True}
    candidates = {}
    with trace.open('w') as log:
        for event in events:
            log.write(json.dumps(event) + '\n')
            log.flush()
            if 'error' in event:
                raise RuntimeError('Gemini stream error: ' + json.dumps(event['error']))
            for key, value in event.items():
                if key != 'candidates':
                    result[key] = copy.deepcopy(value)
            for chunk in event.get('candidates', []):
                index = chunk.get('index', 0)
                candidate = candidates.setdefault(index, {'index': index, 'content': {'role': 'model', 'parts': []}})
                for key, value in chunk.items():
                    if key != 'content':
                        candidate[key] = copy.deepcopy(value)
                for part in chunk.get('content', {}).get('parts', []):
                    call = part.get('functionCall', {})
                    if 'partialArgs' in call or call.get('willContinue'):
                        raise RuntimeError('Partial function-call argument streaming is not enabled/supported')
                    candidate['content']['parts'].append(copy.deepcopy(part))
    result['candidates'] = [candidates[i] for i in sorted(candidates)]
    if not result['candidates'] or not result['candidates'][0].get('finishReason'):
        raise RuntimeError('Gemini stream ended without a candidate finishReason')
    return result


def normalize_response(raw, model, label):
    candidates = raw.get('candidates', [])
    if not candidates:
        raise RuntimeError('Gemini returned no candidates: '+json.dumps(raw)[:1000])
    candidate = candidates[0]
    if candidate.get('finishReason') not in (None, 'STOP'):
        raise RuntimeError('Gemini completion: '+str(candidate.get('finishReason')))
    parts = candidate.get('content', {}).get('parts', [])
    if not parts:
        raise RuntimeError('Gemini returned empty content')
    output = [{'type': 'message', 'role': 'assistant', 'gemini_parts': parts,
        'content': [{'type': 'output_text', 'text': p['text']} for p in parts
                    if 'text' in p and not p.get('thought')]}]
    if raw.get('_streamed'):
        output[0]['content'] = [{'type': 'output_text', 'text': ''.join(
            p['text'] for p in parts if 'text' in p and not p.get('thought'))}]
    for i, part in enumerate(parts):
        if 'functionCall' in part:
            call = part['functionCall']
            output.append({'type': 'function_call', 'call_id': f'{label}_{i}',
                'name': call['name'], 'arguments': json.dumps(call.get('args', {})),
                'gemini_replayed': True})
    return {'model': model, 'provider_model': raw.get('modelVersion'), 'output': output,
        'usage': raw.get('usageMetadata')}

class Adapter:
    name = "generate_content"
    stream = False

    def configure(self, fields):
        stream = fields.get('stream', False)
        if not isinstance(stream, bool):
            raise ValueError('Gemini stream must be a JSON boolean')
        self.stream = stream

    def payload(self, inputs, tools, model, effort):
        payload = make_payload(inputs, tools)
        payload['generationConfig'] = {'thinkingConfig': {'thinkingLevel': effort}}
        return payload

    def send(self, session, base, key, model, payload, trace):
        from urllib.parse import quote
        endpoint = base if base.endswith("/v1beta") else base + "/v1beta"
        method = ':streamGenerateContent?alt=sse' if self.stream else ':generateContent'
        options = {'stream': True} if self.stream else {}
        with session.post(endpoint + '/models/' + quote(model, safe='') + method,
                          json=payload, headers={'x-goog-api-key': key},
                          timeout=(30, 300), **options) as response:
            if response.status_code != 200:
                raise RuntimeError(f'HTTP {response.status_code}: ' + response.text[:1500])
            if self.stream:
                raw = collect_stream(sse_events(response.iter_lines()), trace)
            else:
                raw = response.json()
                trace.write_text(json.dumps(raw, indent=2))
        return raw

    def normalize(self, raw, model, label):
        return normalize_response(raw, model, label)
