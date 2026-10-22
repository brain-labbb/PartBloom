"""Responses wire protocol; shared by all compatible services."""
import json


class Adapter:
    name = "responses"

    def payload(self, inputs, tools, model, effort):
        payload = {"model": model, "reasoning": {"effort": effort}, "input": inputs,
                   "store": False, "stream": True, "include": ["reasoning.encrypted_content"]}
        if tools:
            payload.update(tools=tools, parallel_tool_calls=False)
        return payload

    def send(self, session, base, key, model, payload, trace):
        result = None
        with session.post(base + "/responses", json=payload,
                          headers={"Authorization": "Bearer " + key},
                          stream=True, timeout=(30, 300)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: " + response.text[:1500])
            with trace.open("w") as log:
                for line in response.iter_lines():
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        continue
                    event = json.loads(raw)
                    log.write(json.dumps(event) + "\n")
                    log.flush()
                    if event.get("type") == "response.completed":
                        result = event["response"]
                    if event.get("type") in ("error", "response.failed", "response.incomplete"):
                        raise RuntimeError(json.dumps(event))
        if result is None:
            raise RuntimeError("Stream ended without response.completed")
        return result

    def normalize(self, raw, model, label):
        if raw.get("model") != model:
            raise RuntimeError("Requested model mismatch: " + str(raw.get("model")))
        return raw
