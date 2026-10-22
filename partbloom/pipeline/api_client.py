"""Unified client; protocol adapters handle wire formats only."""
import base64
import json
import os
from pathlib import Path
import re
import time
import requests
from protocols import get_adapter


def load_connection(path, protocol=None):
    fields = json.loads(Path(path).read_text())
    if not isinstance(fields, dict):
        raise ValueError("API configuration must be an object")
    selected = protocol or fields.get("protocol")
    base = fields.get("base_url")
    key_env = fields.get("api_key_env")
    key = os.environ.get(key_env) if key_env else fields.get("api_key")
    if not base or not key:
        raise ValueError("API config requires base_url and api_key (or api_key_env)")
    get_adapter(selected)
    return selected, base.rstrip("/"), key


def image_content(path):
    return {"type": "input_image", "image_url": "data:image/png;base64," +
            base64.b64encode(Path(path).read_bytes()).decode()}


class Client:
    def __init__(self, config, model, logs, reasoning_effort="high", protocol=None):
        if reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("Unsupported reasoning effort")
        self.protocol, self.base, self.key = load_connection(config, protocol)
        self.adapter = get_adapter(self.protocol)
        if hasattr(self.adapter, 'configure'):
            self.adapter.configure(json.loads(Path(config).read_text()))
        self.model, self.reasoning_effort = model, reasoning_effort
        self.logs = Path(logs)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.calls = []
        # One connection pool per agent; keep payloads and retry policy unchanged.
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def call(self, inputs, label, tools=None):
        payload = self.adapter.payload(inputs, tools, self.model, self.reasoning_effort)
        (self.logs / f"{label}_request.json").write_text(json.dumps(payload))
        start = time.time()
        for attempt in range(3):
            try:
                raw = self.adapter.send(self.session, self.base, self.key, self.model, payload,
                                        self.logs / f"{label}_attempt{attempt}_provider.jsonl")
                response = self.adapter.normalize(raw, self.model, label)
                (self.logs / f"{label}_response.json").write_text(json.dumps(response, indent=2))
                self.calls.append({"label": label, "seconds": time.time() - start,
                                   "attempts": attempt + 1, "usage": response.get("usage"),
                                   "model": self.model, "returned_model": response.get("provider_model", response.get("model")),
                                   "reasoning_effort": self.reasoning_effort, "protocol": self.protocol,
                                   "stream": getattr(self.adapter, "stream", None)})
                (self.logs / "calls.json").write_text(json.dumps(self.calls, indent=2))
                print("API_COMPLETE", label, round(time.time() - start, 1), flush=True)
                return response
            except Exception as exc:
                message = str(exc).replace(self.key, "[REDACTED]")
                (self.logs / f"{label}_attempt{attempt}_error.txt").write_text(message)
                print("API_ERROR", label, message[:200], flush=True)
                if attempt == 2:
                    raise RuntimeError(message) from None
                time.sleep(5 * (attempt + 1))


def response_text(response):
    return "\n".join(c["text"] for o in response["output"] if o["type"] == "message"
                     for c in o["content"] if c["type"] == "output_text")


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    return json.loads(text)
