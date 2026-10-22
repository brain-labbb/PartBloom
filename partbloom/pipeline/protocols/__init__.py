"""Select adapters by wire protocol, never by model name or service address."""
from .responses import Adapter as Responses
from .generate_content import Adapter as GenerateContent
from .anthropic import Adapter as Anthropic
from .chat_completions import Adapter as ChatCompletions

ADAPTERS = {adapter.name: adapter for adapter in (Responses, GenerateContent, Anthropic, ChatCompletions)}


def get_adapter(protocol):
    if protocol not in ADAPTERS:
        raise ValueError(f"Unsupported protocol {protocol!r}; choose from {list(ADAPTERS)}")
    return ADAPTERS[protocol]()
