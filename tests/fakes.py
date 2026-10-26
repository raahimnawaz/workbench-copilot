"""Stand-ins for the OpenAI SDK's response objects, so tests run offline."""
from types import SimpleNamespace as NS


def usage(tokens_in=1000, tokens_out=100):
    return NS(input_tokens=tokens_in, output_tokens=tokens_out, input_tokens_details=NS(cached_tokens=0))


def function_call(call_id, name, arguments):
    return NS(type="function_call", call_id=call_id, name=name, arguments=arguments)


def response(rid, output=(), text=""):
    return NS(id=rid, output=list(output), output_text=text, usage=usage())


class FakeResponses:
    """Replays scripted responses and records every request."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, scripted):
        self.responses = FakeResponses(scripted)


class FakeChatCompletions(FakeResponses):
    pass


def chat_response(content="", tool_calls=()):
    calls = [NS(id=cid, function=NS(name=name, arguments=args)) for cid, name, args in tool_calls]
    message = NS(content=content, tool_calls=calls or None)
    return NS(choices=[NS(message=message)], usage={"prompt_tokens": 500, "completion_tokens": 50})


class FakeChatClient:
    def __init__(self, scripted):
        self.chat = NS(completions=FakeChatCompletions(scripted))


class SilentHud:
    def __init__(self):
        self.events = []

    def call(self, label, name, arguments):
        self.events.append(("call", label, name))

    def result(self, label, name, result, seconds):
        self.events.append(("result", label, name, result.get("ok", True)))

    def note(self, text):
        self.events.append(("note", text))

    def reply(self, label, text):
        self.events.append(("reply", label, text))
