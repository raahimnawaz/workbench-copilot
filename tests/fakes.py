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


class FakeSam:
    """A Meta client whose SAM 3.1 replies come from a {phrase: reply} table, so parallel requests each get
    their own. A reply that is an exception is raised instead."""

    def __init__(self, replies):
        self.replies = replies
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies[kwargs["input"][0]["content"][0]["text"]]
        if isinstance(reply, Exception):
            raise reply
        return NS(output_text=reply, usage=None)


def sam_reply(objects, width, height):
    """A SAM 3.1 reply line for a width x height image. Each object is (box, mask, score): box is
    (left, top, right, bottom) with right and bottom exclusive, mask a 0/1 array, score None or 0-1.
    Masks are one_bit, written by meta-sam-parser's own encoder (private, so the version is pinned)."""
    from meta_sam_parser._mask_codec import _encode_raster

    records = []
    for i, (box, mask, score) in enumerate(objects):
        left, top, right, bottom = box
        rows, cols = mask.shape
        c = "" if score is None else f";c={score}"
        payload = _encode_raster([int(v) for v in mask.ravel()], cols, rows)
        records.append(f"{i}<|box;x1={left};y1={top};x2={right - 1};y2={bottom - 1};w={width};h={height}{c}|>"
                       f"<|mask;x=0;y=0{c};data={rows},{cols},{payload}|>")
    return "<0f>" + ",".join(records) if records else ""
