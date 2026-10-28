"""Router loop: one model decides, and every other model and the arm sit behind tools.

Router speaks the Meta Model API's Responses endpoint (Muse Spark). ChatRouter speaks Chat Completions,
for an open-weights model on your own machine (Muse Glimmer) or a partner's server.
"""
from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Call:
    id: str
    name: str
    arguments: str


def image_data_url(image) -> str:
    data = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _private(result: dict) -> dict:
    """Pop the keys meant for this process, not the model: "_image" goes in as an image, "_timing" to the HUD."""
    return {k: result.pop(k) for k in [k for k in result if k.startswith("_")]}


class _RouterBase:
    label = "router"

    def __init__(self, client, model: str, toolbox, budget, hud, system: str, max_steps: int = 12,
                 prefetch: list | None = None):
        self.client = client
        self.model = model
        self.toolbox = toolbox
        self.budget = budget
        self.hud = hud
        self.system = system
        self.max_steps = max_steps
        self.prefetch = prefetch or []  # [{tool, args}] that code runs before the first call of every turn
        self._seen = {}  # each prefetched result as this conversation last saw it
        self.pool = ThreadPoolExecutor(max_workers=4)

    def _bench_facts(self) -> tuple:
        """Run the prefetch tools, so the router starts a turn with the bench facts instead of spending a
        round asking for them. Returns (text, images) to send; facts the conversation already has go out
        as "unchanged" instead of in full."""
        calls = [Call(f"prefetch{i}", p["tool"], json.dumps(p.get("args", {}))) for i, p in enumerate(self.prefetch)]
        if not calls:
            return "", []
        lines, images = [], []
        for c, res in zip(calls, self._run_calls(calls)):
            image = _private(res).get("_image")
            key, value = f"{c.name} {c.arguments}", json.dumps(res)
            if self._seen.get(key) == value:
                lines.append(f"{key}: unchanged since you last saw it")
                continue
            self._seen[key] = value
            lines.append(f"{key}: {value}")
            if image is not None:
                images.append((c.name, image))
        return ("Bench facts that code fetched before this turn. Act on them, and call these tools again only for "
                "something they don't cover:\n" + "\n".join(lines)), images

    def _note_spend(self) -> None:
        warning = self.budget.pop_warning()
        if warning:
            self.hud.note(warning)

    def _run_calls(self, calls: list) -> list:
        """Run one round of tool calls in parallel, so the arm can move while a coder call waits."""
        for c in calls:
            self.hud.call(self.toolbox.label(c.name), c.name, c.arguments)

        def one(c: Call) -> dict:
            t0 = time.monotonic()
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError as e:
                result = {"ok": False, "error": f"arguments were not valid JSON: {e}"}
            else:
                result = self.toolbox.dispatch(c.name, args)
            self.hud.result(self.toolbox.label(c.name), c.name, result, time.monotonic() - t0)
            return result

        return list(self.pool.map(one, calls))


class Router(_RouterBase):
    """Muse Spark on the Meta Model API. Conversation state lives server-side via previous_response_id."""

    label = "Spark"

    def __init__(self, *args, fallback: str | None = None, effort: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback = fallback
        self.effort = effort  # reasoning effort; None leaves the API default
        self.prev_id = None

    def run_turn(self, user_text: str) -> str:
        if self.prev_id is None:
            self._seen = {}  # a new conversation hasn't been sent anything yet
        facts, fetched = self._bench_facts()
        text = f"{user_text}\n\n{facts}" if facts else user_text
        content = text if not fetched else [{"type": "input_text", "text": text}] + [
            part for name, image in fetched for part in (
                {"type": "input_text", "text": f"Image from {name}:"},
                {"type": "input_image", "image_url": image_data_url(image)})]
        r = self._create([{"role": "user", "content": content}], self.prev_id)
        for _ in range(self.max_steps):
            calls = [Call(i.call_id, i.name, i.arguments) for i in r.output
                     if getattr(i, "type", "") == "function_call"]
            if not calls:
                self.prev_id = r.id
                break
            results = self._run_calls(calls)
            items, images = [], []
            for c, res in zip(calls, results):
                image = _private(res).get("_image")
                items.append({"type": "function_call_output", "call_id": c.id, "output": json.dumps(res)})
                if image is not None:
                    images.append({"role": "user", "content": [
                        {"type": "input_text", "text": f"Image from {c.name}:"},
                        {"type": "input_image", "image_url": image_data_url(image)}]})
            r = self._create(items + images, r.id)
        else:
            # The last response still has unanswered calls, so don't chain the next turn onto it.
            self.hud.note(f"stopped after {self.max_steps} tool rounds")
            self.prev_id = None
        text = getattr(r, "output_text", "") or ""
        self.hud.reply(self.label, text)
        return text

    def _create(self, items: list, prev_id):
        self.budget.check()
        kwargs = dict(instructions=self.system, tools=self.toolbox.schemas(), input=items,
                      previous_response_id=prev_id)
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        t0 = time.monotonic()
        try:
            r = self.client.responses.create(model=self.model, **kwargs)
        except Exception as e:
            if type(e).__name__ != "NotFoundError" or not self.fallback:
                raise
            self.hud.note(f"{self.model} not found; falling back to {self.fallback}")
            self.model, self.fallback = self.fallback, None
            r = self.client.responses.create(model=self.model, **kwargs)
        self.hud.note(f"{self.label} answered in {time.monotonic() - t0:.1f} s")
        self.budget.add_tokens(self.model, getattr(r, "usage", None))
        self._note_spend()
        return r


class ChatRouter(_RouterBase):
    """Same loop over Chat Completions, keeping the history locally."""

    label = "Glimmer"

    def __init__(self, *args, reasoning: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        system = self.system + (f"\n\nReasoning strength: {reasoning}" if reasoning else "")
        self.messages = [{"role": "system", "content": system}]

    def run_turn(self, user_text: str) -> str:
        facts, fetched = self._bench_facts()
        self.messages.append({"role": "user", "content": f"{user_text}\n\n{facts}" if facts else user_text})
        self.messages.extend({"role": "user", "content": [
            {"type": "text", "text": f"Image from {name}:"},
            {"type": "image_url", "image_url": {"url": image_data_url(image)}}]} for name, image in fetched)
        for _ in range(self.max_steps):
            self.budget.check()
            r = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                    tools=self.toolbox.chat_schemas())
            self.budget.add_tokens(self.model, getattr(r, "usage", None))
            self._note_spend()
            msg = r.choices[0].message
            calls = [Call(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])]
            turn = {"role": "assistant", "content": msg.content or ""}
            if calls:
                turn["tool_calls"] = [{"id": c.id, "type": "function",
                                       "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
            self.messages.append(turn)
            if not calls:
                self.hud.reply(self.label, msg.content or "")
                return msg.content or ""
            results = self._run_calls(calls)
            images = []
            for c, res in zip(calls, results):
                image = _private(res).get("_image")
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)})
                if image is not None:
                    images.append({"role": "user", "content": [
                        {"type": "text", "text": f"Image from {c.name}:"},
                        {"type": "image_url", "image_url": {"url": image_data_url(image)}}]})
            # Tool messages must follow the assistant turn directly, so images go after all of them.
            self.messages.extend(images)
        self.hud.note(f"stopped after {self.max_steps} tool rounds")
        return ""
