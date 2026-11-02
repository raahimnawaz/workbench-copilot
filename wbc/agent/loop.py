"""Router loop: one model decides, and every other model and the arm sit behind tools.

Router speaks the Meta Model API's Responses endpoint (Muse Spark). ChatRouter speaks Chat Completions,
for an open-weights model on your own machine (Muse Glimmer) or a partner's server.

With a Memory (wbc/agent/memory.py), every turn is recorded, and when a conversation breaks or fills up, the
next one starts from a summary of it, or from the recorded turns, instead of from nothing.

With a FastArm (fast_arm.py), an arm command its rules and the local model agree on runs first, with no call to the
router's model, and joins the conversation as a turn like any other.
"""
from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .budget import BudgetExceeded
from .fast_arm import said
from .memory import (SUMMARY_PROMPT, SUMMARY_RETRY_TURNS, call_record, context_full, input_tokens, summary_input,
                     transient, turn_record)

# What the router's model gets after the fast path's move came back failed, with the request
TRIED = ("The arm's fast path already ran {tool} {args} for this request, and it came back failed: {note}. Don't run "
         "it again unless something should change. Say what happened.")


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


def _replayed(r) -> list:
    """A response's output as input items for the conversation's next call: its text as an assistant message, then
    its function calls. Its reasoning stays out, since that can't be sent back."""
    text = getattr(r, "output_text", "") or ""
    calls = [{"type": "function_call", "call_id": i.call_id, "name": i.name, "arguments": i.arguments}
             for i in r.output if getattr(i, "type", "") == "function_call"]
    return ([{"role": "assistant", "content": text}] if text else []) + calls


def _spoken_list(names: list) -> str:
    """Tool names as words, each once: ['locate', 'pick_up', 'locate'] -> 'Locate and pick up'."""
    words = list(dict.fromkeys(n.replace("_", " ") for n in names))
    text = words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"
    return text[:1].upper() + text[1:]


class _RouterBase:
    label = "router"

    def __init__(self, client, model: str, toolbox, budget, hud, system: str, max_steps: int = 12,
                 prefetch: list | None = None, memory=None, fast=None):
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
        self.memory = memory  # what outlives a conversation; None keeps only the conversation itself
        self.fast = fast  # the arm's fast path (fast_arm.py); None sends every request to the router's model
        self._input_tokens = 0  # the last call's input: how full the conversation is
        self._failures = 0  # turns in a row that failed in a way that may mean the conversation is broken
        self._summary_after = 0  # after a summary that failed, how many recorded turns before the next try
        self.unfinished = None  # (request, tool calls that ran) while a turn runs, for main to record after Ctrl-C
        self.running = []  # the round of tool calls running now
        self._ran_out = False  # whether the last turn stopped at max_steps with calls unanswered

    def _fetch(self) -> list:
        """Run the prefetch tools, so the router starts a turn with the bench facts instead of spending a round
        asking for them: [(call, result, image)]."""
        calls = [Call(f"prefetch{i}", p["tool"], json.dumps(p.get("args", {}))) for i, p in enumerate(self.prefetch)]
        if not calls:
            return []
        return [(c, res, _private(res).get("_image")) for c, res in zip(calls, self._run_calls(calls))]

    def _bench_facts(self, fetched: list) -> tuple:
        """The fetched bench facts as (text, images) to send; facts the conversation already has go out as
        "unchanged" instead of in full."""
        if not fetched:
            return "", []
        lines, images = [], []
        for c, res, image in fetched:
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
        """Run one round of tool calls in parallel, so the arm can move while a coder call waits. Of a round's arm
        calls only the first goes to its tool, even if its arguments turn out wrong: the arm does one thing at a time,
        in the order the router asked for them, and a later one may need the first to have happened (a put after a
        pick)."""
        for c in calls:
            self.hud.call(self.toolbox.label(c.name), c.name, c.arguments)
        arm = [c for c in calls if self.toolbox.label(c.name) == "arm"]
        later = {id(c) for c in arm[1:]}

        def one(c: Call) -> dict:
            t0 = time.monotonic()
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError as e:
                result = {"ok": False, "error": f"arguments were not valid JSON: {e}"}
            else:
                if id(c) in later:
                    result = {"ok": False, "error": f"not run yet: {arm[0].name} came first this round, and arm tools "
                                                    f"run one at a time. Call {c.name} again once {arm[0].name} has "
                                                    "returned, if it's still wanted"}
                else:
                    result = self.toolbox.dispatch(c.name, args)
            self.hud.result(self.toolbox.label(c.name), c.name, result, time.monotonic() - t0)
            return result

        return list(self.pool.map(one, calls))

    # Each router supplies _turn (one turn, recording its tool calls in `done` as they return), _chained (whether
    # the next turn continues a conversation) and _summarize (one chain-free call that turns a transcript into a
    # summary).
    _keeps_failed_turns = False  # whether a turn that raised stays in the conversation the next turn continues

    def run_turn(self, user_text: str) -> str:
        """One request, from the user's words to the reply. With memory, the turn is recorded however it ends, the
        session becomes a summary first when that's due, and a full context costs one summary instead of every later
        turn. Nothing runs twice: not the bench facts, and never a tool that has acted on the bench."""
        done = []
        self.unfinished = (user_text, done)
        asked = user_text  # what the router's model gets: the request, and what the fast path did if it failed
        if self.fast is not None:
            reply = self._fast(user_text, done)
            if reply is not None:
                return reply
            if done:
                asked = f"{user_text}\n\n{TRIED.format(**done[-1])}"
        seen = dict(self._seen)
        try:
            if self.memory is not None:
                self._compact_if_due()
            retry = self.memory is not None and self._chained()  # starting over only helps a turn in a conversation
            fetched = self._fetch()
            try:
                text = self._turn(asked, done, fetched)
            except Exception as e:
                if isinstance(e, BudgetExceeded) or not (retry and context_full(e)):
                    raise
                self._compact("the conversation outgrew the model's context", broken=True)
                if done:  # tools have already acted on the bench, so the turn must not run again
                    text = ("My memory of this conversation filled up, so I started a new one from a summary. "
                            f"{_spoken_list([c['tool'] for c in done])} already ran, so tell me what's still to do.")
                    self.hud.reply(self.label, text)
                    self.unfinished = None
                    self._record(user_text, done, text, live=False)
                    return text
                text = self._turn(asked, done, fetched)  # nothing has acted yet: start over in the new one, once
        except BudgetExceeded:
            self.unfinished = None
            self._record(user_text, done, "(stopped: the budget's hard stop)", live=self._keeps_failed_turns)
            raise
        except Exception as e:
            self.unfinished = None
            if not transient(e):
                self._failures += 1
            if not self._keeps_failed_turns:
                self._seen = seen  # the conversation never received this turn's facts
            self._record(user_text, done, f"(the turn failed: {type(e).__name__}: {e})", live=self._keeps_failed_turns)
            raise
        self.unfinished = None
        self._failures = 0
        kept = text or (f"(stopped after {self.max_steps} tool rounds; the rest wasn't done)" if self._ran_out else "")
        self._record(user_text, done, kept, live=self._chained())
        return text

    def _fast(self, user_text: str, done: list) -> str | None:
        """The fast path: an arm command its rules and the local model agree on runs at once, with no call to the
        router's model, and once it has worked its result is said. The turn joins the conversation and the session as
        if the router had made the call. None when it isn't one, or when the move came back failed: then it's in
        `done`, for the router's model to explain."""
        command = self.fast.take(user_text)
        if command is None:
            return None
        result = self._round([Call("fast0", command["tool"], json.dumps(command["args"]))], done)[0]
        if not result.get("ok", True):
            return None
        text = said(command["tool"], result)
        self.hud.reply("fast path", text)
        self._join(user_text, text)
        self.unfinished = None
        self._failures = 0
        self._record(user_text, done, text, live=True)
        return text

    def _join(self, user_text: str, reply: str) -> None:
        """A turn the fast path answered, added to the conversation the next call sends."""
        raise NotImplementedError

    def record_unfinished(self, why: str) -> None:
        """After Ctrl-C, once the motors are cut: record the turn it stopped, with what ran and what was running."""
        if self.unfinished is None:
            return
        user_text, done = self.unfinished
        cut = [call_record(c.name, c.arguments, {"ok": False, "error": why}) for c in self.running]
        self.unfinished, self.running = None, []
        self._record(user_text, done + cut, f"({why})", live=False)

    def _instructions(self) -> str:
        remembered = self.memory.prompt() if self.memory else ""
        return f"{self.system}\n\n{remembered}" if remembered else self.system

    def _records(self, calls: list, results: list) -> list:
        return [call_record(c.name, c.arguments, res) for c, res in zip(calls, results)]

    def _round(self, calls: list, done: list) -> list:
        """Run one round of tool calls and record them in `done` first thing, so nothing after can lose or repeat
        them."""
        self.running = calls
        results = self._run_calls(calls)
        done.extend(self._records(calls, results))
        self.running = []
        return results

    def _record(self, user_text: str, done: list, reply: str, live: bool) -> None:
        if self.memory is None:
            return
        try:
            self.memory.session.record(turn_record(user_text, done, reply, live))
        except OSError as e:  # the turn happened all the same, so it mustn't look failed and get asked again
            self.hud.note(f"memory: couldn't save the session ({type(e).__name__}: {e})")

    def _compact_if_due(self) -> None:
        chained = self._chained()
        due = self.memory.compact_reason(self._input_tokens, self._failures if chained else 0, chained)
        if due and (due[1] or len(self.memory.session.turns) >= self._summary_after):
            self._compact(*due)

    def _compact(self, reason: str, broken: bool = False) -> None:
        """Fold the session into a summary and start a new conversation, which gets the summary in its instructions.
        If the summary call fails, a conversation that may be broken starts over with the turns as text, and any
        other carries on and tries again a few turns later."""
        session = self.memory.session
        summary, problem = "", "there was nothing to summarize"
        if session.turns:
            try:
                summary = self._summarize(summary_input(session))
                problem = "" if summary else "it came back empty"
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - a missing summary mustn't cost the turn
                problem = type(e).__name__
        n = len(session.turns)
        if summary:
            self.hud.note(f"memory: {reason}, so {n} turn{'' if n == 1 else 's'} became a summary")
            self._summary_after = 0
            try:
                session.fold(summary)
            except OSError as e:
                self.hud.note(f"memory: couldn't save the session ({type(e).__name__}: {e})")
        elif not broken:
            self._summary_after = n + SUMMARY_RETRY_TURNS
            self.hud.note(f"memory: {reason}, but the summary failed ({problem}), so the conversation carries on and "
                          f"the summary waits {SUMMARY_RETRY_TURNS} turns")
            return
        else:
            self.hud.note(f"memory: {reason}, but the summary failed ({problem}), so the turns carry over as they are")
        self._new_conversation()

    def _new_conversation(self) -> None:
        self._seen = {}
        self._input_tokens = self._failures = 0
        if self.memory is not None:
            self.memory.session.drop_live()


class Router(_RouterBase):
    """Muse Spark on the Meta Model API's Responses endpoint.

    The conversation is kept here and sent whole with every call, rather than chained on the last response's id.
    On Nov 1, every call chained with previous_response_id took 11.8–13.8 s, whether or not it carried a tool result,
    while the same conversation's first call, sent whole, took 3.4 s (pipeline-notes/spark-chaining-test). Only a
    finished turn joins the conversation, so a failed one leaves it as it was, as it did when chained."""

    label = "Spark"

    def __init__(self, *args, fallback: str | None = None, effort: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback = fallback
        self.effort = effort  # reasoning effort; None leaves the API default
        self.history = []  # the conversation's finished turns as input items, Spark's replies and calls included

    def _chained(self) -> bool:
        return bool(self.history)

    def _join(self, user_text: str, reply: str) -> None:
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]

    def _new_conversation(self) -> None:
        super()._new_conversation()
        self.history = []

    def _summarize(self, text: str) -> str:
        """One call with no tools and no previous response, so it fits however full the conversation got."""
        self.budget.check()
        kwargs = dict(instructions=SUMMARY_PROMPT, input=[{"role": "user", "content": text}])
        if self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        r = self.client.responses.create(model=self.model, **kwargs)
        self.budget.add_tokens(self.model, getattr(r, "usage", None))
        self._note_spend()
        return (getattr(r, "output_text", "") or "").strip()

    def _turn(self, user_text: str, done: list, fetched: list) -> str:
        self._ran_out = False
        if not self.history:
            self._seen = {}  # a new conversation hasn't been sent anything yet
        facts, pictures = self._bench_facts(fetched)
        text = f"{user_text}\n\n{facts}" if facts else user_text
        content = text if not pictures else [{"type": "input_text", "text": text}] + [
            part for name, image in pictures for part in (
                {"type": "input_text", "text": f"Image from {name}:"},
                {"type": "input_image", "image_url": image_data_url(image)})]
        turn = [{"role": "user", "content": content}]
        r = self._create(self.history + turn)
        for _ in range(self.max_steps):
            turn += _replayed(r)
            calls = [Call(i.call_id, i.name, i.arguments) for i in r.output
                     if getattr(i, "type", "") == "function_call"]
            if not calls:
                self.history += turn
                break
            results = self._round(calls, done)
            items, images = [], []
            for c, res in zip(calls, results):
                image = _private(res).get("_image")
                items.append({"type": "function_call_output", "call_id": c.id, "output": json.dumps(res)})
                if image is not None:
                    images.append({"role": "user", "content": [
                        {"type": "input_text", "text": f"Image from {c.name}:"},
                        {"type": "input_image", "image_url": image_data_url(image)}]})
            turn += items + images
            r = self._create(self.history + turn)
        else:
            # The last response still has unanswered calls, so the next turn starts a new conversation.
            self.hud.note(f"stopped after {self.max_steps} tool rounds")
            self._ran_out = True
            self._new_conversation()
        text = getattr(r, "output_text", "") or ""
        self.hud.reply(self.label, text)
        return text

    def _create(self, items: list):
        self.budget.check()
        kwargs = dict(instructions=self._instructions(), tools=self.toolbox.schemas(), input=items)
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
        self._input_tokens = input_tokens(getattr(r, "usage", None))
        self.budget.add_tokens(self.model, getattr(r, "usage", None))
        self._note_spend()
        return r


class ChatRouter(_RouterBase):
    """Same loop over Chat Completions, keeping the history locally."""

    label = "Glimmer"
    _keeps_failed_turns = True  # a failed turn's messages stay in the local history

    def __init__(self, *args, reasoning: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reasoning = f"\n\nReasoning strength: {reasoning}" if reasoning else ""
        self.messages = [{"role": "system", "content": self._instructions()}]

    def _instructions(self) -> str:
        return super()._instructions() + self._reasoning  # still the system prompt's last line, as before memory

    def _chained(self) -> bool:
        return len(self.messages) > 1

    def _join(self, user_text: str, reply: str) -> None:
        self.messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]

    def _new_conversation(self) -> None:
        super()._new_conversation()
        del self.messages[1:]

    def _summarize(self, text: str) -> str:
        self.budget.check()
        r = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": text}])
        self.budget.add_tokens(self.model, getattr(r, "usage", None))
        self._note_spend()
        return (r.choices[0].message.content or "").strip()

    def _turn(self, user_text: str, done: list, fetched: list) -> str:
        self._ran_out = False
        self.messages[0]["content"] = self._instructions()  # the memory section may have changed since last turn
        facts, pictures = self._bench_facts(fetched)
        self.messages.append({"role": "user", "content": f"{user_text}\n\n{facts}" if facts else user_text})
        self.messages.extend({"role": "user", "content": [
            {"type": "text", "text": f"Image from {name}:"},
            {"type": "image_url", "image_url": {"url": image_data_url(image)}}]} for name, image in pictures)
        for _ in range(self.max_steps):
            self.budget.check()
            r = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                    tools=self.toolbox.chat_schemas())
            self._input_tokens = input_tokens(getattr(r, "usage", None))
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
            results = self._round(calls, done)
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
        self._ran_out = True
        return ""
