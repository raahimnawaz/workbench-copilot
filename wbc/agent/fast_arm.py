"""The arm's fast path: a spoken arm command runs at once, without Spark, when plain rules and a local model agree.

Before any arm tool runs, Spark's first call of a turn takes about 2 s, and its reply once the move is done about 2.4 s
more. Here the rules read the request first: the kind of command from
its trigger words ("put", "pick up", "go home"), the thing and the place from the names in models.yaml's fast_arm
section (or, for "away" or "back", the place a note gives the thing in quotes), and the skill from what the arm holds.
Only when they find an arm command does Llama 3.2 3B, served on this Mac by mlx_lm.server, read the request too, and
the arm moves only if it names the same kind of command. The model is a second opinion that can veto, never a source of
names, and a request the rules don't take costs no model call at all. Everything else goes to Spark as before: a
question, a negation, a statement about yourself, a bare "that", a thing the rules don't know, a model that disagrees,
answers slowly or isn't running.

On 117 phrasings that neither was tuned on, the model alone would have moved the arm wrongly 20 times, the rules alone 3
times (each now caught by a rule here), and the two agreeing never; 49 of the 62 commands ran locally.

Nothing is said before a move. Once it's done, the reply is built from the skill's result; a move that comes back
failed goes to Spark to explain (loop.py).
"""
from __future__ import annotations

import json
import re
import threading
import time

from .ears import wake_request

# The prompt as benchmarked, kept as it was, so its answers are the ones measured.
SYSTEM = """You sort what the user says to the voice co-pilot of an electronics bench, where a robot arm sits on the desk. Most of what the user says is for the main assistant. Your only job is to catch a direct command to move the arm now and turn it into one call.

Answer with one JSON object and nothing else, in this form:
{"tool": "<tool>", "thing": "<name>" or null, "place": "<name>" or null}

The tools:
- "move_thing": pick a thing up off the desk and put it on or in a place. Needs thing and place.
- "pick_up": pick a thing up and keep holding it. Needs thing.
- "put_down": put down what the arm is already holding, on or in a place. Needs place.
- "arm_home": send the arm back to its rest pose.
- "hold": hold something the user is handing the arm, such as a probe ("hold this for me").
- "let_go": open the jaws and let go of what the user handed the arm.
- "not_mine": everything else.

Rules:
1. A name is a short noun phrase the desk camera can see, without "the" or "my": "screwdriver", "sticky note".
2. "that", "it" or "this" for a thing on the desk means the thing that last landed on the desk.
3. "Put it away", "put it back" or "where it goes" means the place the notebook gives for that thing: use the name in quotes.
4. If the arm already holds the thing, "put it on X" is put_down. If the arm holds nothing, "put X on Y" is move_thing.
5. "not_mine" is for questions (even about the arm), wiring, firmware, code, flashing, schematics, voltages, readings, answers such as "yes" or "do it", thanks, being told not to move, and anything you are unsure of. Pick, put, hold, go and let used in another sense are "not_mine".
6. Ignore the wake word ("copilot", "co-pilot") and fillers such as "um", "uh" and "okay".

Examples:
"copilot put the glue gun in the red bin" -> {"tool": "move_thing", "thing": "glue gun", "place": "red bin"}
"Copilot, grab the solder spool." -> {"tool": "pick_up", "thing": "solder spool", "place": null}
"um set it down on the mat" (the arm holds the solder spool) -> {"tool": "put_down", "thing": null, "place": "mat"}
"copilot back to your rest position" -> {"tool": "arm_home", "thing": null, "place": null}
"Copilot, can you hold this wire for me?" -> {"tool": "hold", "thing": null, "place": null}
"okay you can release it now" (the arm holds what the user handed it) -> {"tool": "let_go", "thing": null, "place": null}
"copilot what's the resistor value on R3" -> {"tool": "not_mine", "thing": null, "place": null}
"Copilot, pick the faster option." -> {"tool": "not_mine", "thing": null, "place": null}
"copilot hold up what was that error" -> {"tool": "not_mine", "thing": null, "place": null}
"yeah do it" -> {"tool": "not_mine", "thing": null, "place": null}"""

MAX_TOKENS = 64  # an answer is about 19 tokens
START_TRIES = 3  # requests start() makes before leaving the fast path off, since the server may still be loading
START_GAP_S = 1.0  # seconds between them
# What the model and the rules have to agree on. The hand-over hold's hold and let_go aren't tools yet, so the rules
# leave those requests to Spark; the benchmark took let_go only during the hold, and nothing else then.
KINDS = {"pick_up": "carry", "put_down": "carry", "move_thing": "carry", "arm_home": "home", "hold": "hold",
         "let_go": "let go"}
# Questions, negations, waiting and the notebook are Spark's, and so is talk about yourself ("I put my screwdriver in
# the cup", "I'm going to pick it up myself": two of the rules' three wrong moves in the benchmark)
NOT_A_COMMAND = re.compile(r"^(what|whats|why|how|where|when|which|who|is|are|was|were|does|do|did|has|have|should|"
                           r"can the|could the|will the|i|im|i'm|ive|i've|ill|i'll|i'd|we|we're|we've|we'll)\b|"
                           r"\b(don't|dont|do not|never|not yet|wait|stop|remember|notebook)\b")
TRIGGER = {
    "arm_home": re.compile(r"\b(go|going|send|back|return|move)\b.*\b(home|rest)\b|\bhome position\b|\barm home\b"),
    "pick": re.compile(r"\b(pick|grab|take|lift|get)\b"),
    "put": re.compile(r"\b(put|place|set|stick|move|drop|bring|tidy|return|chuck)\b"),
}
AWAY = re.compile(r"\b(away|back|where (it|they) (goes|go|belongs?|lives?))\b")
# A request in clauses ("put my screwdriver in my cup, and add my screen") is the fast path's only when every clause
# moves the arm ("pick it up and put it on the note") or is a courtesy: one asking for anything else goes to Spark,
# which runs the arm's tool and the other side by side in one round, instead of the fast path dropping half of it.
CLAUSE = re.compile(r"[,;]|\b(?:and|then|while|also|after that|plus)\b")
COURTESY = frozenset("please thanks thank you copilot now ok okay for me".split())
NOTE = re.compile(r'\bthe (.+?) goe?s? .*?"(.+?)"', re.IGNORECASE)  # The screwdriver goes on the "sticky note" in ...
INSIDE = ("cup", "mug", "tray", "box", "bin", "bowl", "jar", "case", "drawer")  # it goes in these, not on them


def plain(text: str) -> str:
    """The request as the rules read it: lower case, without the wake phrase, punctuation or leading fillers."""
    t = (wake_request(text, "copilot") or text).lower().replace("’", "'")
    t = " ".join(re.sub(r"[^a-z0-9' ]+", " ", t).split())
    return re.sub(r"^((um|uh|okay|ok|hey|so|and|well)\s+)+", "", t)


def asks_more(text: str) -> bool:
    """Whether a clause of the request asks for something besides moving the arm (see CLAUSE)."""
    for part in CLAUSE.split(text.lower()):
        part = plain(part)
        words = set(part.split()) - COURTESY
        if words and not any(r.search(part) for r in (*TRIGGER.values(), AWAY)):
            return True
    return False


def named(table: dict, t: str) -> str | None:
    """The first name in `table` ({name: spellings}) that `t` says, by any of its spellings."""
    for name, spellings in table.items():
        if any(re.search(rf"\b{re.escape(s.lower())}\b", t) for s in (name, *spellings)):
            return name
    return None


def goes(notes: list, things: dict, places: dict) -> dict:
    """Where the notebook says things go, by the name it puts in quotes: {"screwdriver": "sticky note"}."""
    out = {}
    for note in notes:
        m = NOTE.search(note)
        thing = m and named(things, m.group(1).lower())
        if thing:
            out[thing] = named(places, m.group(2).lower()) or m.group(2)
    return out


def call(tool: str, thing: str | None = None, place: str | None = None) -> dict:
    return {"tool": tool, "thing": thing, "place": place}


def rules(text: str, holding: str | None, things: dict, places: dict, notes: list = ()) -> dict:
    """The arm command the rules find in `text`, with what the arm holds (a thing's name, or None), or not_mine.
    A thing has to be named: a bare "that" or "it" is Spark's."""
    t = plain(text)
    if not t or NOT_A_COMMAND.search(t) or asks_more(text):
        return call("not_mine")
    if TRIGGER["arm_home"].search(t):
        return call("arm_home")
    held = (named(things, holding.lower()) or holding) if holding else None
    thing = named(things, t)
    place = named(places, t) or (goes(notes, things, places).get(thing or held) if AWAY.search(t) else None)
    if held:
        if thing and thing != held:  # another thing than the one in the jaws
            return call("not_mine")
        return call("put_down", place=place) if place and TRIGGER["put"].search(t) else call("not_mine")
    if thing and place and (TRIGGER["put"].search(t) or TRIGGER["pick"].search(t)):
        return call("move_thing", thing, place)
    if thing and TRIGGER["pick"].search(t):
        return call("pick_up", thing)
    return call("not_mine")


def parse(answer: str) -> dict:
    """The model's JSON object, from the whole answer or the first {...} in it; {} when there's none."""
    for text in (answer.strip(), (re.search(r"\{.*?\}", answer, re.S) or [""])[0]):
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        return obj if isinstance(obj, dict) else {}
    return {}


def arguments(c: dict) -> dict:
    """The tool's arguments for a rules call: put_down's place is its `thing` argument."""
    if c["tool"] == "move_thing":
        return {"thing": c["thing"], "place": c["place"]}
    if c["tool"] == "pick_up":
        return {"thing": c["thing"]}
    if c["tool"] == "put_down":
        return {"thing": c["place"]}
    return {}


def said(tool: str, result: dict) -> str:
    """What to say once a move the fast path ran has worked, from its result."""
    def where(thing: str, place: str) -> str:
        are = thing.endswith("s") and not thing.endswith("ss")  # pliers, tweezers
        return f"The {thing} {'are' if are else 'is'} {'in' if place.split()[-1] in INSIDE else 'on'} the {place}."

    if tool == "move_thing" and result.get("moved"):
        return where(result["moved"], result["onto"])
    if tool == "put_down" and result.get("put"):
        return where(result["put"], result["on"])
    if tool == "pick_up" and result.get("holding"):
        if result.get("unconfirmed"):
            return f"I think I've got the {result['holding']}: the jaws closed all the way, so I can't be sure."
        return f"I've got the {result['holding']}."
    if tool == "arm_home":
        return "Back at rest."
    return "Done."


class FastArm:
    """The rules and the local model, against the bench's toolbox. take() gives the arm command to run now, or None
    for Spark. Off until start() has had an answer from the server."""

    def __init__(self, cfg: dict, toolbox, bench, notebook, hud, client=None, clock=time.monotonic, sleep=time.sleep):
        self.model = cfg["model"]
        self.label = cfg.get("label", "Llama 3.2 3B")
        self.url = cfg.get("base_url", "")
        self.timeout_s = cfg.get("timeout_s", 3.0)
        self.keep_warm_s = cfg.get("keep_warm_s", 20.0)
        self.things = {name: tuple(s) for name, s in (cfg.get("things") or {}).items()}
        self.places = {name: tuple(s) for name, s in (cfg.get("places") or {}).items()}
        self.toolbox, self.bench, self.notebook, self.hud, self.clock = toolbox, bench, notebook, hud, clock
        self.sleep = sleep
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=self.url, api_key="local", timeout=self.timeout_s, max_retries=0)
        self.client = client
        self.ready = False

    def start(self) -> bool:
        """At start-up, on its own thread: one request, since the server loads the model on the first, then one every
        keep_warm_s, which keeps answers near 0.16 s (two idle minutes made them 0.5 s). False, with the fast path left
        off, when the server doesn't answer START_TRIES times; keeping it warm then keeps asking, and turns it on once
        it answers."""
        t0 = self.clock()
        for tries in range(1, START_TRIES + 1):
            try:
                self._ask("(nothing yet)")
                break
            except Exception as e:  # noqa: BLE001 - no server, no model: Spark takes every request, as before
                why = e
                if tries < START_TRIES:
                    self.sleep(START_GAP_S)
        else:
            self.hud.note(f"fast path off: {self.label} at {self.url} didn't answer ({type(why).__name__}: {why})"
                          + (f"; it asks again every {self.keep_warm_s:g} s" if self.keep_warm_s else ""))
            if self.keep_warm_s:
                threading.Thread(target=self._keep_warm, daemon=True).start()
            return False
        self.ready = True
        self.hud.note(f"fast path on: {self.label} answered in {self.clock() - t0:.1f} s")
        if self.keep_warm_s:
            threading.Thread(target=self._keep_warm, daemon=True).start()
        return True

    def take(self, text: str) -> dict | None:
        """{"tool": ..., "args": {...}} to run now when the rules find an arm command in `text` and the model names
        the same kind; None sends it to Spark."""
        if not self.ready:
            return None
        holding = self._holding()
        proposed = rules(text, holding, self.things, self.places, self._notes())
        if proposed["tool"] == "not_mine" or proposed["tool"] not in self.toolbox.names():
            return None
        t0 = self.clock()
        try:
            with self.hud.calling(self.model):
                answer = parse(self._ask(text))
        except Exception as e:  # noqa: BLE001 - slow or gone: Spark takes it
            self.hud.note(f"{self.label} didn't answer ({type(e).__name__}), so Spark takes it")
            return None
        said_tool = answer.get("tool")
        agreed = KINDS.get(said_tool) == KINDS[proposed["tool"]]
        self.hud.note(f"{self.label}: {said_tool} in {self.clock() - t0:.2f} s; the rules: {proposed['tool']}"
                      + ("" if agreed else ", so Spark takes it"))
        return {"tool": proposed["tool"], "args": arguments(proposed)} if agreed else None

    def _holding(self) -> str | None:
        if self.bench.sim:
            return self.bench.sim_held
        hand = self.bench.hand
        return hand.held if hand is not None else None

    def _notes(self) -> list:
        return [n["note"] for n in self.notebook.notes] if self.notebook is not None else []

    def _ask(self, text: str) -> str:
        holding = self._holding()
        notes = " ".join(f"({i}) {n}" for i, n in enumerate(self._notes(), 1)) or "(empty)"
        state = (f"Bench state:\n- Notebook: {notes}\n- Last landed on the desk: nothing new.\n"
                 f"- The arm is holding: {f'the {holding}, which it picked up' if holding else 'nothing'}.\n\n"
                 f'What the user said: "{text}"')
        r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=MAX_TOKENS, messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": state}])
        return r.choices[0].message.content or ""

    def _keep_warm(self) -> None:
        while True:
            self.sleep(self.keep_warm_s)
            try:
                self._ask("(nothing yet)")
            except Exception:  # noqa: BLE001 - the next request finds out, and goes to Spark
                continue
            if not self.ready:  # off since start-up: it answers now
                self.ready = True
                self.hud.note(f"fast path on: {self.label} answered")
