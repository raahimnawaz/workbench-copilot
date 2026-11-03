"""Tools the router can call. Each returns a dict with "ok"; failures come back as data, never as exceptions.

read_breadboard photographs the board and reads it with SAM 3.1 (wbc/perception). locate finds things on the desk
with the desk camera, and pick_up, put_down and arm_home move them with the arm (wbc/motion/hand.py), only with the
agent started with --arm: at once, the request being the go-ahead, or each only after the user presses Enter, as
robot.yaml's hand.go_ahead says. The spoken stop (Bench.halt) holds the arm where it got to. look is still a
stub. With --sim,
read_breadboard and locate return recorded scenes and the arm's tools pretend to move, so the router and coder can be
tested with no hardware. remember, recall and forget keep the router's notebook (wbc/agent/memory.py).
"""
from __future__ import annotations

import itertools
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import coder, dut, parts

MAX_READ = 64_000
SEEN_S = 60.0  # seconds a thing stays where locate found it: after that, or once the arm has moved it, locate again
MISSED_MM = 25.0  # mm from where a pick took a thing that a look finding it there after the put counts as a miss
SAID_STOP = {"ok": False, "halted": True, "error": "the user said stop, so the arm didn't move"}
LOCATES = itertools.count(1)  # each locate's files get the next number, so two at once never share one
MOVES = itertools.count(1)    # and each arm request's move log


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    label: str  # which model or system does the work, for the HUD timeline
    recheck: Callable | None = None  # brings a kept result up to date without running the tool (see Toolbox.recheck)


class Toolbox:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, description: str, parameters: dict, label: str = "code",
                 recheck: Callable | None = None):
        def decorate(fn):
            self._tools[name] = Tool(name, description, parameters, fn, label, recheck)
            return fn
        return decorate

    def names(self) -> list:
        return list(self._tools)

    def label(self, name: str) -> str:
        tool = self._tools.get(name)
        return tool.label if tool else "?"

    def schemas(self) -> list:
        """Function tools in Responses API format."""
        return [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self._tools.values()]

    def chat_schemas(self) -> list:
        """The same tools in Chat Completions format."""
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}}
                for t in self._tools.values()]

    def dispatch(self, name: str, args: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            result = tool.fn(**args)
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    def recheck(self, name: str, result: dict) -> dict:
        """An earlier result of `name`, kept for the session (models.yaml's prefetch `keep`), brought up to date by its
        tool's recheck without running the tool again; as it was for a tool without one."""
        tool = self._tools.get(name)
        return tool.recheck(result) if tool is not None and tool.recheck is not None else result


@dataclass
class Bench:
    root: Path
    models: dict
    client: object
    budget: object
    sim: bool = False
    patches: dict = field(default_factory=dict)
    layout: dict = field(default_factory=dict)  # bench.yaml: the camera, the carrier and where the parts sit
    parts: dict = field(default_factory=dict)  # parts.yaml
    say: Callable = print  # tells the user something mid-turn, such as "press the sensor now"
    flashed_gpio: int | None = None  # --sim only: the SENSOR_PIN of the last firmware "flashed"
    robot_cfg: dict = field(default_factory=dict)  # robot.yaml
    desk_cfg: dict = field(default_factory=dict)  # desk.yaml: the desk camera and its tags
    arm_port: str | None = None  # the arm's servo adapter; None keeps the arm's tools off (python -m wbc.agent --arm)
    move_logs: Path | None = None  # where each arm request's moves are saved (_save_moves); None saves none
    powering: threading.Thread | None = None  # the arm powering on while a locate runs (warm_up); move() waits for it
    desk_stream: object = None  # wbc.perception.camera.Stream: the desk camera kept open, so a locate needn't open it
    calibration_dir: str | None = None  # lerobot-calibrate's folder when it's another account's
    confirm: Callable | None = None  # asks the user and waits: True only for Enter at the keyboard
    go_ahead: str = "enter"  # robot.yaml's hand.go_ahead: "request" moves the arm at once, "enter" asks confirm first
    hand: object = None  # wbc.motion.hand.Hand, opened at the first arm move
    seen: dict = field(default_factory=dict)  # what locate last found, by name: (wbc.motion.hand.Thing, when)
    sim_held: str | None = None  # --sim only: what the jaws pretend to hold
    arm_lock: threading.Lock = field(default_factory=threading.Lock)  # one arm move at a time, whatever the router
    stopped: threading.Event = field(default_factory=threading.Event)  # the session is ending: no arm move starts
    interrupted: threading.Event = field(default_factory=threading.Event)  # Ctrl-C: nothing's staged or flashed either
    halted: threading.Event = field(default_factory=threading.Event)  # the spoken stop: no arm move until resume()
    open: tuple | None = None  # (client, model, label): Muse Spark Open on Baseten, for add_part; None without its key
    show: Callable | None = None  # with --show: puts a schematic in KiCad's window (wbc/agent/show.py)

    def halt(self) -> None:
        """The spoken stop, from any thread: a move running stops at its next tick and the arm holds where it got to,
        its motors on, and no arm move starts until resume(), when the next request comes."""
        self.halted.set()  # first: a tool opening the hand right now checks this once it has
        if self.hand is not None:
            self.hand.arm.halt()

    def resume(self) -> None:
        """A new request after a spoken stop: the arm's tools may move it again."""
        self.halted.clear()
        if self.hand is not None:
            self.hand.arm.resume()

    def stop_arm(self) -> None:
        """Ctrl-C: no arm move starts from now on, and every motor is cut, from any thread. A move running on a tool's
        thread stops at its next tick, and a tool still on its way to the arm (locating, or opening the port) is
        refused when it gets there. Nothing gets staged or flashed after it either."""
        self.interrupted.set()
        self.stopped.set()
        if self.hand is not None:
            self.hand.arm.cut()

    def can_go_home(self) -> bool:
        """Whether park(home=True) would take the arm back to rest: it's open, every motor is on, and it has a rest."""
        hand = self.hand
        return hand is not None and len(hand.arm.powered) == len(hand.arm.joints.servos) and hand.rest is not None

    def park(self, home: bool = True) -> None:
        """At exit: no arm move starts from now on. Then, if the arm is open, back to where it rested when `home` (it
        sat there unpowered before, so it stays there when the motors let go) and it wasn't told to stop since the last
        request, and every motor off either way."""
        self.stopped.set()
        locked = self.arm_lock.acquire(timeout=5.0)  # a tool's move refuses at its next step and lets go
        try:
            if self.hand is None:
                return
            arm = self.hand.arm
            try:
                if home and self.can_go_home() and self.halted.is_set():
                    self.say("The arm was told to stop, so it stays where it is. Hold it: its motors are going off.")
                elif home and self.can_go_home():
                    self.say("The arm is going back to rest, then its motors go off.")
                    self.hand.home()
            except Exception as e:  # noqa: BLE001 - parking is best effort: the motors go off either way
                self.say(f"The arm couldn't go back to rest ({e}). Hold it: the motors are going off.")
            finally:
                arm.close()
        finally:
            if locked:
                self.arm_lock.release()


def key(thing: str) -> str:
    """The name locate's results are kept under: 'the Screwdriver' and 'screwdriver' are one thing."""
    words = thing.lower().split()
    return " ".join(words[1:] if len(words) > 1 and words[0] in ("the", "a", "an", "that", "this") else words)


def _locate_live(bench: Bench, thing: str, height_mm: float) -> dict:
    """locate on the desk camera for one thing, as _locate_all does."""
    return _locate_all(bench, [(thing, height_mm)])[thing]


def _locate_all(bench: Bench, wants: list) -> dict:
    """locate on the desk camera for each (thing, height_mm) in `wants`, as scripts/desk_camera.py locate does for one:
    one photo and the touched tags' map for them all, and a SAM request for each, sent together. {thing: result}.
    Saves the photo, each SAM reply (for desk_camera.py locate --replay) and each drawn view in .wbc/vision."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np

    from wbc.perception import camera, desk, sam_client  # the eyes extra
    from wbc.perception import tags as tagmap

    cfg = bench.desk_cfg
    touched = desk.load_touched(bench.root / desk.TOUCHED)
    t0 = time.monotonic()
    _, frame = desk.find_camera(cfg, touched, bench.desk_stream.capture if bench.desk_stream else camera.capture)
    captured = time.monotonic() - t0
    size = (frame.shape[1], frame.shape[0])
    fit = desk.fit(tagmap.detect(frame, cfg["tags"]["family"]), touched, size, desk.focal_for(cfg, size[0]),
                   cfg["min_tags"])

    def segment(image, phrase):
        return sam_client.segment(bench.client, bench.models["eyes"], bench.budget, image, phrase)

    with ThreadPoolExecutor(max_workers=max(1, len(wants))) as pool:  # SAM takes one concept per request
        located = list(pool.map(lambda w: desk.locate(frame, fit, cfg, segment, w[0], w[1]), wants))
    out = bench.root / ".wbc" / "vision"
    out.mkdir(parents=True, exist_ok=True)
    when = f"{datetime.now():%Y%m%d-%H%M%S}-{next(LOCATES)}"  # two locates at once mustn't share files
    photo = out / f"desk-{when}-photo.png"
    cv2.imwrite(str(photo), frame)
    to_arm = np.linalg.inv(desk.view_matrix(cfg["region_mm"], cfg["px_per_mm"]))
    results = {}
    for (thing, height_mm), (hits, view, seg) in zip(wants, located):
        stem = out / f"desk-{when}-{''.join(c if c.isalnum() else '-' for c in key(thing))[:30]}"
        stem.with_suffix(".json").write_text(json.dumps({"phrase": thing, "height_mm": height_mm, "photo": str(photo),
                                                         "raw": seg.raw}, indent=1), encoding="utf-8")
        cv2.imwrite(str(stem.with_suffix(".png")), desk.draw(view, to_arm, touched, hits))
        results[thing] = {"found": [f.summary() for f in hits], "problems": list(seg.problems),
                          "drawn": str(stem.with_suffix(".png").relative_to(bench.root)),
                          "_timing": {"capture": captured}}
    return results


def _open_hand(bench: Bench):
    """The bench's Hand, opening the arm the first time: connected, nothing written, the motors still off."""
    if bench.hand is None:
        from wbc.motion.arm import Arm
        from wbc.motion.hand import Hand

        arm = Arm.open(bench.arm_port, bench.robot_cfg, bench.root, calibration_dir=bench.calibration_dir,
                       say=bench.say)
        bench.hand = Hand(arm, bench.robot_cfg, say=bench.say)
    return bench.hand


def _save_moves(bench: Bench, request: str, log: list) -> None:
    """One arm request's moves and jaw actions, each with the driver's samples (every joint's goal, angle and load, tick
    by tick), as arm-<time>-<n>.json in bench.move_logs: what a miss, a trip or a slow move went through."""
    out = bench.move_logs
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"arm-{datetime.now():%Y%m%d-%H%M%S}-{next(MOVES)}.json"
    path.write_text(json.dumps({"request": request, "moves": log},
                               default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")


def _not_wired(tool: str, day: str) -> dict:
    return {"ok": False, "error": f"{tool} isn't wired to hardware yet (sheet 10, {day}). "
                                  f"Run with --sim to use the recorded bench."}


def build_toolbox(bench: Bench) -> Toolbox:
    tb = Toolbox()
    patch_ids = (f"p{i}" for i in itertools.count(1))

    @tb.register("read_file", "Read a text file from the project, such as firmware sources or headers.", {
        "type": "object",
        "properties": {"path": {"type": "string",
                                "description": "Path from the repo root, e.g. firmware/dut_demo/include/pins.h"}},
        "required": ["path"]})
    def read_file(path: str) -> dict:
        text = coder.safe_path(bench.root, path).read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "path": path, "text": text[:MAX_READ], "truncated": len(text) > MAX_READ}

    @tb.register("ask_coder", (
        "Have Muse Wonder write a code change. Send the bench facts and only the files involved. Returns a "
        "patch_id plus a diff that already applies cleanly and compiles, or an error. When the result has "
        "`staged`, the patch is already in the working tree and open in VS Code."), {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "What to change and why, in one or two sentences"},
            "files": {"type": "array", "items": {"type": "string"}, "maxItems": 4,
                      "description": "Repo-relative paths the change may touch"},
            "facts": {"type": "object", "description": "Measured facts, e.g. from read_breadboard and read_sensor"}},
        "required": ["task", "files"]}, label="Wonder")
    def ask_coder(task: str, files: list, facts: dict | None = None) -> dict:
        result = coder.ask_coder(bench.client, bench.models["coder"], bench.budget, bench.root, task, files, facts)
        if result.get("ok"):
            result["patch_id"] = next(patch_ids)
            bench.patches[result["patch_id"]] = result
            if bench.models["coder"].get("auto_stage") and not bench.interrupted.is_set():  # not after Ctrl-C
                # Staging a checked diff here saves the router a whole round spent calling propose_patch.
                result["staged"] = coder.propose_patch(bench.root, result["diff"], result["why"])
        return result

    @tb.register("propose_patch", (
        "Apply a patch from ask_coder to the working tree and open it in VS Code, if ask_coder didn't stage it "
        "already. Never commits or flashes."), {
        "type": "object",
        "properties": {"patch_id": {"type": "string", "description": "patch_id returned by ask_coder"}},
        "required": ["patch_id"]})
    def propose_patch(patch_id: str) -> dict:
        patch = bench.patches.get(patch_id)
        if patch is None:
            return {"ok": False, "error": f"no patch {patch_id!r}; call ask_coder first"}
        if coder.git_apply(patch["diff"], bench.root, reverse=True)[0]:  # it reverses cleanly: it's in the tree
            return {"ok": True, "already_staged": True, "why": patch["why"]}
        return coder.propose_patch(bench.root, patch["diff"], patch["why"])

    def board_now(result: dict) -> dict:
        """A board read kept for the session, checked against pins.h as it is now: the wires and the netlist are its
        photo's, and the mismatches are worked out again, so a staged fix shows without another photo. The drawing
        shows the mismatches as they were, so it's left out once they've changed. As it was when pins.h can't be read,
        or without the eyes extra (--sim)."""
        fw = bench.layout.get("firmware") or {}
        try:
            from wbc.perception import netlist  # the eyes extra

            defines = netlist.read_defines(bench.root / fw["file"])
        except (ImportError, KeyError, OSError):
            return result
        found = result.get("mismatches") or []
        mismatches = netlist.recheck(result.get("netlist") or [], defines, fw["file"], found)
        if mismatches == found:
            return result
        return {**{k: v for k, v in result.items() if k != "_image"}, "mismatches": mismatches}

    @tb.register("read_schematic", (
        "Read the demo board's KiCad schematic, the design: each net and the pins it joins (which ESP32 pin drives "
        "what), and KiCad's electrical rules check."), {"type": "object", "properties": {}}, label="KiCad")
    def read_schematic() -> dict:
        drawn = (bench.layout or {}).get("schematic")
        if not drawn:
            return {"ok": False, "error": "bench.yaml names no schematic"}
        from wbc.perception import schematic

        return {"ok": True, **schematic.read(bench.root / drawn)}

    known = {k: v["what"] for k, v in ((bench.parts or {}).get("modules") or {}).items()
             if isinstance(v, dict) and v.get("what") and v.get("pins")}

    @tb.register("add_part", (
        "Add a part to the board's design: Muse Spark Open works out how to wire it to the ESP32's free pins and how "
        "to drive it (library and setup code), saved as a skill, and the part is drawn into a copy of the KiCad "
        "schematic, checked by KiCad's electrical rules check and shown in KiCad. Nothing on the board changes and "
        "nothing is flashed. Parts it knows: " + ("; ".join(f"{k} ({w})" for k, w in known.items()) or "none") + "."), {
        "type": "object",
        "properties": {"part": {"type": "string", "description": "The part's name from the list"},
                       "use": {"type": "string", "description": "What the user wants it for, if they said"}},
        "required": ["part"]}, label="Spark Open")
    def add_part(part: str, use: str = "") -> dict:
        return parts.add(bench, part, use)

    @tb.register("read_breadboard", (
        "Photograph the breadboard with the overhead camera, segment its jumper wires with SAM 3.1, and return "
        "each wire's two holes, the netlist, any mismatches against pins.h, and an annotated image."), {
        "type": "object", "properties": {}}, label="SAM", recheck=board_now)
    def read_breadboard() -> dict:
        if bench.sim:
            return json.loads((bench.root / "fixtures" / "breadboard_mismatch.json").read_text(encoding="utf-8"))
        if not bench.layout:
            return _not_wired("read_breadboard", "D3")
        from wbc.perception import camera, reader  # the eyes extra

        cam = bench.layout["camera"]
        t0 = time.monotonic()
        frame = camera.capture(cam["index"], cam["width"], cam["height"])
        captured = time.monotonic() - t0
        result = reader.read_board(frame, bench.layout, bench.parts, bench.client, bench.models["eyes"],
                                   bench.budget, bench.root, out_dir=bench.root / ".wbc" / "vision")
        result["_timing"] = {"capture": captured, **result["_timing"]}
        return result

    @tb.register("look", "Grab a camera frame, segment the things named in the question, and return masks and an image.", {
        "type": "object",
        "properties": {"question": {"type": "string"},
                       "camera": {"type": "string", "enum": ["overhead", "wrist"], "default": "overhead"}},
        "required": ["question"]}, label="SAM")
    def look(question: str, camera: str = "overhead") -> dict:
        return _not_wired("look", "D3")

    @tb.register("read_sensor", (
        "Read the demo board's force sensor live over serial for a few seconds. Returns the readings in "
        "millivolts (min, max, mean and a short trace), whether they answer a press, and which GPIO the "
        "firmware on the board reads. Pass `prompt` to ask the user to press first."), {
        "type": "object",
        "properties": {"seconds": {"type": "number", "minimum": 0.5, "maximum": 10, "default": 2},
                       "prompt": {"type": "string", "description": "Said to the user before reading, e.g. "
                                                                   "'Press the sensor now.'"}}}, label="board")
    def read_sensor(seconds: float = 2.0, prompt: str = "") -> dict:
        if prompt:
            bench.say(prompt)
            time.sleep(0 if bench.sim else 1.0)  # time to reach the sensor
        if bench.sim:
            return dut.sim_read(bench.flashed_gpio)
        fw = bench.layout.get("firmware", {})
        return dut.read(dut.find_port(fw), seconds, fw.get("baud", 115200))

    @tb.register("build_and_flash", (
        "Build the demo board's firmware as it is in the working tree, staged patches included, and upload it. "
        "Only after the user says go."), {"type": "object", "properties": {}}, label="PlatformIO")
    def build_and_flash() -> dict:
        if bench.sim:
            bench.flashed_gpio = dut.firmware_gpio(bench.root)
            return {"ok": True, "sim": True, "seconds": 0.0, "firmware_gpio": bench.flashed_gpio}
        if bench.interrupted.is_set():  # Ctrl-C
            return {"ok": False, "error": "the session is ending, so nothing was flashed"}
        fw = bench.layout.get("firmware", {})
        result = dut.flash(bench.root, fw["upload_cmd"], dut.find_port(fw))
        if result.get("ok"):
            result["firmware_gpio"] = dut.firmware_gpio(bench.root)
        return result

    thing_params = {
        "type": "object",
        "properties": {"thing": {"type": "string", "description": (
            "what the desk camera looks for: one plain thing in a few words, e.g. 'screwdriver', 'tray' or 'sticky "
            "note'. A longer description, such as 'bit case with yellow sticky note', can match nothing or the wrong "
            "thing. For a place from the notebook, use the name the note puts in quotes")},
                       "height_mm": {"type": "number", "minimum": 0, "maximum": 300, "default": 0,
                                     "description": "how tall it stands, for anything standing up such as a cup; 0 "
                                                    "for something lying flat"}},
        "required": ["thing"]}

    @tb.register("locate", (
        "Find a thing on the desk with the desk camera and SAM 3.1 (about $0.0025): where it is in the arm's frame in "
        "mm, its length and width, the heading of its long axis and its top's height, best match first. One camera "
        "can't see height, so give height_mm for anything standing up."), thing_params, label="SAM")
    def locate(thing: str, height_mm: float = 0) -> dict:
        return {"ok": True, "thing": thing, **located([(thing, height_mm)])[thing]}

    def located(wants: list) -> dict:
        """locate for each (thing, height_mm) in `wants`, from one photo, keeping what it finds in bench.seen:
        {thing: result}."""
        from wbc.motion.hand import Thing

        if bench.sim:
            scene = json.loads((bench.root / "fixtures" / "desk_scene.json").read_text(encoding="utf-8"))
            results = {t: {"sim": True, "found": [s for name, s in scene.items() if name in key(t)]} for t, _ in wants}
        elif len(wants) == 1:
            results = {wants[0][0]: _locate_live(bench, *wants[0])}
        else:
            results = _locate_all(bench, wants)
        for thing, _ in wants:
            if results[thing]["found"]:
                bench.seen[key(thing)] = (Thing.from_summary({**results[thing]["found"][0], "phrase": key(thing)}),
                                          time.monotonic())
            else:
                bench.seen.pop(key(thing), None)
        return results

    def left_behind(t):
        """(x, y) mm where a look finds `t` within MISSED_MM of where a pick took it from, so the jaws missed it; None
        when it's gone from there, or the look failed (which it says)."""
        try:
            found = located([(t.name, t.height_mm)])[t.name]["found"]
        except Exception as e:  # noqa: BLE001 - the camera or the tags: a look that fails tells nothing either way
            bench.say(f"I couldn't look whether the {t.name} left its spot: {e}")
            return None
        for f in found:
            if math.hypot(f["x_mm"] - t.x, f["y_mm"] - t.y) <= MISSED_MM:
                return f["x_mm"], f["y_mm"]
        return None

    def targets(wants: list):
        """([Thing, ...], None) for each (thing, height_mm) in `wants` from what locate found, locating the ones it
        hasn't in the last SEEN_S, or found at another height than one given now, again, together from one photo;
        (None, error) when one can't be found."""
        stale = []
        for thing, height_mm in wants:
            last = bench.seen.get(key(thing))
            if last is None or time.monotonic() - last[1] > SEEN_S or (height_mm > 0 and height_mm != last[0].height_mm):
                stale.append((thing, height_mm or (last[0].height_mm if last else 0)))
        if stale:
            located(stale)
        missing = [thing for thing, _ in wants if key(thing) not in bench.seen]
        if missing:
            return None, {"ok": False, "error": f"locate found no {missing[0]} on the desk"}
        return [bench.seen[key(thing)][0] for thing, _ in wants], None

    def forget(result: dict, *things) -> None:
        """Where locate saw these, forgotten after an arm move was refused (out of reach, past a joint's range): the
        user may move them before asking again ("I moved the cup, try again"), and within SEEN_S the old place would
        be planned for again."""
        if bench.arm_port is not None and "refused" in str(result.get("error", "")):
            for thing in things:
                bench.seen.pop(key(thing), None)

    def target(thing: str, height_mm: float):
        """(Thing, None) as targets() finds one; (None, error) when it can't."""
        found, problem = targets([(thing, height_mm)])
        return (found[0] if found else None), problem

    def move(question: str, run, plan=None) -> dict:
        """run(hand) under the arm lock, powering the arm on first if it's off: at once with hand.go_ahead "request",
        the request itself being the go-ahead, and once the user presses Enter with "enter". Before that, the arm is
        checked (with its motors off, past an end of a range or too hot) and plan(hand) runs, so what the driver or IK
        would refuse is refused before anything is asked or written. The driver's refusals and trips, and the spoken
        stop, come back as errors."""
        if not bench.arm_port:
            return {"ok": False, "error": "the arm's tools are off: start the agent with --arm"}
        if bench.powering is not None:  # warm_up's, started before the locate
            bench.powering.join()
            bench.powering = None
        asking = bench.go_ahead == "enter"
        if asking and bench.confirm is None:
            return {"ok": False, "error": "there's no keyboard to press Enter on, so the arm stays still"}
        from wbc.motion.arm import Halted, Refused, Tripped

        ending = {"ok": False, "error": "the session is ending, so the arm didn't move"}
        if not bench.arm_lock.acquire(blocking=False):  # two arm tools in one round would run in either order
            return {"ok": False, "error": "another arm move is running: call arm tools one at a time"}
        hand = None
        try:
            if bench.stopped.is_set():
                return ending
            if bench.halted.is_set():
                return dict(SAID_STOP)
            hand = _open_hand(bench)
            hand.log = [] if bench.move_logs is not None else None  # what this request moves, saved once it's done
            if bench.stopped.is_set():
                return ending
            off = len(hand.arm.powered) < len(hand.arm.joints.servos)
            if off:
                hand.arm.check()  # what power_on would refuse, refused before the question
            if plan is not None:
                plan(hand)
            if asking:
                answer = bench.confirm(question + (" The motors come on first." if off else ""))
                if answer is None or bench.stopped.is_set():  # the keyboard closed: the session is ending
                    return ending
                if not answer:
                    return {"ok": False, "error": "the user didn't press Enter, so the arm didn't move"}
            if bench.halted.is_set():  # said while the hand opened, the plan ran or the question waited
                return dict(SAID_STOP)
            if off:
                hand.power_on()
            return run(hand)
        except Refused as e:
            return {"ok": False, "error": f"refused, nothing moved for it: {e}"}
        except Tripped as e:
            return {"ok": False, "error": f"the motors were cut: {e}", "motors_cut": True}
        except Halted as e:
            held = bench.hand.held if bench.hand is not None else None
            return {"ok": False, "halted": True, "error": f"the user said stop, so it {e}",
                    **({"holding": held} if held else {})}
        finally:
            if hand is not None:
                log, hand.log = hand.log, None
                if log:
                    try:
                        _save_moves(bench, question, log)
                    except OSError as e:  # a log it couldn't write mustn't hide how the move went
                        bench.say(f"The arm's move log wasn't saved: {e}")
            bench.arm_lock.release()

    def warm_up():
        """With hand.go_ahead "request", the request being the go-ahead, power the arm on in the background while the
        locate before its move runs, rather than after it: powering on holds the arm where it is. move() waits for
        it, then checks the arm and powers it on itself if this didn't, saying why."""
        if bench.go_ahead != "request" or bench.sim or not bench.arm_port or bench.powering is not None:
            return
        if bench.hand is not None and len(bench.hand.arm.powered) == len(bench.hand.arm.joints.servos):
            return

        def power():
            with bench.arm_lock:
                if bench.stopped.is_set() or bench.halted.is_set():
                    return
                try:
                    hand = _open_hand(bench)
                    if bench.stopped.is_set() or bench.halted.is_set():  # Ctrl-C or the stop while the port opened
                        return
                    if len(hand.arm.powered) < len(hand.arm.joints.servos):
                        hand.arm.check()
                        hand.power_on()
                except Exception:  # noqa: BLE001 - a refusal or a trip comes back from move(), which tries again
                    pass

        bench.powering = threading.Thread(target=power, name="arm power-on", daemon=True)
        bench.powering.start()

    def check_over():
        """What pick_up and put_down ask over the thing before going down (a pick with its jaws open), while each move
        waits for Enter and robot.yaml's hand.check_over is on."""
        if bench.go_ahead != "enter":
            return None
        return bench.confirm if (bench.robot_cfg.get("hand") or {}).get("check_over", True) else None

    go = " The user presses Enter before the arm moves." if bench.go_ahead == "enter" else " The arm moves at once."
    listed = (bench.robot_cfg.get("hand") or {}).get("containers") or {}
    containers = {key(name): float(h) for name, h in listed.items()}  # a put stands things in these on end

    @tb.register("pick_up", (
        f"Pick a thing up off the desk with the arm, where locate last found it (locating it first if it hasn't).{go} "
        "Returns what the jaws hold, with `unconfirmed` when they closed all the way, which they do on nothing and on "
        "a thing held loosely alike: then say you can't be sure. An error if the user said no or stop, or the arm "
        "can't reach it. Call arm tools one at a time, never two in one round."), thing_params, label="arm")
    def pick_up(thing: str, height_mm: float = 0) -> dict:
        if bench.halted.is_set():
            return dict(SAID_STOP)
        warm_up()
        t, problem = target(thing, height_mm)
        if problem:
            return problem
        if bench.sim:
            if bench.sim_held:
                return {"ok": False, "error": f"refused, nothing moved for it: the jaws already hold the "
                                              f"{bench.sim_held}: put it down first"}
            bench.sim_held = t.name
            bench.seen.pop(key(thing), None)
            return {"ok": True, "sim": True, "holding": t.name, "x_mm": t.x, "y_mm": t.y}
        planned = []
        result = move(f"Pick up the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm?",
                      lambda hand: hand.pick_up(t, confirm=check_over(), steps=planned[-1]),
                      plan=lambda hand: planned.append(hand.plan_pick(t)))
        if result.get("ok") or "steps" in result or result.get("motors_cut") or result.get("halted"):  # it went to it
            bench.seen.pop(key(thing), None)  # in the jaws now, or nudged trying: locate it again before the next move
        forget(result, thing)
        return result

    @tb.register("put_down", (
        "Put what the arm holds down on or in a thing on the desk, such as a tray, where locate last found it "
        f"(locating it first if it hasn't).{go} Call it only after pick_up returned what the jaws hold."),
        thing_params, label="arm")
    def put_down(thing: str, height_mm: float = 0) -> dict:
        if bench.halted.is_set():
            return dict(SAID_STOP)
        inside = key(thing) in containers  # stood in on end, its height as measured
        height_mm = containers.get(key(thing), height_mm)
        t, problem = target(thing, height_mm)
        if problem:
            return problem
        if bench.sim:
            if not bench.sim_held:
                return {"ok": False, "error": "refused, nothing moved for it: the jaws hold nothing to put down"}
            held, bench.sim_held = bench.sim_held, None
            return {"ok": True, "sim": True, "put": held, "on": t.name}
        planned = []
        if inside:
            result = move(f"Put it in the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm?",
                          lambda hand: hand.put_in(t, confirm=check_over(), steps=planned[-1]),
                          plan=lambda hand: planned.append(hand.plan_insert(t, hand.held_thing, tip=hand.held_tip)
                                                          if hand.held_thing else None))
        else:
            result = move(f"Put it down on the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm?",
                          lambda hand: hand.put_down(t, confirm=check_over(), steps=planned[-1]),
                          plan=lambda hand: planned.append(hand.plan_put(t)))
        forget(result, thing)
        return result

    @tb.register("move_thing", (
        "Move a thing on the desk onto or into a place, such as a screwdriver onto a tray, in one call: one desk photo "
        f"finds both (about $0.005), then the arm picks the thing up and puts it down there.{go} Use it whenever "
        "something goes somewhere, rather than locate, pick_up and put_down one at a time. Name each with a short noun "
        "phrase the camera can see, such as 'screwdriver' or 'sticky note', not where it is, and give heights for "
        "anything standing up."), {
        "type": "object",
        "properties": {
            "thing": {"type": "string", "description": "what to pick up, a short noun phrase, e.g. 'screwdriver'"},
            "place": {"type": "string", "description": "what to put it on or in, a short noun phrase, e.g. 'tray'"},
            "thing_height_mm": {"type": "number", "minimum": 0, "maximum": 300, "default": 0,
                                "description": "how tall the thing stands, for something standing up; 0 for "
                                               "something lying flat"},
            "place_height_mm": {"type": "number", "minimum": 0, "maximum": 300, "default": 0,
                                "description": "how tall the place stands, such as a tray's rim or the case it's "
                                               "in; 0 for something flat on the desk"}},
        "required": ["thing", "place"]}, label="arm")
    def move_thing(thing: str, place: str, thing_height_mm: float = 0, place_height_mm: float = 0) -> dict:
        if bench.halted.is_set():
            return dict(SAID_STOP)
        inside = key(place) in containers  # stood in on end, its height as measured
        place_height_mm = containers.get(key(place), place_height_mm)
        warm_up()
        found, problem = targets([(thing, thing_height_mm), (place, place_height_mm)])
        if problem:
            return problem
        t, p = found
        if bench.sim:
            if bench.sim_held:
                return {"ok": False, "error": f"refused, nothing moved for it: the jaws already hold the "
                                              f"{bench.sim_held}: put it down first"}
            bench.seen.pop(key(thing), None)
            return {"ok": True, "sim": True, "moved": t.name, "onto": p.name, "x_mm": p.x, "y_mm": p.y}

        planned = {}

        def plan(hand):  # both, before anything moves: the put from where the pick leaves the arm
            planned["pick"] = hand.plan_pick(t)
            planned["put"] = (hand.plan_insert(p, t, start=planned["pick"][-1].q,
                                               tip=hand.tip_from_jaws(t, planned["pick"][2].q))
                              if inside else hand.plan_put(p, start=planned["pick"][-1].q))

        def run(hand):
            from wbc.motion.arm import Refused

            picked = hand.pick_up(t, confirm=check_over(), steps=planned.get("pick"))
            if not picked.get("ok"):
                return {**picked, "put": "not tried"}
            try:
                put = (hand.put_in if inside else hand.put_down)(p, confirm=check_over(), steps=planned.get("put"))
            except Refused as e:  # refused before that move started, so it's still in the jaws
                return {"ok": False, "holding": t.name, "error": f"picked up the {t.name}, but putting it on the "
                        f"{p.name} was refused, so the jaws still hold it: {e}", "steps": picked["steps"]}
            steps = picked["steps"] + put["steps"]
            if not put["ok"]:  # stopped over the place
                return {**put, "holding": t.name, "steps": steps}
            if picked.get("unconfirmed"):  # the jaws closed all the way: a look at where it was tells
                there = left_behind(t)
                if there is not None:
                    return {"ok": False, "missed": True, "holding": None, "error": f"the jaws closed on nothing at "
                            f"the {t.name}: it's still where it was, at x {there[0]:.0f}, y {there[1]:.0f} mm",
                            "steps": steps}
            return {"ok": True, "moved": t.name, "onto": p.name, "holding": None, "steps": steps,
                    **({"on_end": True} if inside else {})}

        result = move(f"Pick up the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm and put it {'in' if inside else 'on'} the "
                      f"{p.name} at x {p.x:.0f}, y {p.y:.0f} mm?", run, plan=plan)
        if result.get("ok") or "steps" in result or result.get("motors_cut") or result.get("halted"):  # it went to it
            bench.seen.pop(key(thing), None)
        forget(result, thing, place)
        return result

    @tb.register("arm_home", f"Move the arm back to where it rested when its motors came on.{go}",
                 {"type": "object", "properties": {}}, label="arm")
    def arm_home() -> dict:
        if bench.halted.is_set():
            return dict(SAID_STOP)
        if bench.sim:
            return {"ok": True, "sim": True, "home": True}

        def home(hand):
            hand.home()
            return {"ok": True, "home": True}

        return move("Move the arm back to where it rested?", home)

    return tb


def add_memory_tools(tb: Toolbox, notebook) -> None:
    """remember, recall and forget: the router's notebook of facts worth keeping across sessions
    (wbc/agent/memory.py), such as where a thing goes on the desk. A note never moves the arm: pick_up and put_down
    take only names, which locate turns into coordinates, and only a request moves it."""

    @tb.register("remember", (
        "Keep a fact for later sessions: a preference, a decision and its reason, a fix that worked, or where a thing "
        "goes on the desk. For a place, put in quotes the short name locate should look for there, so a later "
        "session gives locate that name: 'The screwdriver goes on the \"sticky note\" in the long recess of the bit "
        "case.' Never coordinates, wiring or where something is now: those come only from the bench tools."), {
        "type": "object",
        "properties": {"note": {"type": "string", "description": "One self-contained sentence, at most 300 characters"}},
        "required": ["note"]}, label="memory")
    def remember(note: str) -> dict:
        return notebook.add(note)

    @tb.register("recall", "Search the notebook: the best matches, newest first. With no query, the newest notes.", {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Words to look for, e.g. 'screwdriver goes'"}}},
        label="memory")
    def recall(query: str = "") -> dict:
        return {"ok": True, "notes": notebook.search(query)}

    @tb.register("forget", "Remove a note that turned out to be wrong, by the id recall or your instructions show.", {
        "type": "object",
        "properties": {"note_id": {"type": "string", "description": "e.g. 'n3'"}},
        "required": ["note_id"]}, label="memory")
    def forget(note_id: str) -> dict:
        return notebook.remove(note_id)
