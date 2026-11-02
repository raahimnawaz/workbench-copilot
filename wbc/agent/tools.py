"""Tools the router can call. Each returns a dict with "ok"; failures come back as data, never as exceptions.

read_breadboard photographs the board and reads it with SAM 3.1 (wbc/perception). locate finds things on the desk
with the desk camera, and pick_up, put_down and arm_home move them with the arm (wbc/motion/hand.py), each only after
the user presses Enter, and only with the agent started with --arm. look and point_at are still stubs. With --sim,
read_breadboard and locate return recorded scenes and the arm's tools pretend to move, so the router and coder can be
tested with no hardware.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import coder, dut

MAX_READ = 64_000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    label: str  # which model or system does the work, for the HUD timeline


class Toolbox:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, description: str, parameters: dict, label: str = "code"):
        def decorate(fn):
            self._tools[name] = Tool(name, description, parameters, fn, label)
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
    calibration_dir: str | None = None  # lerobot-calibrate's folder when it's another account's
    confirm: Callable | None = None  # asks the user and waits: True only for Enter at the keyboard
    hand: object = None  # wbc.motion.hand.Hand, opened at the first arm move
    seen: dict = field(default_factory=dict)  # what locate last found, by name: wbc.motion.hand.Thing
    sim_held: str | None = None  # --sim only: what the jaws pretend to hold
    arm_lock: threading.Lock = field(default_factory=threading.Lock)  # one arm move at a time, whatever the router

    def stop_arm(self) -> None:
        """Cut every motor now, from any thread (Ctrl-C): a move running on a tool's thread stops at its next tick."""
        if self.hand is not None:
            self.hand.arm.cut()

    def park(self) -> None:
        """At exit: back to where the arm rested, then every motor off. It sat there unpowered before, so it stays
        there when they let go."""
        if self.hand is None:
            return
        arm = self.hand.arm
        if len(arm.powered) == len(arm.joints.servos) and self.hand.rest is not None:
            try:
                with self.arm_lock:
                    self.hand.home()
            except Exception as e:  # noqa: BLE001 - parking is best effort: the motors go off either way
                self.say(f"The arm couldn't go back to rest ({e}). Hold it: the motors are going off.")
        arm.close()


def key(thing: str) -> str:
    """The name locate's results are kept under: 'the Screwdriver' and 'screwdriver' are one thing."""
    words = thing.lower().split()
    return " ".join(words[1:] if len(words) > 1 and words[0] in ("the", "a", "an", "that", "this") else words)


def _locate_live(bench: Bench, thing: str, height_mm: float) -> dict:
    """locate on the desk camera, as scripts/desk_camera.py locate does: a photo, the touched tags' map, one SAM
    request. Saves the photo, SAM's reply (for desk_camera.py locate --replay) and the drawn view in .wbc/vision."""
    import cv2
    import numpy as np

    from wbc.perception import camera, desk, sam_client  # the eyes extra
    from wbc.perception import tags as tagmap

    cfg = bench.desk_cfg
    touched = desk.load_touched(bench.root / desk.TOUCHED)
    t0 = time.monotonic()
    _, frame = desk.find_camera(cfg, touched, camera.capture)
    captured = time.monotonic() - t0
    size = (frame.shape[1], frame.shape[0])
    fit = desk.fit(tagmap.detect(frame, cfg["tags"]["family"]), touched, size, desk.focal_for(cfg, size[0]),
                   cfg["min_tags"])

    def segment(image, phrase):
        return sam_client.segment(bench.client, bench.models["eyes"], bench.budget, image, phrase)

    hits, view, seg = desk.locate(frame, fit, cfg, segment, thing, height_mm)
    out = bench.root / ".wbc" / "vision"
    out.mkdir(parents=True, exist_ok=True)
    when = datetime.now().strftime("%Y%m%d-%H%M%S")
    photo = out / f"desk-{when}-photo.png"
    cv2.imwrite(str(photo), frame)
    (out / f"desk-{when}.json").write_text(json.dumps({"phrase": thing, "height_mm": height_mm, "photo": str(photo),
                                                      "raw": seg.raw}, indent=1), encoding="utf-8")
    drawn = out / f"desk-{when}.png"
    to_arm = np.linalg.inv(desk.view_matrix(cfg["region_mm"], cfg["px_per_mm"]))
    cv2.imwrite(str(drawn), desk.draw(view, to_arm, touched, hits))
    return {"found": [f.summary() for f in hits], "problems": list(seg.problems),
            "drawn": str(drawn.relative_to(bench.root)), "_timing": {"capture": captured}}


def _open_hand(bench: Bench):
    """The bench's Hand, opening the arm the first time: connected, nothing written, the motors still off."""
    if bench.hand is None:
        from wbc.motion.arm import Arm
        from wbc.motion.hand import Hand

        arm = Arm.open(bench.arm_port, bench.robot_cfg, bench.root, calibration_dir=bench.calibration_dir,
                       say=bench.say)
        bench.hand = Hand(arm, bench.robot_cfg, say=bench.say)
    return bench.hand


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
            "facts": {"type": "object", "description": "Measured facts, e.g. the mismatch from read_breadboard"}},
        "required": ["task", "files"]}, label="Wonder")
    def ask_coder(task: str, files: list, facts: dict | None = None) -> dict:
        result = coder.ask_coder(bench.client, bench.models["coder"], bench.budget, bench.root, task, files, facts)
        if result.get("ok"):
            result["patch_id"] = next(patch_ids)
            bench.patches[result["patch_id"]] = result
            if bench.models["coder"].get("auto_stage"):
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

    @tb.register("read_breadboard", (
        "Photograph the breadboard with the overhead camera, segment its jumper wires with SAM 3.1, and return "
        "each wire's two holes, the netlist, any mismatches against pins.h, and an annotated image."), {
        "type": "object", "properties": {}}, label="SAM")
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

    @tb.register("point_at", (
        "Hover the lit stylus over a breadboard hole, row or wire id from read_breadboard. Returns the tip "
        "error measured by the wrist camera."), {
        "type": "object",
        "properties": {"target": {"type": "string", "description": "e.g. 'a5', 'row:5', 'wire:2'"},
                       "hover_mm": {"type": "number", "minimum": 4, "maximum": 30, "default": 8}},
        "required": ["target"]}, label="arm")
    def point_at(target: str, hover_mm: float = 8) -> dict:
        if bench.sim:
            return {"ok": True, "sim": True, "target": target, "hover_mm": hover_mm, "tip_error_mm": 0.0}
        return _not_wired("point_at", "D5")

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
        fw = bench.layout.get("firmware", {})
        result = dut.flash(bench.root, fw["upload_cmd"], dut.find_port(fw))
        if result.get("ok"):
            result["firmware_gpio"] = dut.firmware_gpio(bench.root)
        return result

    thing_params = {
        "type": "object",
        "properties": {"thing": {"type": "string", "description": "a short noun phrase, e.g. 'screwdriver' or 'tray'"},
                       "height_mm": {"type": "number", "minimum": 0, "maximum": 300, "default": 0,
                                     "description": "how tall it stands, for anything standing up such as a cup; 0 "
                                                    "for something lying flat"}},
        "required": ["thing"]}

    @tb.register("locate", (
        "Find a thing on the desk with the desk camera and SAM 3.1 (about $0.0025): where it is in the arm's frame in "
        "mm, its length and width, the heading of its long axis and its top's height, best match first. One camera "
        "can't see height, so give height_mm for anything standing up."), thing_params, label="SAM")
    def locate(thing: str, height_mm: float = 0) -> dict:
        from wbc.motion.hand import Thing

        if bench.sim:
            scene = json.loads((bench.root / "fixtures" / "desk_scene.json").read_text(encoding="utf-8"))
            result = {"sim": True, "found": [s for name, s in scene.items() if name in key(thing)]}
        else:
            result = _locate_live(bench, thing, height_mm)
        if result["found"]:
            bench.seen[key(thing)] = Thing.from_summary({**result["found"][0], "phrase": key(thing)})
        return {"ok": True, "thing": thing, **result}

    def target(thing: str, height_mm: float):
        """(Thing, None) from what locate last found, locating it first if it hasn't; (None, error) when it can't."""
        if key(thing) not in bench.seen and not locate(thing, height_mm)["found"]:
            return None, {"ok": False, "error": f"locate found no {thing} on the desk"}
        return bench.seen[key(thing)], None

    def move(question: str, run) -> dict:
        """run(hand) under the arm lock once the user presses Enter, powering the arm on first if it's off. The
        driver's refusals and trips come back as errors."""
        if not bench.arm_port:
            return {"ok": False, "error": "the arm's tools are off: start the agent with --arm"}
        if bench.confirm is None:
            return {"ok": False, "error": "there's no keyboard to press Enter on, so the arm stays still"}
        from wbc.motion.arm import Refused, Tripped

        with bench.arm_lock:
            try:
                hand = _open_hand(bench)
                off = len(hand.arm.powered) < len(hand.arm.joints.servos)
                if not bench.confirm(question + (" The motors come on first." if off else "")):
                    return {"ok": False, "error": "the user didn't press Enter, so the arm didn't move"}
                if off:
                    hand.power_on()
                return run(hand)
            except Refused as e:
                return {"ok": False, "error": f"refused, nothing moved for it: {e}"}
            except Tripped as e:
                return {"ok": False, "error": f"the motors were cut: {e}", "motors_cut": True}

    def check_over():
        """What pick_up and put_down ask over the thing before going down, while robot.yaml's hand.check_over is on."""
        return bench.confirm if (bench.robot_cfg.get("hand") or {}).get("check_over", True) else None

    @tb.register("pick_up", (
        "Pick a thing up off the desk with the arm, where locate last found it (locating it first if it hasn't). The "
        "user presses Enter before the arm moves. Returns what the jaws hold, or an error if the user said no, the arm "
        "can't reach it, or the jaws came up empty. Call arm tools one at a time, never two in one round."),
        thing_params, label="arm")
    def pick_up(thing: str, height_mm: float = 0) -> dict:
        t, problem = target(thing, height_mm)
        if problem:
            return problem
        if bench.sim:
            bench.sim_held = t.name
            return {"ok": True, "sim": True, "holding": t.name, "x_mm": t.x, "y_mm": t.y}
        return move(f"Pick up the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm?",
                    lambda hand: hand.pick_up(t, confirm=check_over()))

    @tb.register("put_down", (
        "Put what the arm holds down on or in a thing on the desk, such as a tray, where locate last found it "
        "(locating it first if it hasn't). The user presses Enter before the arm moves. Call it only after pick_up "
        "returned what the jaws hold."), thing_params, label="arm")
    def put_down(thing: str, height_mm: float = 0) -> dict:
        t, problem = target(thing, height_mm)
        if problem:
            return problem
        if bench.sim:
            if not bench.sim_held:
                return {"ok": False, "error": "refused, nothing moved for it: the jaws hold nothing to put down"}
            held, bench.sim_held = bench.sim_held, None
            return {"ok": True, "sim": True, "put": held, "on": t.name}
        return move(f"Put it down on the {t.name} at x {t.x:.0f}, y {t.y:.0f} mm?",
                    lambda hand: hand.put_down(t, confirm=check_over()))

    @tb.register("arm_home", "Move the arm back to where it rested when its motors came on. The user presses Enter "
                             "first.", {"type": "object", "properties": {}}, label="arm")
    def arm_home() -> dict:
        if bench.sim:
            return {"ok": True, "sim": True, "home": True}

        def home(hand):
            hand.home()
            return {"ok": True, "home": True}

        return move("Move the arm back to where it rested?", home)

    return tb
