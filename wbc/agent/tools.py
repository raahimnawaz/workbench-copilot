"""Tools the router can call. Each returns a dict with "ok"; failures come back as data, never as exceptions.

Physical and camera tools are stubs until their sheet-10 day. With --sim, read_breadboard returns a
recorded scene and point_at pretends to move, so the router and coder can be tested with no hardware.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import coder

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
        "Move the wrist camera over the breadboard, segment the wires with SAM 3.1, and return the netlist, "
        "any mismatches against pins.h, and an annotated image."), {
        "type": "object", "properties": {}}, label="SAM")
    def read_breadboard() -> dict:
        if bench.sim:
            return json.loads((bench.root / "fixtures" / "breadboard_mismatch.json").read_text(encoding="utf-8"))
        return _not_wired("read_breadboard", "D3")

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
        "properties": {"target": {"type": "string", "description": "e.g. 'c41', 'row:41', 'wire:7'"},
                       "hover_mm": {"type": "number", "minimum": 4, "maximum": 30, "default": 8}},
        "required": ["target"]}, label="arm")
    def point_at(target: str, hover_mm: float = 8) -> dict:
        if bench.sim:
            return {"ok": True, "sim": True, "target": target, "hover_mm": hover_mm, "tip_error_mm": 0.0}
        return _not_wired("point_at", "D5")

    return tb
