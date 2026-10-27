"""Setting servo ids on an assembled arm, against a fake line of factory-fresh servos."""
import importlib.util

import pytest

from wbc.config import ROOT
from wbc.motion import sts

_spec = importlib.util.spec_from_file_location("set_servo_ids", ROOT / "scripts" / "set_servo_ids.py")
set_servo_ids = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(set_servo_ids)


class FakeLine:
    """A servo bus with several servos, each plugged in or not. Servos sharing an id all answer at once,
    so their replies collide and nothing parses, like on the real arm."""

    def __init__(self, n=6):
        self.servos = [{"id": 1, "regs": bytearray(128), "on": False} for _ in range(n)]
        for s in self.servos:
            s["regs"][sts.LOCK] = 1  # the EEPROM is locked at power-up
        self.servos[0]["on"] = True  # only the base is connected to start with
        self.out = bytearray()

    def ids(self):
        return [s["id"] for s in self.servos]

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, data):
        sid, instr, params = data[2], data[4], data[5:-1]
        hits = [s for s in self.servos if s["on"] and s["id"] == sid]
        if len(hits) > 1:
            self.out += b"\xff\xff\x01\x02\x00\x00"  # two replies on top of each other: the checksum fails
            return
        if not hits:
            return
        s, reply = hits[0], b""
        if instr == sts.READ:
            addr, n = params
            reply = bytes(s["regs"][addr:addr + n])
        elif instr == sts.WRITE:
            addr, payload = params[0], bytes(params[1:])
            if addr == sts.ID:
                assert s["regs"][sts.LOCK] == 0, "id written while the EEPROM was locked"
                s["id"] = payload[0]  # and it replies under the new id
            s["regs"][addr:addr + len(payload)] = payload
        body = bytes([s["id"], len(reply) + 2, 0]) + reply
        self.out += b"\xff\xff" + body + bytes([sts.checksum(body)])

    def read(self, n):
        chunk = bytes(self.out[:n])
        del self.out[:n]
        return chunk

    def close(self):
        pass


def bus_on(line):
    return sts.Bus("fake", serial_factory=lambda port, **kw: line)


def plug_next(line, extra=0):
    """What Raahim does at each prompt: plug in the next servo (and, by mistake, `extra` more)."""
    off = [s for s in line.servos if not s["on"]]
    for s in off[:1 + extra]:
        s["on"] = True


def test_set_id_unlocks_writes_and_relocks():
    line = FakeLine(1)
    bus = bus_on(line)
    bus.set_id(1, 7)
    assert line.ids() == [7] and line.servos[0]["regs"][sts.LOCK] == 1
    assert bus.ping(7) and not bus.ping(1)


def test_set_id_refuses_an_id_that_is_taken():
    line = FakeLine(2)
    line.servos[1].update(id=2, on=True)
    with pytest.raises(sts.STSError, match="already answers"):
        bus_on(line).set_id(1, 2)
    assert line.ids() == [1, 2]


def test_whole_arm_gets_ids_one_to_six_in_chain_order():
    line = FakeLine()
    said = []
    ok = set_servo_ids.run(bus_on(line), ask=lambda p: plug_next(line) or "", say=said.append)
    assert ok and line.ids() == [1, 2, 3, 4, 5, 6]
    assert all(s["regs"][sts.LOCK] == 1 for s in line.servos)
    text = "\n".join(said)
    assert "at the elbow end" in text and "into the gripper servo" in text
    assert "Unplug the cable from the gripper" not in text  # the gripper is the end of the chain


def test_two_new_servos_at_once_are_caught_and_retried():
    line = FakeLine()
    said, calls = [], {"n": 0}

    def ask(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            plug_next(line, extra=1)  # forgot to unplug the cable leading on
        elif calls["n"] == 2:
            line.servos[2]["on"] = False  # unplugged it after the warning
        else:
            plug_next(line)
        return ""

    assert set_servo_ids.run(bus_on(line), ask=ask, say=said.append)
    assert line.ids() == [1, 2, 3, 4, 5, 6]
    assert any("two new servos at once" in s for s in said)


def test_stopping_partway_and_running_again_carries_on():
    line = FakeLine()
    answers = iter(["", "", "q"])
    bus = bus_on(line)

    def ask(prompt):
        a = next(answers)
        if a != "q":
            plug_next(line)
        return a

    assert not set_servo_ids.run(bus, ask=ask, say=lambda s: None)
    assert line.ids()[:3] == [set_servo_ids.PARK_ID, 2, 3]
    assert set_servo_ids.run(bus, ask=lambda p: plug_next(line) or "", say=lambda s: None)
    assert line.ids() == [1, 2, 3, 4, 5, 6]


def test_refuses_to_start_with_several_servos_on_the_line():
    line = FakeLine()
    line.servos[1]["on"] = True  # the base -> shoulder lift cable is still plugged in
    said = []
    assert not set_servo_ids.run(bus_on(line), ask=lambda p: "", say=said.append)
    assert "Unplug the base -> shoulder lift cable" in said[0] and line.ids() == [1] * 6


def test_a_finished_arm_is_left_alone():
    line = FakeLine()
    for i, s in enumerate(line.servos, 1):
        s.update(id=i, on=True)
    said = []
    assert set_servo_ids.run(bus_on(line), ask=lambda p: pytest.fail("shouldn't ask"), say=said.append)
    assert "Nothing to do" in said[0]
