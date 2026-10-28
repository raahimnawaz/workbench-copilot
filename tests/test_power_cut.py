"""The power cut-off test, against a fake follower whose supply can be switched off."""
import importlib.util
import re

import pytest

from test_first_moves import JOINTS, TPD, Clock, FakeArm, FakeBus
from wbc.config import ROOT

_spec = importlib.util.spec_from_file_location("power_cut", ROOT / "scripts" / "power_cut.py")
power_cut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(power_cut)

WRITES = ("goal ", "gain ", "cap ", "torque on", "all torque off")  # events for every write the fake records
POSE = {"shoulder_lift": -60.0, "elbow_flex": 40.0, "wrist_flex": 10.0}


class PoweredBus(FakeBus):
    """FakeBus behind the power strip: nothing answers while the servos are unpowered, and a broadcast sent
    then goes nowhere."""

    def sync_read(self, name, motors=None, *, normalize=True, num_retry=0):
        arm = self.arm
        arm.tick()
        if arm.reads in arm.lost or not arm.powered:
            raise ConnectionError("There is no status packet!")
        if name == "Present_Voltage":
            return {m: round(10 * arm.volts) for m in JOINTS}
        if name == "Max_Torque_Limit":
            return dict(arm.max_limit)
        if name == "P_Coefficient":
            return dict(arm.gain)
        if name in ("Present_Temperature", "Status", "Lock"):
            return {m: {"Present_Temperature": 28, "Status": 0, "Lock": 1}[name] for m in JOINTS}
        return super().sync_read(name, motors, normalize=normalize, num_retry=num_retry)

    def sync_write(self, name, values, *, normalize=True, num_retry=0):
        if self.arm.powered:
            super().sync_write(name, values, normalize=normalize, num_retry=num_retry)
        else:
            self.arm.events.append("all torque off, unheard")

    def ping(self, motor, num_retry=0, raise_on_error=False):
        self.arm.tick()
        return 777 if self.arm.powered else None


class PoweredArm(FakeArm):
    """FakeArm on a supply behind the power strip. Switched off, the voltage holds for `holdup` reads, then
    falls by `fall` volts a read, down to `floor`, and the servos stop answering below 4.0 V. Once it's under
    4.8 V, each joint in `falls` that had torque on drops by its rate (degrees) a read. Switched on, the servos
    start up the way the real ones do: torque off unless listed in `wakes_on`, Torque_Limit back to
    Max_Torque_Limit, and every goal 0."""

    def __init__(self, holdup=10, fall=0.1, floor=0.0, falls=None, wakes_on=(), **kw):
        super().__init__(**kw)
        self.bus = PoweredBus(self)
        self.volts, self.reads, self.off_at, self.lost = 5.3, 0, None, set()
        self.holdup, self.fall, self.floor = holdup, fall, floor
        self.falls, self.wakes_on = falls or {}, set(wakes_on)
        self.max_limit = {**{m: 1000 for m in JOINTS}, "gripper": 500}
        self.torque_limit = dict(self.max_limit)

    @property
    def powered(self):
        return self.volts >= 4.0

    def switch_off(self):
        self.off_at = self.reads
        self.events.append("switched off")

    def switch_on(self):
        self.off_at, self.volts = None, 5.3
        self.torque = {m: m in self.wakes_on for m in JOINTS}
        self.torque_limit = dict(self.max_limit)
        self.goal_raw = {m: 0 for m in JOINTS}
        self.events.append("switched on")

    def tick(self):
        self.reads += 1
        if self.off_at is None or self.reads - self.off_at <= self.holdup:
            return
        self.volts = max(self.floor, self.volts - self.fall)
        if self.volts < 4.8:
            for m, rate in self.falls.items():
                if self.torque[m]:
                    self.pos[m] += rate


class Bench:
    """You at the bench: you do `cut` (switch the strip off, unless a test says otherwise) when the script
    asks you to switch it off, switch it back on while the script waits for the power, and answer prompts
    with `replies` in order. A reply can be a function, which runs as you'd act at that prompt."""

    def __init__(self, arm, replies=(), cut=None):
        self.arm, self.replies, self.cut, self.said = arm, list(replies), cut or arm.switch_off, []

    def say(self, text):
        self.said.append(text)
        if "switch the strip off" in text.lower():
            self.cut()

    def ask(self, prompt):
        self.arm.events.append(f"ask: {prompt.strip()}")
        if not self.replies:
            raise EOFError
        reply = self.replies.pop(0)
        return reply() if callable(reply) else reply

    def sleep(self, seconds):
        if not self.arm.powered:
            self.arm.switch_on()

    def text(self):
        return "\n".join(self.said)


def idle(arm, **kw):
    you, record = Bench(arm, **kw), {}
    ok = power_cut.run_idle(arm, record, say=you.say, sleep=you.sleep, clock=Clock())
    return ok, record, you


def settles(arm, degrees=3.0):
    """A reply for the let-go prompt: you let go, and the shoulder sinks as its motor takes the weight."""
    def let_go():
        arm.pos["shoulder_lift"] -= degrees
        return ""
    return let_go


def hold(arm, replies=None, **kw):
    """The hold step, answering by default: motors on, it settles and holds on its own, and it let go at once."""
    replies = ("", settles(arm), "1") if replies is None else replies
    you, record = Bench(arm, replies=replies, **kw), {}
    ok = power_cut.run_hold(arm, record, ask=you.ask, say=you.say, sleep=you.sleep, clock=Clock())
    return ok, record, you


def test_idle_times_the_cut_and_finds_every_servo_limp_without_writing_anything():
    arm = PoweredArm()
    ok, record, you = idle(arm)
    cut, text = record["cut"], you.text()
    assert ok and 0 < cut["sag_at"] < cut["last_at"] < cut["sag_at"] + 0.02
    assert cut["volts_at_start"] == {m: 5.3 for m in JOINTS} and cut["glitches"] == 0
    assert arm.events == ["switched off", "switched on"]  # no write of any kind
    assert record["after"]["Torque_Enable"] == {m: 0 for m in JOINTS}
    assert "The strip cuts the servos' power." in text and "Torque came back off on all six" in text
    assert "Every goal came back as 0" in text and "Nothing moved" in text
    assert "The torque cap came back as it was" in text and "P gains came back as they were" in text


def test_a_servo_that_comes_back_with_torque_on_is_turned_off_and_reported():
    arm = PoweredArm(wakes_on={"elbow_flex"})
    ok, record, you = idle(arm)
    assert not ok and arm.events[-1] == "all torque off" and not any(arm.torque.values())
    assert "Torque came back ON for the elbow" in you.text()


def test_idle_wont_start_with_a_motor_on():
    arm = PoweredArm()
    arm.torque["wrist_flex"] = True
    ok, record, you = idle(arm)
    assert not ok and "cut" not in record and arm.events == []
    assert "Torque is on for the wrist flex" in you.text()


def test_replies_that_go_missing_once_arent_taken_for_the_cut():
    arm = PoweredArm(holdup=40)
    ok, record, you = idle(arm, cut=lambda: (arm.switch_off(), arm.lost.update({arm.reads + 5, arm.reads + 9})))
    assert ok and record["cut"]["glitches"] == 2 and record["cut"]["sag_at"] is not None
    assert "2 replies went missing before the cut" in you.text()


def test_a_strip_that_also_powers_the_usb_adapter_is_reported():
    arm = PoweredArm()

    def gone(*args, **kwargs):
        raise OSError(6, "Device not configured")

    def unplug():
        arm.switch_off()
        arm.bus.sync_read = gone

    ok, record, you = idle(arm, cut=unplug)
    assert not ok and "USB adapter went away too" in you.text()


def test_hold_records_when_the_arm_lets_go_and_finds_it_limp_after():
    arm = PoweredArm(start=POSE, falls={"shoulder_lift": 0.5, "elbow_flex": -0.3})
    ok, record, you = hold(arm)
    cut, text = record["cut"], you.text()
    when, joint = power_cut.let_go(cut["readings"], record["hold"])
    assert ok and cut["sag_at"] < when < cut["last_at"] and joint == "shoulder_lift"
    assert record["before"]["Torque_Limit"] == {m: 500 for m in JOINTS}
    assert record["before"]["P_Coefficient"]["elbow_flex"] == 32
    assert record["after"]["Torque_Enable"] == {m: 0 for m in JOINTS}
    assert "The strip stops the arm" in text and "after the voltage did, the shoulder lift first" in text
    assert "base 50% → 100%" in text and "gripper 50%" not in text
    assert record["let_go"] == "at once"
    off, on = arm.events.index("switched off"), arm.events.index("switched on")
    assert set(arm.events[off + 1:on]) == {"all torque off, unheard"}  # the cut, sent into the dead bus
    assert "all torque off" in arm.events[on:] and not any(e.startswith("torque on") for e in arm.events[on:])


def test_an_arm_with_no_weight_on_its_motors_is_flagged_and_its_run_doesnt_count():
    arm = PoweredArm(start=POSE)  # balanced: it doesn't sink as you let go, and nothing falls in the cut
    ok, record, you = hold(arm, replies=("", "", ""))  # motors on, let go, cut the power anyway
    prompts = [e for e in arm.events if e.startswith("ask:")]
    assert "It didn't sink as you let go" in prompts[2] and len(prompts) == 3  # and no let-go question after
    assert not ok and "switched on" in arm.events and "let_go" not in record
    assert "this run couldn't show it letting go" in you.text() and "The strip stops the arm" not in you.text()


def test_q_at_the_no_weight_warning_turns_the_motors_off_once_youre_holding_the_arm():
    arm = PoweredArm(start=POSE)
    ok, record, you = hold(arm, replies=("", "", "q", ""))
    hold_prompt = max(i for i, e in enumerate(arm.events) if e.startswith("ask: Hold the arm"))
    assert not ok and "cut" not in record and "switched off" not in arm.events
    assert arm.events[hold_prompt + 1] == "all torque off" and not any(arm.torque.values())


def test_a_joint_that_moves_with_the_power_on_cuts_every_motor_at_once():
    arm = PoweredArm(start=POSE)
    ok, record, you = hold(arm, cut=lambda: arm.runaway.update(elbow_flex=0.3))
    assert not ok and re.search(r"the elbow moved \+5\.\d° with the power still on", record["stopped"])
    assert "all torque off" in arm.events and not any(arm.torque.values())
    assert "cut" not in record and "switched off" not in arm.events


def test_an_arm_that_cant_hold_its_pose_is_cut_before_the_test():
    arm = PoweredArm(start=POSE)
    ok, record, you = hold(arm, replies=("", settles(arm, 20.0)))  # the shoulder can't take the weight
    assert not ok and "shoulder lift sank -20.0° as you let go" in record["stopped"]
    assert "all torque off" in arm.events and "cut" not in record


def test_servos_that_outlast_the_grace_are_cut_by_the_script():
    arm = PoweredArm(start=POSE, floor=4.9)
    arm.torque = {m: True for m in JOINTS}
    held = arm.bus.sync_read("Present_Position", normalize=False)
    arm.switch_off()
    clock = Clock()

    def later():
        t = Clock.__call__(clock)
        if t > 3.0:
            arm.floor = 0.0  # the supply gives out in the end
        return t

    cut = power_cut.wait_for_cut(arm, hold=held, clock=later)
    assert cut["cut_at"] - cut["sag_at"] == pytest.approx(power_cut.GRACE, abs=0.01)
    assert cut["last_at"] > cut["cut_at"] and "all torque off" in arm.events
    assert any("so this cut every motor itself" in line for line in power_cut.describe(cut, held))


def test_ctrl_c_before_the_cut_lets_the_arm_go_only_once_youre_holding_it():
    arm = PoweredArm(start=POSE)

    def interrupt():
        raise KeyboardInterrupt

    ok, record, you = hold(arm, replies=("", settles(arm), ""), cut=interrupt)
    hold_prompt = max(i for i, e in enumerate(arm.events) if e.startswith("ask: Hold the arm"))
    assert not ok and "cut" not in record and arm.events[hold_prompt + 1] == "all torque off"
    assert not any(arm.torque.values()) and arm.torque_limit == arm.max_limit  # the cap is lifted again


def test_let_go_finds_where_the_final_fall_began():
    held = {m: 2047 for m in JOINTS}

    def reading(t, **moved):
        return {"t": t, **{m: 2047 + round(moved.get(m, 0.0) * TPD) for m in JOINTS}}

    readings = [reading(0.0), reading(0.004, elbow_flex=1.5), reading(0.008),  # a bump that came back
                reading(0.012, wrist_flex=0.5), reading(0.016, wrist_flex=1.2, shoulder_lift=-1.1),
                reading(0.020, wrist_flex=2.5, shoulder_lift=-3.0)]
    assert power_cut.let_go(readings, held) == (0.016, "shoulder_lift")
    assert power_cut.let_go(readings[:3], held) == (None, None)
