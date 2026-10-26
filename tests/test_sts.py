"""The STS3215 client and the gripper probe, against a fake servo that answers like the real one."""
import pytest

from wbc.motion import probe, sts


class FakeServo:
    """One STS3215 on the bus: answers PING, READ and WRITE from a register map."""

    def __init__(self, servo_id=6, echo=False, corrupt=False, silent=False):
        self.id, self.echo, self.corrupt, self.silent = servo_id, echo, corrupt, silent
        self.regs = bytearray(128)
        self.error = 0
        self.writes = []
        self.out = bytearray()

    def set_u16(self, addr, value):
        self.regs[addr], self.regs[addr + 1] = value & 0xFF, value >> 8

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, data):
        assert data[:2] == b"\xff\xff" and sts.checksum(data[2:-1]) == data[-1]
        if self.echo:
            self.out += data
        sid, instr, params = data[2], data[4], data[5:-1]
        if self.silent or sid != self.id:
            return
        reply = b""
        if instr == sts.READ:
            addr, n = params
            reply = bytes(self.regs[addr:addr + n])
        elif instr == sts.WRITE:
            addr, payload = params[0], bytes(params[1:])
            self.regs[addr:addr + len(payload)] = payload
            self.writes.append((addr, payload))
        body = bytes([self.id, len(reply) + 2, self.error]) + reply
        pkt = b"\xff\xff" + body + bytes([sts.checksum(body) ^ (0xFF if self.corrupt else 0)])
        self.out += pkt

    def read(self, n):
        chunk = bytes(self.out[:n])
        del self.out[:n]
        return chunk

    def close(self):
        pass


def bus_for(servo):
    return sts.Bus("fake", serial_factory=lambda port, **kw: servo)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def test_packet_matches_the_protocol():
    # read 2 bytes at address 0x38 from servo 1: FF FF 01 04 02 38 02 BE
    assert sts.packet(1, sts.READ, bytes([0x38, 2])) == bytes.fromhex("ffff0104023802be")


def test_sign_magnitude():
    assert sts.sign_magnitude(250, 10) == 250
    assert sts.sign_magnitude(250 | 0x400, 10) == -250
    assert sts.sign_magnitude(100 | 0x8000, 15) == -100


def test_read_feedback_decodes_every_field():
    servo = FakeServo()
    servo.set_u16(56, 2048)
    servo.set_u16(58, 120 | 0x8000)
    servo.set_u16(60, 250 | 0x400)
    servo.regs[62], servo.regs[63], servo.regs[66] = 121, 41, 1
    servo.set_u16(69, 37)
    fb = bus_for(servo).read_feedback(6)
    assert (fb.position, fb.speed, fb.load, fb.voltage, fb.temperature, fb.moving, fb.current) == \
        (2048, -120, -250, 12.1, 41, True, 37)


def test_error_flags_come_back_with_the_data():
    servo = FakeServo()
    servo.error = 0x20  # overload
    assert bus_for(servo).read_feedback(6).error == 0x20


def test_writes_land_little_endian():
    servo = FakeServo()
    bus_for(servo).write_u16(6, sts.GOAL_POSITION, 0x0812)
    assert servo.regs[42:44] == b"\x12\x08"


def test_echoing_adapter_is_tolerated():
    servo = FakeServo(echo=True)
    servo.set_u16(56, 1234)
    assert bus_for(servo).read_feedback(6).position == 1234


@pytest.mark.parametrize("kind, match", [("silent", "no reply"), ("corrupt", "checksum")])
def test_bad_replies_raise(kind, match):
    servo = FakeServo(**{kind: True})
    with pytest.raises(sts.STSError, match=match):
        bus_for(servo).read_feedback(6)


def test_ping():
    assert bus_for(FakeServo()).ping(6)
    assert not bus_for(FakeServo()).ping(3)


def test_watch_never_writes():
    servo, clock = FakeServo(), FakeClock()
    rows, _ = probe.run_probe(bus_for(servo), 6, "watch", hz=100, seconds=0.5, clock=clock, sleep=clock.sleep)
    assert 45 <= len(rows) <= 51
    assert servo.writes == []


def test_squeeze_ramps_caps_torque_and_releases():
    servo, clock = FakeServo(), FakeClock()
    servo.set_u16(56, 2000)               # jaws start open at 2000
    servo.set_u16(sts.TORQUE_LIMIT, 1000)  # the power-up value, restored at the end
    rows, _ = probe.run_probe(bus_for(servo), 6, "squeeze", closed=2300, torque_limit=300, speed=300,
                              hz=100, seconds=2.0, clock=clock, sleep=clock.sleep)
    order = [addr for addr, _ in servo.writes]
    # goal set to the current position, then the torque cap, then torque on
    assert order[:3] == [sts.GOAL_POSITION, sts.TORQUE_LIMIT, sts.TORQUE_ENABLE]
    assert servo.writes[1][1] == (300).to_bytes(2, "little")
    goals = [r["goal"] for r in rows]
    assert goals == sorted(goals) and goals[-1] == 2300
    reached = next(r["t"] for r in rows if r["goal"] == 2300)
    assert 0.9 <= reached <= 1.1  # 300 ticks at 300 ticks/s
    assert servo.writes[-2] == (sts.TORQUE_ENABLE, b"\x00")
    assert servo.writes[-1] == (sts.TORQUE_LIMIT, (1000).to_bytes(2, "little"))


def test_squeeze_releases_torque_when_the_run_crashes():
    servo, clock = FakeServo(), FakeClock()
    calls = iter(range(1000))

    def poll():
        if next(calls) == 20:
            raise RuntimeError("boom")
        return []

    with pytest.raises(RuntimeError):
        probe.run_probe(bus_for(servo), 6, "squeeze", closed=2300, hz=100, poll_marks=poll,
                        clock=clock, sleep=clock.sleep)
    assert (sts.TORQUE_ENABLE, b"\x00") in servo.writes


@pytest.mark.parametrize("kwargs", [{}, {"closed": 2300, "torque_limit": 900}])
def test_squeeze_refuses_unsafe_settings(kwargs):
    with pytest.raises(ValueError):
        probe.run_probe(bus_for(FakeServo()), 6, "squeeze", **kwargs)


def test_marks_are_timestamped():
    servo, clock = FakeServo(), FakeClock()
    script = {30: ["pull"]}
    n = iter(range(1000))
    rows, marks = probe.run_probe(bus_for(servo), 6, "watch", hz=100, seconds=0.5,
                                  poll_marks=lambda: script.get(next(n), []), clock=clock, sleep=clock.sleep)
    assert marks == [(pytest.approx(0.30, abs=0.011), "pull")]
    assert rows[30]["mark"] == "pull"


def test_summary_separates_signal_from_noise():
    rows = []
    for i in range(1000):  # 5 s at 200 Hz
        t = i / 200
        jitter = 1 if i % 2 else -1
        rows.append({"t": t, "load": 120 + jitter + (40 if t >= 3.05 else 0),
                     "current": 30 + jitter, "position": 2210})
    lines = probe.summarize(rows, [(3.0, "pull")])
    assert "noise while still" in lines[1]
    pull = lines[-1]
    assert pull.startswith("'pull' at 3.00 s")
    assert "load +4" in pull and "seen" in pull.split(";")[0]
    assert "lost in noise" in pull.split(";")[1]


def test_summary_without_marks_says_how_to_add_them():
    rows = [{"t": i / 100, "load": 0, "current": 0, "position": 0} for i in range(50)]
    assert "no marks" in probe.summarize(rows, [])[-1]
