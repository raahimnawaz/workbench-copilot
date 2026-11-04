"""The STS3215 client, against a fake servo that answers like the real one."""
import pytest

from wbc.motion import sts


class FakeServo:
    """One STS3215 on the bus: answers PING, READ and WRITE from a register map."""

    def __init__(self, servo_id=6, echo=False, corrupt=False, silent=False):
        self.id, self.echo, self.corrupt, self.silent = servo_id, echo, corrupt, silent
        self.regs = bytearray(128)
        self.error = 0
        self.out = bytearray()
        self.drop = 0          # replies still to lose, one per request

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
        if self.drop:
            self.drop -= 1
            return
        reply = b""
        if instr == sts.READ:
            addr, n = params
            reply = bytes(self.regs[addr:addr + n])
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


def test_packet_matches_the_protocol():
    # read 2 bytes at address 0x38 from servo 1: FF FF 01 04 02 38 02 BE
    assert sts.packet(1, sts.READ, bytes([0x38, 2])) == bytes.fromhex("ffff0104023802be")


def test_echoing_adapter_is_tolerated():
    servo = FakeServo(echo=True)
    servo.set_u16(56, 1234)
    assert bus_for(servo).read_u16(6, sts.FEEDBACK_START) == 1234


@pytest.mark.parametrize("kind, match", [("silent", "no reply"), ("corrupt", "checksum")])
def test_bad_replies_raise(kind, match):
    servo = FakeServo(**{kind: True})
    with pytest.raises(sts.STSError, match=match):
        bus_for(servo).read_u16(6, sts.FEEDBACK_START)


def test_ping():
    assert bus_for(FakeServo()).ping(6)
    assert not bus_for(FakeServo()).ping(3)
