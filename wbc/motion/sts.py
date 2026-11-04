"""Minimal Feetech STS3215 client: ping a servo and read its registers over the 1 Mbps half-duplex bus.

LeRobot's bus does calibration and multi-servo sync writes. This module exists so a probe can read one
servo a few hundred times a second without depending on LeRobot's API, which moved between releases.

Register addresses follow the STS series memory table, the same one LeRobot uses. Multi-byte values
are little-endian.
"""
from __future__ import annotations

# EEPROM registers (kept through power cycles)
MIN_POSITION_LIMIT = 9    # 2 bytes. lerobot-calibrate writes each joint's recorded range here and at 11,
MAX_POSITION_LIMIT = 11   # 2 bytes, so the servo won't drive past either end

# RAM registers
TORQUE_ENABLE = 40    # 1 byte
FEEDBACK_START = 56   # Present_Position (56) through Present_Current (69-70), read in one transaction

PING, READ = 0x01, 0x02


class STSError(IOError):
    pass


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    body = bytes([servo_id, len(params) + 2, instruction]) + params
    return b"\xff\xff" + body + bytes([checksum(body)])


class Bus:
    def __init__(self, port: str, baudrate: int = 1_000_000, timeout: float = 0.02, serial_factory=None):
        if serial_factory is None:
            try:
                import serial
            except ImportError:
                raise RuntimeError("The arm extras are missing: pip install -e '.[arm]'") from None
            # exclusive: a second process on the same half-duplex bus corrupts every reply
            serial_factory = lambda p, **kw: serial.Serial(p, exclusive=True, **kw)  # noqa: E731
        self.ser = serial_factory(port, baudrate=baudrate, timeout=timeout)

    def close(self) -> None:
        self.ser.close()

    def ping(self, servo_id: int) -> bool:
        try:
            self._transact(servo_id, PING)
            return True
        except STSError:
            return False

    def read(self, servo_id: int, address: int, length: int) -> tuple:
        """Returns (error_flags, data)."""
        err, data = self._transact(servo_id, READ, bytes([address, length]))
        if len(data) != length:
            raise STSError(f"servo {servo_id}: asked for {length} bytes at {address}, got {len(data)}")
        return err, data


    def read_u8(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[1][0]

    def read_u16(self, servo_id: int, address: int) -> int:
        _, data = self.read(servo_id, address, 2)
        return data[0] | (data[1] << 8)


    def _transact(self, servo_id: int, instruction: int, params: bytes = b"") -> tuple:
        sent = packet(servo_id, instruction, params)
        self.ser.reset_input_buffer()
        self.ser.write(sent)
        body = self._read_packet()
        if b"\xff\xff" + body + bytes([checksum(body)]) == sent:  # some adapters echo the request
            body = self._read_packet()
        if body[0] != servo_id:
            raise STSError(f"reply came from servo {body[0]}, expected {servo_id}")
        return body[2], bytes(body[3:])

    def _read_packet(self) -> bytes:
        """Find the FF FF header, then return id, length, error/instruction and params (checksum verified)."""
        prev = None
        while True:
            b = self.ser.read(1)
            if not b:
                raise STSError("no reply: check the port, servo power and servo id")
            if prev == 0xFF and b[0] == 0xFF:
                break
            prev = b[0]
        head = self._read_exact(2)
        if head[1] < 2:  # every reply carries at least an error byte and a checksum
            raise STSError("reply garbled: impossible length")
        rest = self._read_exact(head[1])
        body = head + rest[:-1]
        if checksum(body) != rest[-1]:
            raise STSError("reply failed its checksum")
        return body

    def _read_exact(self, n: int) -> bytes:
        data = self.ser.read(n)
        if len(data) != n:
            raise STSError(f"reply cut short: wanted {n} bytes, got {len(data)}")
        return data
