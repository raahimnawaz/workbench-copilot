"""Read the force sensor sampler (firmware/fsr_sampler) over USB serial.

Frame, 18 bytes, little-endian:
    A5 5A | seq u16 | t_us u32 | mV u16 x 4 | CRC-16/CCITT-FALSE u16 over seq..mV
The four channels are the jaw sensors, in CHANNELS order. The parser skips anything that isn't a valid
frame (such as the ESP32's boot messages), counts bad CRCs, and counts gaps in seq as dropped frames.
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

SYNC = b"\xa5\x5a"
BODY = struct.Struct("<HI4H")
FRAME_LEN = len(SYNC) + BODY.size + 2  # 18
CHANNELS = ("SD", "SP", "MD", "MP")    # static jaw distal, proximal; moving jaw distal, proximal
PINS = {"SD": 32, "SP": 33, "MD": 34, "MP": 35}
BAUD = 921_600
RATE_HZ = 500


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection. Matches the firmware."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else crc << 1
        crc &= 0xFFFF
    return crc


def encode(seq: int, t_us: int, mv) -> bytes:
    """One frame as the firmware sends it. Used by the tests."""
    body = BODY.pack(seq & 0xFFFF, t_us & 0xFFFFFFFF, *mv)
    return SYNC + body + struct.pack("<H", crc16(body))


@dataclass
class Frame:
    seq: int
    t_us: int            # the sampler's clock; wraps every 71.6 minutes
    mv: tuple            # one reading per channel, in CHANNELS order
    host_t: float = 0.0  # when the host received it (time.monotonic)


class Parser:
    """Turns a byte stream into frames."""

    def __init__(self):
        self.buf = bytearray()
        self.frames = 0
        self.bad_crc = 0
        self.dropped = 0
        self.restarts = 0
        self._last_seq = None

    def feed(self, data: bytes, host_t: float = 0.0) -> list:
        self.buf += data
        out = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                del self.buf[:-1]  # the last byte may be the start of the next sync
                return out
            del self.buf[:i]
            if len(self.buf) < FRAME_LEN:
                return out
            body = bytes(self.buf[2:2 + BODY.size])
            (crc,) = struct.unpack_from("<H", self.buf, 2 + BODY.size)
            if crc16(body) != crc:
                self.bad_crc += 1
                del self.buf[:1]  # a false sync or a damaged frame: look again from the next byte
                continue
            del self.buf[:FRAME_LEN]
            seq, t_us, *mv = BODY.unpack(body)
            if self._last_seq is not None:
                gap = (seq - self._last_seq - 1) & 0xFFFF
                if gap > 0x8000:
                    self.restarts += 1  # the sampler reset and started counting again
                else:
                    self.dropped += gap
            self._last_seq = seq
            self.frames += 1
            out.append(Frame(seq, t_us, tuple(mv), host_t))


def unwrap_seconds(t_us) -> np.ndarray:
    """The sampler's microsecond clock as seconds from the first frame, with wraparounds removed."""
    t = np.asarray(t_us, dtype=np.int64)
    if t.size == 0:
        return np.zeros(0)
    steps = np.diff(t)
    steps[steps < 0] += 1 << 32
    return np.concatenate([[0], np.cumsum(steps)]) / 1e6


class Sampler:
    """Opens the sampler's port and collects frames on a background thread."""

    def __init__(self, port: str, baudrate: int = BAUD, serial_factory=None, clock=time.monotonic):
        if serial_factory is None:
            try:
                import serial
            except ImportError:
                raise RuntimeError("The arm extras are missing: pip install -e '.[arm]'") from None
            serial_factory = lambda p, **kw: serial.Serial(p, exclusive=True, **kw)  # noqa: E731
        self.ser = serial_factory(port, baudrate=baudrate, timeout=0.05)
        self.parser = Parser()
        self.clock = clock
        self.frames: list[Frame] = []  # 500 Hz for a few minutes is well under a million frames
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "Sampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                data = self.ser.read(self.ser.in_waiting or 1)
                if data:
                    frames = self.parser.feed(data, self.clock())
                    if frames:
                        with self._lock:
                            self.frames.extend(frames)
        except Exception as e:  # surfaced to the caller through wait_for and check
            self.error = e

    def check(self) -> None:
        if self.error is not None:
            raise IOError(f"sampler connection failed: {self.error}")

    def count(self) -> int:
        with self._lock:
            return len(self.frames)

    def latest(self) -> Frame | None:
        with self._lock:
            return self.frames[-1] if self.frames else None

    def window(self, t0: float, t1: float) -> list:
        """Frames received between host times t0 and t1."""
        with self._lock:
            return [f for f in self.frames if t0 <= f.host_t < t1]

    def wait_for(self, n: int = 1, timeout: float = 3.0, sleep=time.sleep) -> bool:
        end = self.clock() + timeout
        while self.count() < n:
            self.check()
            if self.clock() >= end:
                return False
            sleep(0.01)
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.ser.close()
