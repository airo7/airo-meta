"""Byte layout of the car's pre-existing firmware (airo7/sim2real-robot-car, hardware/arduino/main/main.ino)."""

import time
from dataclasses import dataclass

CMD_HEADER = 0xAA
SENS_HEADER = 0xBB
SENS_LEN = 8
MODE_PWM = 1
PWM_MAX = 230


@dataclass(frozen=True)
class Sens:
    ir: tuple[int, int, int]
    dist_mm: int
    t: float


def cmd_packet(pwm_l: int, pwm_r: int) -> bytes:
    """Raw signed PWM per side; firmware decodes (byte - 128) * 2."""
    l, r = (min(255, max(0, round(128 + max(-PWM_MAX, min(PWM_MAX, p)) / 2))) for p in (pwm_l, pwm_r))
    return bytes([CMD_HEADER, l, r, MODE_PWM, 0, l ^ r ^ MODE_PWM])


def parse_sens(buf: bytearray) -> list[Sens]:
    """Consume complete SENS packets from buf in place, resyncing on garbage."""
    out = []
    while len(buf) >= SENS_LEN:
        if buf[0] != SENS_HEADER or buf[1] ^ buf[2] ^ buf[3] ^ buf[4] ^ buf[5] != buf[7]:
            del buf[0]
            continue
        out.append(Sens((buf[1], buf[2], buf[3]), buf[4] << 8 | buf[5], time.monotonic()))
        del buf[:SENS_LEN]
    return out
