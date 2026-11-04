"""Hardware check: BLE connects, the car drives straight then pivots, link stats are printed."""

import argparse
import asyncio
import statistics
import time

from airo.car import Car

HZ = 20


async def hold(car: Car, pwm_l: int, pwm_r: int, seconds: float) -> tuple[int, int]:
    sent = ticks = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sent += car.drive(pwm_l, pwm_r)
        ticks += 1
        await asyncio.sleep(1 / HZ)
    return sent, ticks


async def main(args):
    async with await Car.connect(args.address) as car:
        n0 = car.sens_count
        await asyncio.sleep(2)
        print(f"sens {(car.sens_count - n0) / 2:.1f} Hz, last {car.sens}")

        sent = ticks = 0
        for name, l, r in [("straight", args.pwm, args.pwm), ("pause", 0, 0),
                           ("pivot", args.pwm, -args.pwm), ("pause", 0, 0)]:
            print(name)
            s, t = await hold(car, l, r, args.seconds)
            sent, ticks = sent + s, ticks + t

        ack = sorted(car.ack_ms)
        print(f"ack median {statistics.median(ack):.0f} ms, p95 {ack[int(len(ack) * 0.95)]:.0f} ms, "
              f"sent {sent}/{ticks} at {HZ} Hz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--address")
    p.add_argument("--pwm", type=int, default=80)
    p.add_argument("--seconds", type=float, default=1.0)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        pass
