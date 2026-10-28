import asyncio
import time

import cv2
import numpy as np
import pytest

from airo import rig as rig_mod
from airo.camera import CAR_ID, CAR_MARKER_M, CORNER_MARKER_M, floor_layout
from airo.car import Car, CarLost
from airo.rig import Rig, RigAbort
from tests.test_camera import H as IMG_H
from tests.test_camera import W as IMG_W
from tests.test_camera import stamp

FLOOR = (2.0, 1.2)


class FakeCap:
    def __init__(self, frame):
        self.frame = frame

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def grab(self):
        time.sleep(0.01)
        return True

    def retrieve(self):
        return True, self.frame

    def release(self):
        pass


def frame(ids):
    canvas = np.full((IMG_H, IMG_W), 255, np.uint8)
    for i, centre in floor_layout(*FLOOR).items():
        if i in ids:
            stamp(canvas, i, centre, CORNER_MARKER_M)
    if CAR_ID in ids:
        stamp(canvas, CAR_ID, (1.0, 0.6), CAR_MARKER_M)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def camera(monkeypatch):
    def use(ids):
        monkeypatch.setattr(rig_mod, "open_camera", lambda index: FakeCap(frame(ids)))
    return use


def test_names_the_missing_markers(camera):
    camera({1, 2, 0})
    with pytest.raises(RigAbort, match=r"seen \[0, 1, 2\], missing \[3, 4\]"):
        asyncio.run(Rig.open(FLOOR, timeout=0.5, log_dir=None))


def test_ble_not_found_is_a_rig_abort(camera, monkeypatch):
    async def no_car(address):
        raise CarLost("no BLE device named HC-05")

    camera({0, 1, 2, 3, 4})
    monkeypatch.setattr(Car, "connect", staticmethod(no_car))
    with pytest.raises(RigAbort, match="no BLE device"):
        asyncio.run(Rig.open(FLOOR, log_dir=None))


def test_dry_run_tracks_the_pose_without_a_car(camera, monkeypatch):
    async def run():
        rig = await Rig.open(FLOOR, dry=True, log_dir=None)
        await rig.step(150, 150)
        await rig.stop()
        await rig.close()
        return rig

    monkeypatch.setattr(Car, "connect", staticmethod(lambda address: pytest.fail("dry run connected")))
    camera({0, 1, 2, 3, 4})
    rig = asyncio.run(run())
    assert abs(rig.pose.x - 1.0) < 0.01 and len(rig.history[0]) == 0


class FakeClient:
    def __init__(self, fail):
        self.fail, self.is_connected, self.writes = fail, not fail, 0

    async def write_gatt_char(self, char, pkt, response):
        if self.fail:
            self.fail = False
            raise OSError("link dropped")
        self.writes += 1


def test_one_reconnect_then_car_lost():
    async def run(reconnects):
        car = Car("addr")
        car._client = FakeClient(fail=True)

        async def reopen(timeout=20):
            if not reconnects:
                raise CarLost("could not connect")
            car._client = FakeClient(fail=False)

        car._open = reopen
        car.drive(100, 100)
        await asyncio.gather(car._write, return_exceptions=True)
        return car.drive(0, 0), car._client.writes

    assert asyncio.run(run(True)) == (True, 1)
    with pytest.raises(CarLost, match="BLE link lost"):
        asyncio.run(run(False))
