from types import SimpleNamespace

import cv2
import numpy as np

from airo.floorwatch import FloorWatch

FLOOR = (0.8, 1.22)
PX = 800
H = np.diag([1 / PX, 1 / PX, 1.0])
TEXTURE = cv2.GaussianBlur(np.random.default_rng(0).integers(90, 170, (1080, 1920), dtype=np.uint8), (0, 0), 3)


def frame(car, carton=None, gain=1.0):
    img = TEXTURE.astype(np.float32) * gain
    cv2.circle(img, (round(car[0] * PX), round(car[1] * PX)), round(0.1 * PX), 20, -1)
    if carton is not None:
        (x0, y0), (x1, y1) = (np.array(carton) - 0.05) * PX, (np.array(carton) + 0.05) * PX
        cv2.rectangle(img, (round(x0), round(y0)), (round(x1), round(y1)), 250, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def at(x, y):
    return SimpleNamespace(x=x, y=y)


def test_floor_the_car_leaves_is_not_an_obstacle():
    w = FloorWatch(FLOOR)
    w.update(frame((0.4, 0.4)), H, at(0.4, 0.4))
    w.update(frame((0.4, 0.9)), H, at(0.4, 0.9))
    assert not w.polygons


def test_a_dropped_carton_is_found_and_blocks_the_path_through_it():
    w = FloorWatch(FLOOR)
    w.update(frame((0.4, 0.3)), H, at(0.4, 0.3))
    w.update(frame((0.4, 0.3), carton=(0.4, 0.6)), H, at(0.4, 0.3))
    assert len(w.polygons) == 1
    assert np.allclose(w.polygons[0].mean(axis=0), (0.4, 0.6), atol=0.02)
    hit = w.ahead([(0.4, 0.3), (0.4, 1.0)], reach=0.35, radius=0.12)
    assert hit is not None and 0.4 < hit[1] < 0.56
    assert w.ahead([(0.4, 0.3), (0.2, 0.3)], reach=0.35, radius=0.12) is None
    assert w.scene().grid.any()


def test_the_light_changing_is_not_an_obstacle():
    w = FloorWatch(FLOOR)
    w.update(frame((0.4, 0.4)), H, at(0.4, 0.4))
    w.update(frame((0.4, 0.4), gain=1.25), H, at(0.4, 0.4))
    assert not w.polygons
