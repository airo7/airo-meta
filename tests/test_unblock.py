import math

import numpy as np
import pytest

from airo.mission import PUSH_MAX, PUSH_M, STANDOFF, push_plan

START, GOAL = (1.0, 0.2), (1.0, 1.2)


def box(cx, cy, s):
    h = s / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


def test_pushes_to_the_side_with_room():
    plan = push_plan(box(1.0, 0.7, 0.1), START, GOAL, lambda x, y: True,
                     lambda xy: 0.3 if xy[0] > 1.0 else 0.01)
    assert plan is not None and plan["n"][0] > 0.99
    assert plan["push_from"][0] < 1.0 < plan["push_to"][0]
    assert math.dist(plan["push_from"], plan["push_to"]) == pytest.approx(PUSH_M)


def test_a_box_bigger_than_push_max_is_not_pushed():
    assert push_plan(box(1.0, 0.7, PUSH_MAX + 0.05), START, GOAL,
                     lambda x, y: True, lambda xy: 0.3) is None


def test_no_side_with_room_is_not_pushed():
    assert push_plan(box(1.0, 0.7, 0.1), START, GOAL,
                     lambda x, y: math.dist((x, y), (1.0, 0.7)) < 0.1, lambda xy: 0.3) is None


def test_standoff_grows_with_size():
    small = push_plan(box(1.0, 0.7, 0.08), START, GOAL, lambda x, y: True, lambda xy: 0.3)
    big = push_plan(box(1.0, 0.7, 0.16), START, GOAL, lambda x, y: True, lambda xy: 0.3)
    assert small is not None and big is not None
    assert math.dist(big["push_from"], big["centroid"]) > math.dist(small["push_from"], small["centroid"])
    assert math.dist(small["push_from"], small["centroid"]) == pytest.approx(STANDOFF + (0.08 - 0.1) / 2)
