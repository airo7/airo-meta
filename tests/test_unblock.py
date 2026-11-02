import math

import numpy as np
import pytest

from airo.mission import PUSH_M, PUSH_MAX, STANDOFF, push_out_plan, push_plan

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


def test_a_side_without_room_beyond_the_box_is_not_used():
    assert push_plan(box(1.0, 0.7, 0.1), START, GOAL, lambda x, y: True, lambda xy: 0.01) is None


def inset(x, y):
    return 0.15 <= x <= 0.65 and 0.15 <= y <= 1.07


def test_push_out_shoves_toward_the_nearest_edge_until_clear_of_the_driving_area():
    plan = push_out_plan(box(0.70, 0.28, 0.12), inset, (0.8, 1.22), 0.15)
    assert plan is not None and plan["n"] == (1.0, 0.0)
    assert 0.70 + plan["d"] - 0.06 >= 0.65
    assert all(inset(*plan[k]) for k in ("run_in", "push_from", "push_to"))


def test_push_out_from_mid_floor_goes_along_the_long_side():
    plan = push_out_plan(box(0.40, 0.45, 0.12), inset, (0.8, 1.22), 0.15)
    assert plan is not None and plan["n"] == (0.0, -1.0)
    assert all(inset(*plan[k]) for k in ("run_in", "push_from", "push_to"))


def test_push_out_refuses_a_big_box():
    assert push_out_plan(box(0.40, 0.45, PUSH_MAX + 0.05), inset, (0.8, 1.22), 0.15) is None
