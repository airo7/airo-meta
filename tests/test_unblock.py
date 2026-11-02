import math

import numpy as np
import pytest

from airo import geometry
from airo.mission import OVERPUSH, PUSH_M, PUSH_MAX, STANDOFF, push_out_plan, push_plan, push_to_plan

START, GOAL = (1.0, 0.2), (1.0, 1.2)


def box(cx, cy, s):
    h = s / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])


def test_pushes_to_the_side_with_room():
    plan = push_plan(box(1.0, 0.7, 0.1), START, GOAL, lambda x, y: True,
                     lambda xy: 0.3 if xy[0] > 1.0 else 0.01)
    assert plan is not None and plan["n"][0] > 0.99
    assert plan["push_from"][0] < 1.0 < plan["push_to"][0]
    # the car stops where the box centre, half a car and half a box ahead, has moved PUSH_M
    reach = geometry.CAR[0] + 0.05 - OVERPUSH
    assert plan["push_to"][0] + reach == pytest.approx(1.0 + PUSH_M)


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


def test_push_out_from_mid_floor_takes_the_nearest_edge():
    plan = push_out_plan(box(0.40, 0.45, 0.12), inset, (0.8, 1.22), 0.15)
    assert plan is not None and abs(plan["n"][0]) == 1.0 and plan["d"] < 0.4
    assert all(inset(*plan[k]) for k in ("run_in", "push_from", "push_to"))


def test_push_out_refuses_a_big_box():
    assert push_out_plan(box(0.40, 0.45, PUSH_MAX + 0.05), inset, (0.8, 1.22), 0.15) is None


def test_push_to_a_point_lines_up_behind_the_box():
    plan = push_to_plan(box(0.40, 0.40, 0.12), (0.40, 0.80), inset)
    assert plan is not None and plan["n"] == pytest.approx((0.0, 1.0)) and plan["d"] == pytest.approx(0.40)
    assert plan["push_from"][1] < 0.40 < plan["push_to"][1] < 0.80


def test_push_to_needs_room_behind_the_box():
    # behind a box at the edge, pushing it inward, is off the driving area
    assert push_to_plan(box(0.67, 0.77, 0.12), (0.40, 0.61), inset) is None


def test_push_to_an_unreachable_place_goes_as_far_as_it_can():
    # "push it to marker 4": the corner itself is off the driving area
    plan = push_to_plan(box(0.45, 0.53, 0.12), (0.0, 1.22), inset)
    assert plan is not None and plan["short"] > 0.05
    assert all(inset(*plan[k]) for k in ("run_in", "push_from", "push_to"))
