import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from airo.body import Body, Segment, ensemble, save
from airo.memory import Memory, since_change
from airo.probes import LIBRARY, commands
from airo.rig import DT
from airo.scientist import Scientist
from airo.sim import SimRig
from test_scientist import HEAVY, TRUTH, drive, fake_wonder, results, teleport

MISSION = [(140, 140, 15), (80, 140, 20), (140, 80, 20)]
CHANGE_AT = 0.75


@pytest.fixture(scope="module")
def library():
    """Body #1 learned bare, Body #2 relearned after the weight went on."""
    with pytest.MonkeyPatch.context() as mp:
        fake_wonder(mp, [], fail_after=0)
        rig = SimRig(TRUTH, log_dir=None)
        sci = Scientist(rig, teleport, k=4)
        asyncio.run(sci.learn())
        rig.truth = HEAVY
        window = asyncio.run(changed(rig, HEAVY))
        asyncio.run(sci.relearn(window))
    assert [e.n for e in sci.memory.entries] == [1, 2] and sci.memory.entries[1].verified
    return sci.memory.entries


async def changed(rig, truth, before=None):
    """The mission's window; with before, the body changes from before to truth after the straight leg and the
    window is that leg plus a 0.75 s turn, as when the surprise fires soon after. The bare body drives much further, so it starts on the left."""
    rig.truth = before or truth
    rig._truth.append((rig.t, 0.5, 0.7, 0.0))
    await rig.step(0, 0)
    t0 = rig.t
    await drive(rig, MISSION[:1])
    t = rig.t
    rig.truth = truth
    await drive(rig, [(80, 140, 15)] if before else MISSION[1:])
    return rig.segment(t - CHANGE_AT if before else t0)


def scientist(library, monkeypatch, tmp_path, current: int, truth, wonder=()):
    """With wonder, Wonder answers those tool calls; otherwise it is offline and the scripted policy runs."""
    seen = fake_wonder(monkeypatch, list(wonder), fail_after=None if wonder else 0)
    mem = Memory(tmp_path / "m.json")
    for e in copy.deepcopy(library):
        mem.add(e.name, e.cause, e.body, e.members, e.verified)
    sci = Scientist(SimRig(truth, log_dir=None), teleport, k=4, memory=mem)
    sci.adopt(mem.get(current))
    return sci, seen


def since_alert(sci, kind):
    alert = max(i for i, e in enumerate(sci.notebook) if e["text"].startswith("Body changed"))
    return [e for e in sci.notebook[alert:] if e["kind"] == kind]


def test_round_trip(library, tmp_path):
    mem = Memory(tmp_path / "m.json", robot="airo-2")
    for e in library:
        mem.add(e.name, e.cause, e.body, e.members, e.verified)
    mem.recall(mem.get(1))
    back = Memory(tmp_path / "m.json", robot="airo-2")
    assert [(e.n, e.name, e.body, e.members, e.recalls, e.robot) for e in back.entries] == \
           [(e.n, e.name, e.body, e.members, e.recalls, e.robot) for e in mem.entries]
    assert back.current.n == 1 and back.entries[0].recalls == 1 and json.loads(back.path.read_text())


def test_saved_body_becomes_the_active_entry(tmp_path, monkeypatch):
    monkeypatch.setattr("airo.memory.MEMORY_DIR", tmp_path / "memory")
    save(TRUTH, ensemble([], TRUTH, 3), "bare chassis", tmp_path / "body_sim.json")
    mem = Memory.open(tmp_path / "body_sim.json")
    assert mem.path.name == "elegoo-v4_sim.json" and (mem.current.n, mem.current.name) == (1, "bare chassis")


def test_match_ranks_the_body_that_explains_the_window(library):
    mem = Memory()
    for e in library:
        mem.add(e.name, e.cause, e.body, e.members, e.verified)
    window = asyncio.run(changed(SimRig(TRUTH, log_dir=None), HEAVY))
    assert [e.n for e, _ in mem.match(window)] == [2]
    assert mem.match(window, exclude=2) == []


def test_match_ignores_the_motion_before_the_change(library):
    mem = Memory()
    for e in library:
        mem.add(e.name, e.cause, e.body, e.members, e.verified)
    rig = SimRig(HEAVY, log_dir=None)
    window = asyncio.run(changed(rig, TRUTH, HEAVY))
    t = window.t[0] + CHANGE_AT
    assert window.t[0] < t <= since_change(library[1].members, window, strict=True).t[0]
    assert [e.n for e, _ in mem.match(window, library[1].members, exclude=2)] == [1]


def test_match_on_the_factory_miss():
    """Recorded on the factory twin: restarted with the heavy body active on a bare car, so the whole window is the
    new body. Scored only after the first surprising window, the bare body missed it; it scores 3.2 there."""
    d = json.loads((Path(__file__).parent / "fixtures" / "factory_miss.json").read_text())
    window = Segment(*(np.array(d[k]) for k in ("cmd_t", "cmd", "t", "pose")))
    mem = Memory()
    for e in d["entries"]:
        mem.add(f"#{e['n']}", "", Body(**e["members"][0]), [Body(**m) for m in e["members"]], True)
    assert [e.n for e, _ in mem.match(window, [Body(**b) for b in d["current"]], exclude=2)] == [1]


@pytest.mark.parametrize("current, truth, expect", [(2, TRUTH, 1), (1, HEAVY, 2)])
def test_recall_confirms_with_one_probe(library, monkeypatch, tmp_path, current, truth, expect):
    sci, _ = scientist(library, monkeypatch, tmp_path, current, truth)
    window = asyncio.run(changed(sci.rig, truth, library[current - 1].body))
    asyncio.run(sci.relearn(window))

    assert (sci.n, sci.name) == (expect, library[expect - 1].name) and sci.verified
    assert len(since_alert(sci, "result")) == 1 and sci.notebook[-1]["kind"] == "recall"
    recall = sci.notebook[-1]
    assert recall["n"] == expect and recall["recall_s"] < 3 and "I know this one" in recall["text"]
    assert {"predicted", "measured", "z"} <= recall.keys()
    assert len(sci.memory.entries) == 2 and sci.memory.get(expect).recalls == 1
    assert Memory(tmp_path / "m.json").current.n == expect


def test_wrong_candidate_falls_back_to_relearn(library, monkeypatch, tmp_path):
    """A body that explains the window only because it is vague cannot pass the confirming probe."""
    sci, _ = scientist(library[1:], monkeypatch, tmp_path, 1, TRUTH)
    vague = sci.memory.add("vague", "a guess", TRUTH, ensemble([], TRUTH, 4), False)
    sci.adopt(sci.memory.get(1))
    window = asyncio.run(changed(sci.rig, TRUTH))
    assert [e.n for e, _ in sci.memory.match(window, exclude=1)] == [vague.n]
    asyncio.run(sci.relearn(window))

    assert not since_alert(sci, "recall") and len(since_alert(sci, "result")) > 1
    # the relearn lands on the vague entry's body, so it refines that entry rather than minting a third
    assert (sci.n, len(sci.memory.entries), vague.recalls, vague.verified) == (2, 2, 0, True)


def test_wonder_gets_the_candidates_and_a_recall_tool(library, monkeypatch, tmp_path):
    sci, seen = scientist(library, monkeypatch, tmp_path, 2, TRUTH, [("recall", {"n": 2}), ("recall", {"n": 1})])
    window = asyncio.run(changed(sci.rig, TRUTH))
    asyncio.run(sci.relearn(window))

    assert '#1 "bare chassis"' in seen[0][1]["content"] and "#2" not in seen[0][1]["content"]
    assert "not a candidate" in results(seen)[0]["error"]
    assert sci.n == 1 and sci.notebook[-1]["kind"] == "recall" and not any(e.get("fallback") for e in sci.notebook)


def test_fleet_shares_one_pool_with_per_robot_active_bodies(tmp_path):
    pool = Memory(tmp_path / "pool.json")
    first = pool.view("airo-1").add("bare", "learned from scratch", TRUTH, [TRUTH], True)
    two = pool.view("airo-2")
    assert two.inherit() is first and "airo-2" not in first.tested
    heavy = two.add("heavier", "weight", HEAVY, [HEAVY], True)
    one = pool.view("airo-1")
    assert one.current is first and one.get(heavy.n) is heavy
    one.recall(heavy)
    assert heavy.tested == ["airo-2", "airo-1"] and Memory(tmp_path / "pool.json").active == {"airo-1": 2, "airo-2": 2}


def test_check_body_after_loads_and_unloads(library, monkeypatch, tmp_path):
    """Load, unload, load again at a parked cell: learned in place, then recalled each time with the one pivot."""
    sci, _ = scientist(library[:1], monkeypatch, tmp_path, 1, TRUTH)
    rig = sci.rig
    rig._truth.append((rig.t, 1.0, 0.7, 0.0))
    asyncio.run(rig.step(0, 0))
    out = []
    for truth in (HEAVY, TRUTH, HEAVY, HEAVY):
        rig.truth = truth
        out.append(asyncio.run(sci.check_body("Tote loaded" if truth is HEAVY else "Tote unloaded")))
    assert [(o["result"], o["n"]) for o in out] == [("learned", 2), ("recalled", 1), ("recalled", 2), ("same", 2)]
    assert sci.memory.get(2).verified and out[1]["seconds"] < 3 and len(sci.memory.entries) == 2
    probes = [e["probe"] for e in sci.notebook if e["kind"] == "hypothesis"]
    assert {LIBRARY[p].kind for p in probes} <= {"pivot", "step"} and probes.count("step_60") <= 1
    assert "Tote loaded: checking" in sci.notebook[0]["text"]


def test_check_body_without_room_learns_partly_and_the_next_relearn_completes_it(library, monkeypatch, tmp_path):
    monkeypatch.setattr("airo.scientist.CHECK_PLAN", ["pivot_r_150", "pivot_l_60"])
    sci, _ = scientist(library[:1], monkeypatch, tmp_path, 1, TRUTH)
    rig = sci.rig
    rig.truth = HEAVY
    out = asyncio.run(sci.check_body("Tote loaded"))
    assert (out["result"], out["n"], sci.verified) == ("learned", 2, False)
    window = asyncio.run(changed(rig, HEAVY))
    asyncio.run(sci.relearn(window))
    assert sci.n == 2 and len(sci.memory.entries) == 2 and sci.verified and sci.memory.get(2).verified


@pytest.fixture
def carried(library, monkeypatch, tmp_path):
    """A parked car whose library holds Body #1 (bare) and Body #2 (heavy), both learned with the full schedule."""
    sci, _ = scientist(library, monkeypatch, tmp_path, 1, TRUTH)
    sci.rig._truth.append((sci.rig.t, 1.0, 0.7, 0.0))
    asyncio.run(sci.rig.step(0, 0))
    return sci


def test_a_light_tote_within_noise_is_the_same_body(carried):
    carried.rig.truth = replace(TRUTH, g_l=TRUTH.g_l * 0.97, g_r=TRUTH.g_r * 0.97, eta=TRUTH.eta * 0.99)
    assert asyncio.run(carried.check_body("loaded a light tote"))["result"] == "same"


def test_a_vague_current_body_does_not_keep_a_changed_car(carried):
    """Unloaded while on a heavy body whose cloud is wide enough to fit the bare car at |z| < 2: the tight bare body
    is far more likely, so it is recalled rather than "same"."""
    tote = replace(TRUTH, g_l=TRUTH.g_l * 0.9, g_r=TRUTH.g_r * 0.9)
    vague = carried.memory.add("heavy tote, vague", "", tote, [replace(tote, g_l=tote.g_l * f, g_r=tote.g_r * f)
                                                              for f in np.linspace(0.92, 1.08, 6)], True)
    carried.adopt(vague)
    out = asyncio.run(carried.check_body("unloaded"))
    assert (out["result"], out["n"]) == ("recalled", 1)


def test_a_check_bumped_once_is_decided_by_the_repeat(carried, monkeypatch):
    bumps = probes_run(monkeypatch, [0.15])
    carried.rig.truth = HEAVY
    out = asyncio.run(carried.check_body("loaded a heavy tote"))
    assert (out["result"], out["n"], len(carried.memory.entries)) == ("recalled", 2, 2) and not bumps


def test_a_check_bumped_twice_keeps_the_body_and_learns_nothing(carried, monkeypatch):
    probes_run(monkeypatch, [0.15, -0.1])
    out = asyncio.run(carried.check_body("loaded a heavy tote"))
    assert (out["result"], out["n"], len(carried.memory.entries)) == ("same", 1, 2)
    assert "something is moving the car" in carried.notebook[-1]["text"]


def test_a_probe_the_rig_ran_long_is_scored_on_what_it_ran(carried, monkeypatch):
    """A car that falls behind in a busy world holds each command longer than planned."""
    async def held(rig, p):
        t0 = rig.pose.t
        for l, r in commands(p):
            for _ in range(2):
                await rig.step(int(l), int(r))
        return rig.segment(t0 + DT / 2)

    monkeypatch.setattr("airo.scientist.run", held)
    assert asyncio.run(carried.check_body("loaded a light tote"))["result"] == "same"


def probes_run(monkeypatch, shoves: list[float]) -> list[float]:
    """The scientist's probe runner, shoving the car sideways (m, and as many rad) halfway through the next runs."""
    from airo.probes import run

    async def bumped(rig, p):
        if not shoves:
            return await run(rig, p)
        d, t0 = shoves.pop(0), rig.pose.t
        for i, (l, r) in enumerate(commands(p)):
            if i == 10:
                x, y, th = rig.true_pose
                rig._truth.append((rig.t, x - d * np.sin(th), y + d * np.cos(th), th + 4 * d))
            await rig.step(int(l), int(r))
        return rig.segment(t0 + DT / 2)

    monkeypatch.setattr("airo.scientist.run", bumped)
    return shoves


def test_two_bodies_that_fit_alike_get_a_second_pivot(carried):
    twin = carried.memory.add("twin", "", carried.best, carried.members, True)
    out = asyncio.run(carried.check_body("loaded a light tote"))
    ran = [e["probe"] for e in carried.notebook if e["kind"] == "result" and "probe" in e]
    assert ran == ["pivot_l_100", "pivot_r_100"] and out["result"] == "same" and twin.recalls == 0


@pytest.mark.parametrize("current", [1, 2])
def test_a_relearn_that_lands_on_a_known_body_refines_it(library, monkeypatch, tmp_path, current):
    """Nothing changed (current 1), or the bare car is back but its recall probe was too loose to confirm (current 2):
    the relearn ends on bare chassis either way, and refines #1 instead of minting an "unchanged" #3."""
    monkeypatch.setattr("airo.scientist.CONFIRM_TOL", np.array([3.0, 3.0, 6.0]) if current == 1 else np.zeros(3))
    sci, _ = scientist(library, monkeypatch, tmp_path, current, TRUTH)
    asyncio.run(sci.relearn(asyncio.run(changed(sci.rig, TRUTH, library[current - 1].body))))
    assert (sci.n, len(sci.memory.entries), sci.name) == (1, 2, "bare chassis")
    assert "same body as #1" in sci.notebook[-1]["text"] and sci.memory.get(1).verified


@pytest.mark.parametrize("room, result", [({"pivot_l_60"}, "recalled"), (set(), "same")])
def test_a_tight_bay_checks_with_the_pivot_it_has_room_for(carried, monkeypatch, room, result):
    from airo import probes

    monkeypatch.setattr("airo.scientist.safe", lambda rig, p, body: p.name in room or p.kind != "pivot"
                        and probes.safe(rig, p, body))
    carried.rig.truth = HEAVY
    out = asyncio.run(carried.check_body("loaded a heavy tote"))
    ran = {e["probe"] for e in carried.notebook if e["kind"] == "result" and "probe" in e}
    assert out["result"] == result and ran <= room
    assert result != "same" or "no pivot clears this bay" in carried.notebook[-1]["text"]
