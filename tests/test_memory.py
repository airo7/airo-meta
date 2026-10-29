import asyncio
import copy
import json

import pytest

from airo.body import ensemble, save
from airo.memory import Memory
from airo.scientist import Scientist
from airo.sim import SimRig
from test_scientist import HEAVY, TRUTH, drive, fake_wonder, results, teleport

MISSION = [(140, 140, 15), (80, 140, 20), (140, 80, 20)]


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


async def changed(rig, truth):
    """The bare body drives the mission much further, so it starts on the left of the floor."""
    rig.truth = truth
    rig._truth.append((rig.t, 0.5, 0.7, 0.0))
    await rig.step(0, 0)
    return await drive(rig, MISSION)


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
    alert = max(i for i, e in enumerate(sci.notebook) if e["kind"] == "alert")
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


@pytest.mark.parametrize("current, truth, expect", [(2, TRUTH, 1), (1, HEAVY, 2)])
def test_recall_confirms_with_one_probe(library, monkeypatch, tmp_path, current, truth, expect):
    sci, _ = scientist(library, monkeypatch, tmp_path, current, truth)
    window = asyncio.run(changed(sci.rig, truth))
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
    assert sci.n == 3 and vague.recalls == 0 and sci.memory.current.n == 3


def test_wonder_gets_the_candidates_and_a_recall_tool(library, monkeypatch, tmp_path):
    sci, seen = scientist(library, monkeypatch, tmp_path, 2, TRUTH, [("recall", {"n": 2}), ("recall", {"n": 1})])
    window = asyncio.run(changed(sci.rig, TRUTH))
    asyncio.run(sci.relearn(window))

    assert '#1 "bare chassis"' in seen[0][1]["content"] and "#2" not in seen[0][1]["content"]
    assert "not a candidate" in results(seen)[0]["error"]
    assert sci.n == 1 and sci.notebook[-1]["kind"] == "recall" and not any(e.get("fallback") for e in sci.notebook)
