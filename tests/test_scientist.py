import asyncio
import json
from dataclasses import replace

import numpy as np
import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function

from airo import meta_api
from airo.body import Body, surprise
from airo.probes import SCHEDULE
from airo.scientist import RELEARN, Scientist
from airo.sim import SimRig

TRUTH = Body(g_l=0.0030, g_r=0.0033, dz=25, eta=0.3, tau=0.2, lag=0.1, kappa=-0.1)
HEAVY = replace(TRUTH, g_l=TRUTH.g_l * 0.75, g_r=TRUTH.g_r * 0.75, eta=TRUTH.eta * 0.7)


async def teleport(rig):
    rig._truth.append((rig.t, 1.0, 0.6, 0.0))
    await rig.step(0, 0)


def fake_wonder(monkeypatch, calls, fail_after=None):
    """Answers each turn with the next scripted tool call; raises TimeoutError once fail_after turns are used."""
    seen = []

    async def wonder(messages, tools, **_):
        seen.append(list(messages))
        if fail_after is not None and len(seen) > fail_after:
            raise TimeoutError
        name, args = calls.pop(0)
        call = ChatCompletionMessageFunctionToolCall(id=f"c{len(seen)}", type="function",
                                                     function=Function(name=name, arguments=json.dumps(args)))
        return ChatCompletionMessage(role="assistant", content=None, tool_calls=[call])

    monkeypatch.setattr(meta_api, "wonder", wonder)
    return seen


def probe(name):
    return "run_probe", {"probe": name, "hypothesis": f"{name} will tell us something"}


def results(seen):
    return [json.loads(m["content"]) for m in seen[-1] if m["role"] == "tool"]


async def drive(rig, ticks):
    t0 = rig.pose.t
    for l, r, n in ticks:
        for _ in range(n):
            await rig.step(l, r)
    return rig.segment(t0)


def assert_close(got: Body, truth: Body):
    for k in ("g_l", "g_r", "dz", "eta"):
        assert getattr(got, k) == pytest.approx(getattr(truth, k), rel=0.1), k
    assert got.tau + got.lag == pytest.approx(truth.tau + truth.lag, rel=0.1)
    assert got.kappa == pytest.approx(truth.kappa, abs=0.03)


def test_learn_with_wonder(monkeypatch):
    calls = [("list_probes", {}), *map(probe, SCHEDULE[:5]), ("refit", {}), ("predict", {"probe": "arc_r"}),
             *map(probe, SCHEDULE[5:]), ("conclude", {"text": "A stock car.", "body_name": "stock"})]
    seen = fake_wonder(monkeypatch, calls)
    updates = []
    sci = Scientist(SimRig(TRUTH, log_dir=None), teleport, k=4, on_update=updates.append)
    asyncio.run(sci.learn())

    assert not calls and (sci.n, sci.name) == (1, "stock")
    assert_close(sci.best, TRUTH)
    assert len(sci.segments) == len(SCHEDULE) and len(sci.members) == 4 and updates
    kinds = [e["kind"] for e in sci.notebook]
    assert kinds[:3] == ["hypothesis", "prediction", "result"] and kinds[-1] == "conclusion"
    assert "fallback" not in kinds
    result = next(e for e in sci.notebook if e["kind"] == "result")
    assert {"hypothesis", "predicted", "measured", "z"} <= result.keys()
    z = [r["z"] for r in results(seen) if "z" in r]
    assert z[0]["forward_cm"] > 3 and abs(z[5]["turn_deg"]) > 3, "prior and a straight-line-only fit are wrong"
    assert max(map(abs, z[-1].values())) < 3, "pulse after the straight-line refit"


def test_fallback_finishes_the_schedule(monkeypatch):
    seen = fake_wonder(monkeypatch, [probe("pivot_l_100"), probe("step_80")], fail_after=2)
    sci = Scientist(SimRig(TRUTH, log_dir=None), teleport, k=4)
    asyncio.run(sci.learn())

    ran = [e["probe"] for e in sci.notebook if e["kind"] == "result"]
    assert ran[:2] == ["pivot_l_100", "step_80"] and len(ran) == 10 and len(set(ran)) == 10
    assert len(seen) == 3 and sci.n == 1
    fb = [e for e in sci.notebook if e.get("fallback")]
    assert fb[0]["kind"] == "fallback" and "TimeoutError" in fb[0]["text"] and fb[-1]["kind"] == "conclusion"
    assert not any(e.get("fallback") for e in sci.notebook[:6])
    assert_close(sci.best, TRUTH)


def test_relearn_after_heavy_load(monkeypatch):
    rig = SimRig(TRUTH, log_dir=None)
    sci = Scientist(rig, teleport, k=6)
    fake_wonder(monkeypatch, [], fail_after=0)
    asyncio.run(sci.learn())
    mission = [(140, 140, 15), (80, 140, 20), (140, 80, 20)]

    rig.truth = HEAVY
    asyncio.run(teleport(rig))
    window = asyncio.run(drive(rig, mission))
    assert np.nanmedian(surprise(sci.members, window)) > 3

    async def describe(frame, prompt):
        return "a weight pack strapped on top of the car"

    monkeypatch.setattr(meta_api, "describe", describe)
    calls = [*map(probe, RELEARN), probe("pivot_l_60"), ("conclude", {"text": "Heavier.", "body_name": "heavy"})]
    seen = fake_wonder(monkeypatch, calls)
    secs = asyncio.run(sci.relearn(window))

    assert secs > 0 and (sci.n, sci.name) == (2, "heavy")
    prompt = seen[0][1]["content"]
    assert "weight pack" in prompt and "surprise" in prompt
    assert "budget" in results(seen)[-1]["error"]
    assert sci.notebook[[e["kind"] for e in sci.notebook].index("alert")]["seen"].startswith("a weight")
    assert len(sci.segments) == 1 + 3

    asyncio.run(teleport(rig))
    after = asyncio.run(drive(rig, mission))
    assert np.nanmedian(surprise(sci.members, after)) < 3
    assert sci.best.eta == pytest.approx(HEAVY.eta, rel=0.15)


def test_rejects_unsafe_and_unknown_probes(monkeypatch):
    calls = [probe("step_150"), probe("warp_9"), probe("pivot_l_100"),
             ("conclude", {"text": "Pivoted only.", "body_name": "edge"})]
    seen = fake_wonder(monkeypatch, calls)
    rig = SimRig(TRUTH, pose0=(1.65, 0.6, 0.0), log_dir=None)
    sci = Scientist(rig, k=3)
    asyncio.run(sci.learn(budget=3))

    out = results(seen)
    assert "leave the floor" in out[0]["error"] and "unknown" in out[1]["error"] and "z" in out[2]
    assert [e["probe"] for e in sci.notebook if e["kind"] == "result"] == ["pivot_l_100"]
    assert any(e["kind"] == "alert" and "step_150" in e["text"] for e in sci.notebook)
    assert rig.true_pose[0] < 1.7 and sci.n == 1
