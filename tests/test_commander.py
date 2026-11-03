import asyncio
import json
import math
from types import SimpleNamespace

from airo import commander, meta_api
from airo.voice import Intent


def reply(name, **args):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    async def wonder(messages, tools, model, timeout, effort=None, **_):
        reply.seen = messages, tools
        return SimpleNamespace(content=None, tool_calls=[call])
    return wonder


def test_tool_calls_become_intents(monkeypatch):
    for (name, args), want in [
        (("go_to", {"x": 0.4, "y": 0.61}), Intent("goto", xy=(0.4, 0.61))),
        (("turn", {"degrees": -90}), Intent("turn", angle=-math.pi / 2)),
        (("patrol", {}), Intent("patrol")),
        (("say", {"text": "I am a car."}), Intent("say", text="I am a car.")),
        (("deliver", {"station": "packing"}), Intent("deliver", "packing")),
        (("push", {"x": 0.7, "y": 0.28}), Intent("push", xy=(0.7, 0.28))),
        (("push", {"x": 0.5, "y": 0.4, "to_x": 0.4, "to_y": 0.61}), Intent("push", xy=(0.5, 0.4), to=(0.4, 0.61))),
    ]:
        monkeypatch.setattr(meta_api, "wonder", reply(name, **args))
        assert asyncio.run(commander.understand("…", "The floor is 0.80 x 1.22 m.", {"packing": "x"})) == [want]
    messages, tools = reply.seen
    assert messages[0]["content"].endswith("The floor is 0.80 x 1.22 m.")
    assert tools[-1]["function"]["parameters"]["properties"]["station"]["enum"] == ["packing"]


def test_a_plain_answer_without_a_tool_is_said(monkeypatch):
    async def wonder(messages, tools, model, timeout, effort=None, **_):
        return SimpleNamespace(content="Hello.", tool_calls=None)
    monkeypatch.setattr(meta_api, "wonder", wonder)
    assert asyncio.run(commander.understand("hi", "")) == [Intent("say", text="Hello.")]


def test_a_slow_reply_is_asked_once_more(monkeypatch):
    waits = []

    async def wonder(messages, tools, model, timeout, effort=None, **_):
        waits.append(timeout)
        if len(waits) == 1:
            raise TimeoutError
        return SimpleNamespace(content="Hello.", tool_calls=None)
    monkeypatch.setattr(meta_api, "wonder", wonder)
    assert asyncio.run(commander.understand("hi", "")) == [Intent("say", text="Hello.")]
    assert len(waits) == 2


def test_a_chain_of_steps_runs_in_the_order_given(monkeypatch):
    calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
             for i, (n, a) in enumerate([("go_to", {"x": 0.4, "y": 0.6}), ("turn", {"degrees": 90}), ("say", {"text": "Done."})])]

    async def wonder(messages, tools, model, timeout, effort=None, **_):
        return SimpleNamespace(content=None, tool_calls=calls)
    monkeypatch.setattr(meta_api, "wonder", wonder)
    assert [i.kind for i in asyncio.run(commander.understand("…", ""))] == ["goto", "turn", "say"]


def test_the_follow_up_shows_only_the_command_in_the_feed(monkeypatch):
    seen = {}

    async def wonder(messages, tools, model, timeout, effort=None, shown=None):
        seen.update(asked=messages[-1]["content"], shown=shown)
        return SimpleNamespace(content="done", tool_calls=None)
    monkeypatch.setattr(meta_api, "wonder", wonder)
    assert asyncio.run(commander.understand("look around and tell me", "", done=["Turned 90° left."])) == []
    assert "Turned 90° left." in seen["asked"] and seen["shown"] == "look around and tell me"
