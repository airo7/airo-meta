import asyncio
import json
import math
from types import SimpleNamespace

from airo import commander, meta_api
from airo.voice import Intent


def reply(name, **args):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    async def wonder(messages, tools, model, timeout):
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
    ]:
        monkeypatch.setattr(meta_api, "wonder", reply(name, **args))
        assert asyncio.run(commander.understand("…", "The floor is 0.80 x 1.22 m.", {"packing": "x"})) == want
    messages, tools = reply.seen
    assert messages[0]["content"].endswith("The floor is 0.80 x 1.22 m.")
    assert tools[-1]["function"]["parameters"]["properties"]["station"]["enum"] == ["packing"]


def test_a_plain_answer_without_a_tool_is_said(monkeypatch):
    async def wonder(messages, tools, model, timeout):
        return SimpleNamespace(content="Hello.", tool_calls=None)
    monkeypatch.setattr(meta_api, "wonder", wonder)
    assert asyncio.run(commander.understand("hi", "")) == Intent("say", text="Hello.")
