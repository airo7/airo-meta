"""Muse Spark reads a command, typed or spoken, and picks one action for the robot from where it is and what it knows."""

import asyncio
import json
import math
from contextlib import suppress

from airo import meta_api
from airo.voice import Intent

MODEL = "muse-spark-1.3"
PROMPT = """You are AIRO, a small robot car that learned how its own body moves. Turn the user's command into the
tool calls that carry it out, in the order to run them: one for a simple command, several for "do this, then that",
shapes or routes. Include every step asked for, also the ones after "then" or "and". Work each step out from where the
previous one leaves the robot.
Positions are floor coordinates in metres. A heading of 0 deg points along +x and 90 deg along +y; positive turns are
to the left (counter-clockwise seen from above). Work relative commands ("forward 20 cm", "back a bit", "the far
corner", "face marker 2") out from the robot's pose. Keep go_to inside the drivable area; if the asked place is outside
it, go to the nearest drivable point. The robot can push a small light object (up to 20 cm) with push, using that
object's position from the list: to a place given as to_x, to_y, or with no place out of its driving area toward the
nearest edge. A command that asks a question or asks to be told something (what you see, where you are) always
ends with one more call after any moves: say with the answer from the situation below, or answer_after when the
answer depends on what the moves find ("look around and tell me what you see" is turn, then answer_after). For something the robot cannot do, call say.

"""


def _tool(name: str, description: str, required=(), **props) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


_NUM = {"type": "number"}
TOOLS = [
    _tool("go_to", "Drive to a floor point, around anything in the way.", ("x", "y"), x=_NUM, y=_NUM),
    _tool("turn", "Pivot in place by an angle in degrees, positive to the left.", ("degrees",), degrees=_NUM),
    _tool("patrol", "Drive laps of the floor's corners until stopped."),
    _tool("push", "Push the object on the floor at x, y to to_x, to_y; without them out of the driving area, toward "
          "the nearest edge.", ("x", "y"), x=_NUM, y=_NUM, to_x=_NUM, to_y=_NUM),
    _tool("stop", "Stop now."),
    _tool("learn", "Drive test moves to measure how the body moves now and update the body model. Use it whenever "
          "the body may have changed (a load put on or taken off, a part swapped, driving oddly) or the user asks "
          "the robot to check, test or relearn itself."),
    _tool("status", "Report the body model already stored and how the robot is doing, without moving or testing."),
    _tool("answer_after", "Call last when the answer to the user depends on what the moves before it find: you are "
          "asked again once they ran, with the situation then."),
    _tool("say", "Answer in one or two plain sentences for a person: name places by the nearest marker or "
          "corner and distances in cm, not coordinates or degrees.", ("text",), text={"type": "string"}),
]


def _plain(i: Intent) -> str:
    if i.kind == "goto":
        return f"Drive to ({i.xy[0]:.2f}, {i.xy[1]:.2f}) m."
    if i.kind == "turn":
        return f"Turn {abs(math.degrees(i.angle)):.0f}° {'left' if i.angle > 0 else 'right'}."
    if i.kind == "deliver":
        return f"Deliver to {i.target}."
    if i.kind == "push":
        to = f"to ({i.to[0]:.2f}, {i.to[1]:.2f})" if i.to else "out of the way"
        return f"Push the object at ({i.xy[0]:.2f}, {i.xy[1]:.2f}) {to}."
    if i.kind == "answer_after":
        return "Then tell what I found."
    return i.text if i.kind == "say" else i.kind.capitalize() + "."


async def understand(text: str, situation: str, stations=None, done: list[str] | None = None) -> list[Intent]:
    """With done, the plan already ran: return only what is still owed, an answer judged from the situation now."""
    tools = TOOLS
    if stations:
        tools = [*TOOLS, _tool("deliver", "Take a tote or parcel to a station.", ("station",),
                               station={"type": "string", "enum": list(stations)})]
    ask = text
    if done is not None:
        ask += (f"\n\nAlready done, in order: {' '.join(done) or 'nothing'} Don't repeat any of it. If the command "
                "also asks to be told something, call say with the answer from the situation now; if nothing is left, "
                "reply done without a tool.")
    messages = [{"role": "system", "content": PROMPT + situation}, {"role": "user", "content": ask}]
    # a reply usually takes 1-2 s; a slow one rarely recovers, a fresh request usually does
    for wait in (8, 10):
        try:
            msg = await meta_api.wonder(messages, tools, model=MODEL, timeout=wait, effort="low", shown=text)
            break
        except TimeoutError:
            if wait == 10:
                raise
    if not msg.tool_calls:
        return [] if done is not None else [Intent("say", text=msg.content or "I did not catch that.")]
    intents = [_intent(call) for call in msg.tool_calls]
    if done is not None:
        return [i for i in intents if i.kind == "say"]
    # Spark sometimes says it will report back but leaves out the call that lets it: ask again after the moves anyway
    if msg.content and not any(i.kind in ("say", "answer_after") for i in intents):
        intents.append(Intent("answer_after"))
    return intents


def _intent(call) -> Intent:
    a = json.loads(call.function.arguments or "{}")
    intent = {
        "go_to": lambda: Intent("goto", xy=(float(a["x"]), float(a["y"]))),
        "push": lambda: Intent("push", xy=(float(a["x"]), float(a["y"])),
                               to=(float(a["to_x"]), float(a["to_y"])) if "to_x" in a and "to_y" in a else None),
        "turn": lambda: Intent("turn", angle=math.radians(float(a["degrees"]))),
        "deliver": lambda: Intent("deliver", a["station"]),
        "say": lambda: Intent("say", text=a["text"]),
    }.get(call.function.name, lambda: Intent(call.function.name))()
    meta_api.tool_result(call.id, json.dumps(a), plain=_plain(intent))
    return intent


async def warm():
    """The first request from a process takes several seconds; pay that at start, not on the first command."""
    with suppress(Exception):
        await asyncio.wait_for(meta_api.client().chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Reply with: ok"}], reasoning_effort="low"), 30)
