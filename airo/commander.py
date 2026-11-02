"""Muse Spark reads a command, typed or spoken, and picks one action for the robot from where it is and what it knows."""

import asyncio
import json
import math
from contextlib import suppress

from airo import meta_api
from airo.voice import Intent

MODEL = "muse-spark-1.3"
PROMPT = """You are AIRO, a small robot car that learned how its own body moves. Turn the user's command into exactly
one tool call. Positions are floor coordinates in metres. A heading of 0 deg points along +x and 90 deg along +y;
positive turns are to the left (counter-clockwise seen from above). Work relative commands ("forward 20 cm", "back a
bit", "the far corner", "face marker 2") out from the robot's pose. Keep go_to inside the drivable area; if the asked
place is outside it, go to the nearest drivable point. The robot can push a small light object (up to 20 cm) with push,
using that object's position from the list: to a place given as to_x, to_y, or with no place out of its driving area
toward the nearest edge. For a question, or something the robot cannot
do, call say.

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
    _tool("learn", "Run experiments to learn or relearn how the body moves."),
    _tool("status", "Report the robot's body and how it is doing."),
    _tool("say", "Answer in one or two plain sentences.", ("text",), text={"type": "string"}),
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
    return i.text if i.kind == "say" else i.kind.capitalize() + "."


async def understand(text: str, situation: str, stations=None) -> Intent:
    tools = TOOLS
    if stations:
        tools = [*TOOLS, _tool("deliver", "Take a tote or parcel to a station.", ("station",),
                               station={"type": "string", "enum": list(stations)})]
    messages = [{"role": "system", "content": PROMPT + situation}, {"role": "user", "content": text}]
    # a reply usually takes 1-2 s; a slow one rarely recovers, a fresh request usually does
    for wait in (8, 10):
        try:
            msg = await meta_api.wonder(messages, tools, model=MODEL, timeout=wait, effort="low")
            break
        except TimeoutError:
            if wait == 10:
                raise
    if not msg.tool_calls:
        return Intent("say", text=msg.content or "I did not catch that.")
    call = msg.tool_calls[0]
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
