"""Muse Spark reads a command, typed or spoken, and picks one action for the robot from where it is and what it knows."""

import json
import math

from airo import meta_api
from airo.voice import Intent

MODEL = "muse-spark-1.3"
PROMPT = """You are AIRO, a small robot car that learned how its own body moves. Turn the user's command into exactly
one tool call. Positions are floor coordinates in metres. A heading of 0 deg points along +x and 90 deg along +y;
positive turns are to the left (counter-clockwise seen from above). Work relative commands ("forward 20 cm", "back a
bit", "the far corner", "face marker 2") out from the robot's pose. Keep go_to inside the drivable area; if the asked
place is outside it, go to the nearest drivable point. For a question, or something the robot cannot do, call say.

"""


def _tool(name: str, description: str, required=(), **props) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


_NUM = {"type": "number"}
TOOLS = [
    _tool("go_to", "Drive to a floor point, around anything in the way.", ("x", "y"), x=_NUM, y=_NUM),
    _tool("turn", "Pivot in place by an angle in degrees, positive to the left.", ("degrees",), degrees=_NUM),
    _tool("patrol", "Drive laps of the floor's corners until stopped."),
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
    return i.text if i.kind == "say" else i.kind.capitalize() + "."


async def understand(text: str, situation: str, stations=None) -> Intent:
    tools = TOOLS
    if stations:
        tools = [*TOOLS, _tool("deliver", "Take a tote or parcel to a station.", ("station",),
                               station={"type": "string", "enum": list(stations)})]
    msg = await meta_api.wonder([{"role": "system", "content": PROMPT + situation}, {"role": "user", "content": text}],
                                tools, model=MODEL, timeout=8)
    if not msg.tool_calls:
        return Intent("say", text=msg.content or "I did not catch that.")
    call = msg.tool_calls[0]
    a = json.loads(call.function.arguments or "{}")
    intent = {
        "go_to": lambda: Intent("goto", xy=(float(a["x"]), float(a["y"]))),
        "turn": lambda: Intent("turn", angle=math.radians(float(a["degrees"]))),
        "deliver": lambda: Intent("deliver", a["station"]),
        "say": lambda: Intent("say", text=a["text"]),
    }.get(call.function.name, lambda: Intent(call.function.name))()
    meta_api.tool_result(call.id, json.dumps(a), plain=_plain(intent))
    return intent
