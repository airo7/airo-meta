"""Muse Image self-portrait of the current body. Offline or on any API error it quietly returns None."""

import asyncio

from airo import meta_api

CAR = ("a small four-wheeled robot car seen from slightly above: black acrylic chassis, yellow wheels, a blue circuit "
       "board and a square printed marker on top")
STYLE = "Friendly flat illustration, soft colours, plain light background, no text."


def prompt(name: str, seen: str | None = None) -> str:
    look = f" Its camera sees: {seen}." if seen else ""
    return f"A self-portrait of {CAR}. How it now knows its own body: {name}.{look} {STYLE}"


async def draw(name: str, seen: str | None = None, timeout: float = 60) -> bytes | None:
    try:
        return await asyncio.wait_for(meta_api.image(prompt(name, seen)), timeout)
    except Exception:
        return None
