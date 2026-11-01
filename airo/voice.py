"""Voice commands: realtime Muse Voice Transcribe over WebSocket, then regex intents.

Protocol: https://dev.meta.ai/docs/api-reference/voice/realtime and https://dev.meta.ai/docs/speech-to-text
"""

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import Literal

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from airo import meta_api

REALTIME_URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"
RATE = 24_000
FRAME = RATE * 80 // 1000


async def mic(device=None) -> AsyncIterator[bytes]:
    """80 ms frames of 24 kHz mono int16 PCM, paced in real time by the sound card."""
    import sounddevice as sd  # needs PortAudio at import time; keeps tests runnable without it

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[bytes] = asyncio.Queue()
    with sd.RawInputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="int16", device=device,
                           callback=lambda data, *_: loop.call_soon_threadsafe(q.put_nowait, bytes(data))):
        while True:
            yield await q.get()


def _handshake() -> dict:
    return {
        "authorization": {"accessToken": f"Bearer {os.environ['MODEL_API_KEY']}"},
        "audioEncoding": "PCM_24KHZ",
        "model": MODEL,
        "mode": "ENDPOINTING",
        "emitAudioProgress": False,
    }


def _on_event(ev: dict, on_text: Callable[[str, bool], None], turn: dict) -> None:
    match ev.get("type"):
        case "transcript":
            turn.setdefault("t0", time.perf_counter())
            on_text(ev["transcript"], False)
        case "speechComplete":
            ms = (time.perf_counter() - turn.pop("t0", time.perf_counter())) * 1000
            meta_api.record(MODEL, "transcribe", "microphone", ev["transcript"], ms)
            on_text(ev["transcript"], True)
        case "error":
            raise RuntimeError(f"ASR error: {ev.get('message')}")


async def _send_audio(ws: ClientConnection, source: AsyncIterator[bytes]) -> None:
    with suppress(ConnectionClosed):
        async with aclosing(source) as frames:
            async for frame in frames:
                await ws.send(frame)
        # Closing the socket instead would drop pending transcripts; the server closes with 1000 when done.
        await ws.send(json.dumps({"type": "endStream"}))


async def transcribe(on_text: Callable[[str, bool], None], device=None, frames: AsyncIterator[bytes] | None = None):
    """Stream the mic, or frames (80 ms of 24 kHz int16) until they end, until cancelled. Partials are cumulative
    (replace, don't append); finals end a turn."""
    async with connect(REALTIME_URL) as ws:
        await ws.send(json.dumps(_handshake()))
        ack = json.loads(await ws.recv())
        if "sessionId" not in ack:
            raise RuntimeError(f"ASR handshake rejected: {ack}")
        turn: dict = {}
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_send_audio(ws, frames or mic(device)))
                async for raw in ws:
                    _on_event(json.loads(raw), on_text, turn)
        except ExceptionGroup as eg:
            raise eg.exceptions[0] from None


@dataclass
class Intent:
    kind: Literal["deliver", "learn", "stop", "status", "goto", "patrol", "turn", "say"]
    target: str | None = None
    xy: tuple[float, float] | None = None
    angle: float | None = None
    text: str | None = None


_STOP = re.compile(r"\b(stop|halt|freeze|abort|cancel|emergency)\b")
_DELIVER = re.compile(r"\b(deliver|take|bring|push|carry|move|drop)\b(?:.*?\b(?:to|towards?|at)\s+"
                      r"(?:the\s+)?(?:station\s+)?([a-z]+))?")
_LEARN = re.compile(r"\b(re)?learn|\bcalibrat|\bexperiment|\bprobe")
_NUM = r"(-?\d*\.?\d+)"
_GOTO = re.compile(rf"\b(?:go|drive|move|head)\s+(?:to\s+)?\(?{_NUM}\s*(?:,\s*|\s+){_NUM}")
_PATROL = re.compile(r"\bpatrol")
_STATUS = re.compile(r"\bhow\b.*\b(feel|doing|going)\b|\bstatus\b|\bare you (ok|okay|alright)\b")


def parse_intent(text: str, stations=None) -> Intent | None:
    """With the site's station names, a station named anywhere in a delivery counts ("bring it over to Q.A.").
    "go to 0.4 0.6" is a floor point in metres."""
    t = text.lower()
    if _STOP.search(t):
        return Intent("stop")
    if _LEARN.search(t):
        return Intent("learn")
    if _STATUS.search(t):
        return Intent("status")
    if m := _GOTO.search(t):
        return Intent("goto", xy=(float(m[1]), float(m[2])))
    if _PATROL.search(t):
        return Intent("patrol")
    if m := _DELIVER.search(t):
        target = m[2]
        if stations and target not in stations:
            words = re.sub(r"[^a-z ]", "", t).split()
            target = next((s for s in stations if s in words), target)
        return Intent("deliver", target)
    return None
