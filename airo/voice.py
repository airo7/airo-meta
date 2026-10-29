"""Voice commands: realtime Muse Voice Transcribe over WebSocket, then regex intents.

Protocol: https://dev.meta.ai/docs/api-reference/voice/realtime and https://dev.meta.ai/docs/speech-to-text
"""

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import Literal

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

REALTIME_URL = "wss://api.meta.ai/v1/asr/realtime"
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
        "model": "muse-voice-transcribe-1.0",
        "mode": "ENDPOINTING",
        "emitAudioProgress": False,
    }


def _on_event(ev: dict, on_text: Callable[[str, bool], None]) -> None:
    match ev.get("type"):
        case "transcript":
            on_text(ev["transcript"], False)
        case "speechComplete":
            on_text(ev["transcript"], True)
        case "error":
            raise RuntimeError(f"ASR error: {ev.get('message')}")


async def _send_audio(ws: ClientConnection, device) -> None:
    with suppress(ConnectionClosed):
        async with aclosing(mic(device)) as frames:
            async for frame in frames:
                await ws.send(frame)
        # Closing the socket instead would drop pending transcripts; the server closes with 1000 when done.
        await ws.send(json.dumps({"type": "endStream"}))


async def transcribe(on_text: Callable[[str, bool], None], device=None) -> None:
    """Stream the mic until cancelled. Partials are cumulative (replace, don't append); finals end a turn."""
    async with connect(REALTIME_URL) as ws:
        await ws.send(json.dumps(_handshake()))
        ack = json.loads(await ws.recv())
        if "sessionId" not in ack:
            raise RuntimeError(f"ASR handshake rejected: {ack}")
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(_send_audio(ws, device))
                async for raw in ws:
                    _on_event(json.loads(raw), on_text)
        except ExceptionGroup as eg:
            raise eg.exceptions[0] from None


@dataclass
class Intent:
    kind: Literal["deliver", "learn", "stop", "status"]
    target: str | None = None


_STOP = re.compile(r"\b(stop|halt|freeze|abort|cancel|emergency)\b")
_DELIVER = re.compile(r"\b(deliver|take|bring|push|carry|move|drop)\b(?:.*?\b(?:to|towards?|at)\s+"
                      r"(?:the\s+)?(?:station\s+)?([a-z]+))?")
_LEARN = re.compile(r"\b(re)?learn|\bcalibrat|\bexperiment|\bprobe")
_STATUS = re.compile(r"\bhow\b.*\b(feel|doing|going)\b|\bstatus\b|\bare you (ok|okay|alright)\b")


def parse_intent(text: str) -> Intent | None:
    t = text.lower()
    if _STOP.search(t):
        return Intent("stop")
    if _LEARN.search(t):
        return Intent("learn")
    if _STATUS.search(t):
        return Intent("status")
    if m := _DELIVER.search(t):
        return Intent("deliver", m[2])
    return None
