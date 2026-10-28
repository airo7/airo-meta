import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from websockets.asyncio.server import serve

from airo import meta_api, voice
from airo.voice import Intent, parse_intent


@pytest.mark.parametrize("text, intent", [
    ("take the parcel to the blue station", Intent("deliver", "blue")),
    ("Deliver to red.", Intent("deliver", "red")),
    ("bring it to station green please", Intent("deliver", "green")),
    ("deliver the parcel", Intent("deliver", None)),
    ("learn your body", Intent("learn")),
    ("OK, relearn yourself", Intent("learn")),
    ("Stop!", Intent("stop")),
    ("stop taking it to blue", Intent("stop")),
    ("how do you feel", Intent("status")),
    ("Hey, how are you doing?", Intent("status")),
    ("what a nice day", None),
])
def test_parse_intent(text, intent):
    assert parse_intent(text) == intent


async def silence(device=None):
    for _ in range(3):
        yield bytes(voice.FRAME * 2)


def run_transcribe(monkeypatch, handler, mic=silence):
    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setattr(voice, "REALTIME_URL", f"ws://127.0.0.1:{port}")
            got = []
            await asyncio.wait_for(voice.transcribe(lambda t, final: got.append((t, final))), 5)
            return got

    monkeypatch.setenv("MODEL_API_KEY", "sk-test")
    monkeypatch.setattr(voice, "mic", mic)
    return asyncio.run(main())


def test_transcribe_follows_realtime_flow(monkeypatch):
    seen = {}

    async def server(ws):
        seen["handshake"] = json.loads(await ws.recv())
        await ws.send(json.dumps({"sessionId": "s1"}))
        frames = []
        async for msg in ws:
            if isinstance(msg, str):
                seen["end"] = json.loads(msg)
                break
            frames.append(msg)
            if len(frames) == 1:
                await ws.send(json.dumps({"type": "speechStart", "turnId": 1, "audioProcessedMs": 80}))
                await ws.send(json.dumps({"type": "transcript", "transcript": "take the", "final": False,
                                          "audioProcessedMs": 160}))
        seen["frames"] = frames
        for ev in [{"type": "speechEnd", "turnId": 1, "audioProcessedMs": 240},
                   {"type": "speechComplete", "turnId": 1, "transcript": "Take the parcel to blue.",
                    "audioProcessedMs": 240},
                   {"type": "somethingNew"}]:
            await ws.send(json.dumps(ev))
        await ws.close(1000)

    got = run_transcribe(monkeypatch, server)

    hs = seen["handshake"]
    assert hs["authorization"] == {"accessToken": "Bearer sk-test"}
    assert hs["audioEncoding"] == "PCM_24KHZ" and hs["model"] == "muse-voice-transcribe-1.0"
    assert [len(f) for f in seen["frames"]] == [3840] * 3
    assert seen["end"] == {"type": "endStream"}
    assert got == [("take the", False), ("Take the parcel to blue.", True)]


def test_transcribe_raises_on_server_error(monkeypatch):
    async def server(ws):
        await ws.recv()
        await ws.send(json.dumps({"sessionId": "s2"}))
        await ws.send(json.dumps({"type": "error", "message": "bad audio", "sessionId": "s2"}))
        await ws.close(1008)

    with pytest.raises(RuntimeError, match="bad audio"):
        run_transcribe(monkeypatch, server)


def test_transcribe_surfaces_mic_failure(monkeypatch):
    async def server(ws):
        await ws.recv()
        await ws.send(json.dumps({"sessionId": "s3"}))
        await ws.wait_closed()

    async def broken_mic(device=None):
        raise OSError("no input device")
        yield

    with pytest.raises(OSError, match="no input device"):
        run_transcribe(monkeypatch, server, broken_mic)


def fake_api(monkeypatch, result, delay=0.0):
    calls = []

    async def create(**kw):
        calls.append(kw)
        await asyncio.sleep(delay)
        return result

    monkeypatch.setattr(meta_api, "client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), images=SimpleNamespace(generate=create)))
    return calls


def test_wonder_passes_tools_and_returns_message(monkeypatch):
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
        name="get_body", arguments="{}"))])
    calls = fake_api(monkeypatch, SimpleNamespace(choices=[SimpleNamespace(message=msg)]))

    tools = [{"type": "function", "function": {"name": "get_body", "parameters": {"type": "object"}}}]
    out = asyncio.run(meta_api.wonder([{"role": "user", "content": "hi"}], tools))
    assert out is msg
    assert calls == [{"model": "muse-wonder-1.0", "messages": [{"role": "user", "content": "hi"}], "tools": tools}]


def test_wonder_times_out(monkeypatch):
    fake_api(monkeypatch, None, delay=1)
    with pytest.raises(TimeoutError):
        asyncio.run(meta_api.wonder([], [], timeout=0.01))


def test_image_decodes_png(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n fake"
    calls = fake_api(monkeypatch, SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(png).decode())]))

    assert asyncio.run(meta_api.image("a robot car self-portrait")) == png
    assert calls[0]["model"] == "muse-image-1.0" and calls[0]["output_format"] == "png"
