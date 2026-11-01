import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from airo import meta_api

# Sample SAM 3.1 output from the meta-sam-parser docs: one box (inclusive x2, y2) in a 320x334 image.
SAM_TEXT = ("<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
            "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>\n")


async def fake_events():
    lane = {"item_id": "msg", "output_index": 0, "content_index": 0}
    yield {"type": "response.output_text.delta", "delta": SAM_TEXT, **lane}
    yield {"type": "response.content_part.done", "part": {"type": "output_text", "text": SAM_TEXT}, **lane}
    yield {"type": "response.completed"}


async def fake_create(**_):
    return fake_events()


def test_segment_maps_box_local_mask_to_frame(monkeypatch):
    fake = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(meta_api, "client", lambda: fake)

    segs = asyncio.run(meta_api.segment(np.zeros((334, 320, 3), np.uint8), "box"))
    assert len(segs) == 1
    s = segs[0]
    assert s.box == (211, 228, 271, 255)
    ys, xs = np.nonzero(s.mask)
    assert s.mask.shape == (334, 320) and len(xs) > 0
    assert xs.min() >= 211 and xs.max() < 271 and ys.min() >= 228 and ys.max() < 255


def test_segment_times_out_into_the_feed(monkeypatch):
    async def hang(**_):
        await asyncio.sleep(10)

    monkeypatch.setattr(meta_api, "client", lambda: SimpleNamespace(responses=SimpleNamespace(create=hang)))
    meta_api.FEED.clear()
    with pytest.raises(TimeoutError):
        asyncio.run(meta_api.segment(np.zeros((10, 10, 3), np.uint8), "box", timeout=0.01))
    e = meta_api.FEED[-1]
    assert (e["kind"], e["ok"]) == ("segment", False) and e["output"].startswith("TimeoutError")


def test_calls_land_in_the_feed_with_robot_image_tools_and_failures(monkeypatch):
    fake = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(meta_api, "client", lambda: fake)
    meta_api.FEED.clear()
    meta_api.ROBOT.set("airo-2")
    asyncio.run(meta_api.segment(np.zeros((334, 320, 3), np.uint8), "box"))
    seg = meta_api.FEED[-1]
    assert (seg["model"], seg["kind"], seg["robot"], seg["output"]) == ("sam-3.1", "segment", "airo-2", "1 × box")
    assert meta_api.IMAGES[seg["id"]][:2] == b"\xff\xd8" and seg["image"] == f"/models/{seg['id']}.jpg"

    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="run_probe", arguments='{"probe": "step_80"}'))
    msg = SimpleNamespace(content="Testing the gain.", tool_calls=[call])

    async def chat(**_):
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(meta_api, "client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=chat))))
    asyncio.run(meta_api.wonder([{"role": "user", "content": "Learn the body."}], []))
    meta_api.tool_result("c1", "z +0.4")
    w = meta_api.FEED[-1]
    assert w["input"] == "Learn the body."
    assert (w["tools"][0]["name"], w["tools"][0]["result"]) == ("run_probe", "z +0.4")

    async def down(**_):
        raise ConnectionError("offline")

    monkeypatch.setattr(meta_api, "client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=down))))
    try:
        asyncio.run(meta_api.describe(np.zeros((10, 10, 3), np.uint8), "What is on the car?"))
    except ConnectionError:
        pass
    assert meta_api.FEED[-1]["ok"] is False and "offline" in meta_api.FEED[-1]["output"]
