import asyncio
from types import SimpleNamespace

import numpy as np

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
