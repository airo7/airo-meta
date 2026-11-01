"""Meta Model API: SAM 3.1, Muse Spark, Muse Wonder and Muse Image, via the OpenAI-compatible endpoint.

Every call lands in FEED (newest last) for the dashboard; ROBOT names the robot whose task made the call.
"""

import asyncio
import base64
import itertools
import os
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
from meta_sam_parser import (SegmentationBoxRecord, SegmentationMaskRecord, decode_mask_to_raster,
                             image_segmentation_format, parse_responses_stream)
from openai import AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"
ROBOT: ContextVar[str | None] = ContextVar("robot", default=None)
FEED: deque[dict] = deque(maxlen=40)
IMAGES: dict[int, bytes] = {}
_ids = itertools.count(1)


@dataclass
class Segment:
    box: tuple[int, int, int, int]
    mask: np.ndarray
    score: float | None


def record(model: str, kind: str, input: str, output: str, ms: float, ok: bool = True, image: bytes | None = None,
           tools: list[dict] | None = None) -> dict:
    i = next(_ids)
    if len(FEED) == FEED.maxlen:
        IMAGES.pop(FEED[0]["id"], None)
    if image:
        IMAGES[i] = image
    e = {"id": i, "t": time.time(), "robot": ROBOT.get(), "model": model, "kind": kind, "input": input[:400],
         "output": output[:800], "ms": round(ms), "ok": ok, "image": f"/models/{i}.jpg" if image else None,
         "tools": tools}
    FEED.append(e)
    return e


def tool_result(call_id: str, result: str, plain: str | None = None):
    """Fill in a Wonder tool call's result once the engine has run it; plain is the same in words for the feed."""
    for e in reversed(FEED):
        for t in e["tools"] or ():
            if t.get("id") == call_id:
                t["result"] = result[:200]
                if plain:
                    t["plain"] = plain[:240]
                return


@contextmanager
def _call(model: str, kind: str, input: str):
    """Yields a dict the caller fills with output/image/tools; failures are recorded and re-raised."""
    out = {"output": "", "image": None, "tools": None}
    t0 = time.perf_counter()
    try:
        yield out
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        record(model, kind, input, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000, ok=False)
        raise
    record(model, kind, input, out["output"], (time.perf_counter() - t0) * 1000, image=out["image"],
           tools=out["tools"])


def _thumb(image_bgr: np.ndarray | None, width: int = 360) -> bytes | None:
    if image_bgr is None:
        return None
    small = cv2.resize(image_bgr, (width, round(image_bgr.shape[0] * width / image_bgr.shape[1])),
                       interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def _overlay(image_bgr: np.ndarray, segs: list["Segment"]) -> np.ndarray:
    out = image_bgr.copy()
    for s in segs:
        out[s.mask] = (0.45 * out[s.mask] + (140, 90, 0)).astype(np.uint8)
        cv2.rectangle(out, s.box[:2], s.box[2:], (255, 170, 60), 3)
    return out


@cache
def client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])


def _data_url(image_bgr: np.ndarray, max_side: int = 1024) -> tuple[str, float]:
    scale = min(1.0, max_side / max(image_bgr.shape[:2]))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(), scale


async def segment(image_bgr: np.ndarray, concept: str, timeout: float = 8) -> list[Segment]:
    """All instances of one short noun phrase; boxes and masks in original frame pixels."""
    with _call("sam-3.1", "segment", concept) as rec:
        out = await asyncio.wait_for(_segment(image_bgr, concept), timeout)
        scores = ", ".join(f"{s.score:.2f}" for s in out if s.score is not None)
        rec["output"] = f"{len(out)} × {concept}" + (f" ({scores})" if scores else "")
        rec["image"] = _thumb(_overlay(image_bgr, out))
    return out


async def _segment(image_bgr: np.ndarray, concept: str) -> list[Segment]:
    url, scale = _data_url(image_bgr)
    stream = await client().responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": url},
        ]}],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"},
        stream=True,
    )
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()

    h, w = image_bgr.shape[:2]
    boxes = {r.object_id: r for r in result.records if isinstance(r, SegmentationBoxRecord)}
    out = []
    for r in result.records:
        if not isinstance(r, SegmentationMaskRecord):
            continue
        # Masks are box-local rasters; bounds place them in the sent image.
        b = r.bounds
        x0, y0 = max(0, round(b.left / scale)), max(0, round(b.top / scale))
        x1, y1 = min(w, round(b.right / scale)), min(h, round(b.bottom / scale))
        if x1 <= x0 or y1 <= y0:
            continue
        raster = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = cv2.resize(raster, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
        score = r.confidence if r.confidence is not None else getattr(boxes.get(r.object_id), "confidence", None)
        out.append(Segment((x0, y0, x1, y1), full, score))
    return out


async def describe(image_bgr: np.ndarray, prompt: str, model: str = "muse-spark-1.3", timeout: float = 15) -> str:
    with _call(model, "describe", prompt) as rec:
        url, _ = _data_url(image_bgr)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
        resp = await asyncio.wait_for(client().chat.completions.create(model=model, messages=messages), timeout)
        rec["output"], rec["image"] = resp.choices[0].message.content or "", _thumb(image_bgr)
    return rec["output"]


async def wonder(messages: list[dict], tools: list[dict], model: str = "muse-wonder-1.0", timeout: float = 30):
    """One chat.completions turn with function tools; returns the assistant message (may carry tool_calls)."""
    n = next((i for i, m in enumerate(reversed(messages)) if m.get("role") != "tool"), len(messages))
    asked = f"{n} tool result{'s' * (n > 1)}" if n else (messages[-1].get("content") or "" if messages else "")
    with _call(model, "wonder", asked if isinstance(asked, str) else "") as rec:
        resp = await asyncio.wait_for(client().chat.completions.create(model=model, messages=messages, tools=tools),
                                      timeout)
        msg = resp.choices[0].message
        rec["output"] = msg.content or ""
        rec["tools"] = [{"id": c.id, "name": c.function.name, "args": c.function.arguments, "result": None}
                        for c in msg.tool_calls or []] or None
    return msg


async def image(prompt: str, model: str = "muse-image-1.0", timeout: float = 60) -> bytes:
    with _call(model, "image", prompt) as rec:
        # output_format defaults to webp: https://dev.meta.ai/docs/api-reference/images/schemas
        resp = await asyncio.wait_for(client().images.generate(model=model, prompt=prompt, n=1, output_format="png",
                                                               response_format="b64_json"), timeout)
        png = base64.b64decode(resp.data[0].b64_json)
        rec["output"], rec["image"] = "self-portrait", _thumb(cv2.imdecode(np.frombuffer(png, np.uint8),
                                                                           cv2.IMREAD_COLOR), 480)
    return png
