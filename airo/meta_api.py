"""Meta Model API: SAM 3.1 segmentation and Muse Spark vision, via the OpenAI-compatible endpoint."""

import base64
import os
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
from meta_sam_parser import (SegmentationBoxRecord, SegmentationMaskRecord, decode_mask_to_raster,
                             image_segmentation_format, parse_responses_stream)
from openai import AsyncOpenAI

BASE_URL = "https://api.meta.ai/v1"


@dataclass
class Segment:
    box: tuple[int, int, int, int]
    mask: np.ndarray
    score: float | None


@cache
def client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])


def _data_url(image_bgr: np.ndarray, max_side: int = 1024) -> tuple[str, float]:
    scale = min(1.0, max_side / max(image_bgr.shape[:2]))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(), scale


async def segment(image_bgr: np.ndarray, concept: str) -> list[Segment]:
    """All instances of one short noun phrase; boxes and masks in original frame pixels."""
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


async def describe(image_bgr: np.ndarray, prompt: str, model: str = "muse-spark-1.3") -> str:
    url, _ = _data_url(image_bgr)
    resp = await client().chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ]}])
    return resp.choices[0].message.content or ""
