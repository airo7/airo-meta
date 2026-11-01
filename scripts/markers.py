"""Printable A4 marker pages at 300 DPI. Print at 100% and check the 100 mm bar with a ruler."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from airo.camera import CAR_ID, CAR_MARKER_M, CORNER_MARKER_M, DICT
from airo.paths import DATA

DPI = 300
A4 = (2480, 3508)


def mm(v: float) -> int:
    return round(v / 25.4 * DPI)


def page(marker_id: int, size_m: float) -> np.ndarray:
    w, h = A4
    img = np.full((h, w), 255, np.uint8)
    side = mm(size_m * 1000)
    x, y = (w - side) // 2, (h - side) // 2
    img[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(DICT, marker_id, side)

    text = f"ID {marker_id}  {size_m * 1000:.0f} mm  top edge = forward / +y"
    cv2.putText(img, text, (mm(15), y - mm(15)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 5)
    cv2.rectangle(img, (mm(15), h - mm(30)), (mm(115), h - mm(25)), 0, -1)
    cv2.putText(img, "100 mm", (mm(15), h - mm(35)), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    return img


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    sizes = {CAR_ID: CAR_MARKER_M} | {i: CORNER_MARKER_M for i in range(1, 5)}
    for marker_id, size in sizes.items():
        path = out / f"marker_{marker_id}.png"
        Image.fromarray(page(marker_id, size)).save(path, dpi=(DPI, DPI))
        print(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DATA / "markers")
    main(p.parse_args().out)
