"""Live dashboard: state over a websocket, the rectified floor as MJPEG, commands back over POST.

State contract, published as one dict (numpy values are fine, they are converted with .tolist()):

    {"t": float, "status": str, "command": str,
     "floor": {"w": m, "h": m},
     "pose": {"x", "y", "theta"} | None,
     "cloud": [[[x, y], ...], ...],                 # k predicted paths, floor metres
     "path": [[x, y], ...],                         # planned path
     "polygons": [{"label": str, "points": [[x, y], ...]}],
     "body": {"n": int, "name": str, "relearned_s": float | None,
              "params": [{"name", "value", "key", "unit", "lo", "hi"}]},
     "surprise": {"values": [float], "threshold": float},
     "notebook": [{"t": float, "kind": "alert|hypothesis|prediction|result|conclusion|fallback", "text": str}]}

Floor frame: x right, y up, origin at corner marker 1. The video frame covers exactly floor w x h.
"""

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB = Path(__file__).resolve().parent.parent / "web"


class Command(BaseModel):
    text: str


def topdown(frame: np.ndarray, H: np.ndarray, floor_wh: tuple[float, float], px_per_m: float = 400) -> np.ndarray:
    """Warp a camera frame to the floor rectangle; H maps image pixels to floor metres."""
    w, h = floor_wh
    to_px = np.array([[px_per_m, 0, 0], [0, -px_per_m, h * px_per_m], [0, 0, 1]])
    return cv2.warpPerspective(frame, to_px @ H, (round(w * px_per_m), round(h * px_per_m)))


class Dashboard:
    def __init__(self, hz: float = 10):
        self.state: dict = {}
        self.commands: asyncio.Queue[str] = asyncio.Queue()
        self.period = 1 / hz
        self._jpeg: bytes | None = None
        self.app = self._routes(FastAPI())

    def publish(self, state: dict) -> None:
        self.state = state

    def set_frame(self, bgr: np.ndarray) -> None:
        self._jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    async def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        await uvicorn.Server(uvicorn.Config(self.app, host=host, port=port, log_level="warning")).serve()

    async def _mjpeg(self):
        last = None
        while True:
            if self._jpeg is not last:
                last = self._jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
            await asyncio.sleep(1 / 30)

    def _routes(self, app: FastAPI) -> FastAPI:
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB / "index.html")

        @app.post("/cmd")
        def cmd(c: Command):
            self.commands.put_nowait(c.text)
            return {"ok": True}

        @app.get("/video")
        def video():
            return StreamingResponse(self._mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

        @app.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            try:
                while True:
                    await sock.send_text(json.dumps(self.state, default=lambda o: o.tolist()))
                    await asyncio.sleep(self.period)
            except WebSocketDisconnect:
                pass

        return app
