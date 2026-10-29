"""Live dashboard: state over a websocket, the rectified floor and a cinematic view as MJPEG, commands over POST.

State contract, published as one dict (numpy values are fine, they are converted with .tolist()):

    {"t": float, "status": str, "command": str,
     "floor": {"w": m, "h": m},
     "pose": {"x", "y", "theta"} | None,
     "cloud": [[[x, y], ...], ...],                 # k predicted paths, floor metres
     "path": [[x, y], ...],                         # planned path
     "polygons": [{"label": str, "points": [[x, y], ...]}],
     "body": {"n": int, "name": str, "relearned_s": float | None, "probing_s": float | None,
              "portrait": int | None,                # bumps when GET /portrait has a new self-portrait
              "params": [{"name", "value", "key", "unit", "lo", "hi"}]},
     "surprise": {"values": [float], "threshold": float},
     "notebook": [{"t": float, "kind": "alert|hypothesis|prediction|result|conclusion|fallback", "text": str}]}

Optional fields:

    "mode": "sim" | "real",
    "episode": {"phases": [{"t": float, "kind": "drive|surprise|probe|refit|relearn", "label": str}]},
    "relearn": {"phase": "changed|testing|done", "elapsed_s": float, "probe": str, "step": int, "steps": int} | None,
    notebook result entries: "predicted", "sd", "measured", "unit" draw a prediction-vs-measurement gauge.

Floor frame: x right, y up, origin at corner marker 1. /video (set_frame) covers exactly floor w x h;
/view (set_view) is the hero view: a 3/4 sim render with ghost cars, or the raw phone frame on hardware.
"""

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB = Path(__file__).resolve().parent.parent / "web"
MJPEG = "multipart/x-mixed-replace; boundary=frame"


def _encode(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


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
        self._jpeg: dict[str, bytes] = {}
        self.portrait: bytes | None = None
        self.sim: dict = {}
        self.camera: asyncio.Queue[tuple[dict, asyncio.Future]] = asyncio.Queue()
        self.app = self._routes(FastAPI())

    def publish(self, state: dict) -> None:
        self.state = state

    def set_frame(self, bgr: np.ndarray) -> None:
        self._jpeg["video"] = _encode(bgr)

    def set_view(self, bgr: np.ndarray) -> None:
        self._jpeg["view"] = _encode(bgr)

    async def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        await uvicorn.Server(uvicorn.Config(self.app, host=host, port=port, log_level="warning")).serve()

    async def _mjpeg(self, name: str):
        last = None
        while True:
            if (jpeg := self._jpeg.get(name)) is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 30)

    def _routes(self, app: FastAPI) -> FastAPI:
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/")
        def index():
            return FileResponse(WEB / "index.html")

        @app.get("/portrait")
        def portrait():
            return Response(self.portrait, media_type="image/png") if self.portrait else Response(status_code=404)

        @app.post("/cmd")
        def cmd(c: Command):
            self.commands.put_nowait(c.text)
            return {"ok": True}

        @app.post("/camera")
        async def camera(req: dict):
            """Handled by whoever renders the view (demo.film); nobody answers on hardware."""
            fut = asyncio.get_running_loop().create_future()
            self.camera.put_nowait((req, fut))
            try:
                return await asyncio.wait_for(fut, 1)
            except TimeoutError:
                return {"camera": None, "part": None}

        @app.get("/video")
        def video():
            return StreamingResponse(self._mjpeg("video"), media_type=MJPEG)

        @app.get("/view")
        def view():
            return StreamingResponse(self._mjpeg("view"), media_type=MJPEG)

        @app.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            try:
                while True:
                    await sock.send_text(json.dumps({**self.state, "sim": self.sim}, default=lambda o: o.tolist()))
                    await asyncio.sleep(self.period)
            except WebSocketDisconnect:
                pass

        return app
