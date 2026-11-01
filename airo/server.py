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
     "notebook": [{"t": float, "kind": "alert|hypothesis|prediction|result|conclusion|fallback|eye", "text": str}]}

Optional fields:

    "mode": "sim" | "real",
    "episode": {"phases": [{"t": float, "kind": "drive|surprise|probe|refit|relearn", "label": str}]},
    "relearn": {"phase": "changed|testing|done", "elapsed_s": float, "probe": str, "step": int, "steps": int} | None,
    notebook result entries: "predicted", "sd", "measured", "unit" draw a prediction-vs-measurement gauge;
    any entry's "detail" is the technical line behind its plain "text".

Floor frame: x right, y up, origin at corner marker 1. /video (set_frame) covers exactly floor w x h;
/view (set_view) is the hero view: a 3/4 sim render with ghost cars, or the raw phone frame on hardware.
"""

import asyncio
import json
import logging
from contextlib import suppress

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from airo import meta_api
from airo.paths import DATA, WEB

MJPEG = "multipart/x-mixed-replace; boundary=frame"
PORTRAITS = DATA / "portraits"
log = logging.getLogger(__name__)


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
        # the MJPEG streams never end, so a graceful shutdown with no timeout would wait on them forever
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=1)
        await uvicorn.Server(config).serve()

    async def _mjpeg(self, name: str):
        last = None
        while True:
            if (jpeg := self._jpeg.get(name)) is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 30)

    def _routes(self, app: FastAPI) -> FastAPI:
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.middleware("http")
        async def revalidate(request, call_next):
            # Browsers otherwise keep an old dash.js after an update and the page breaks half-new.
            response = await call_next(request)
            if request.url.path == "/" or request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache"
            return response
        PORTRAITS.mkdir(parents=True, exist_ok=True)
        app.mount("/portraits", StaticFiles(directory=PORTRAITS), name="portraits")

        @app.get("/")
        def index():
            return FileResponse(WEB / "index.html")

        @app.get("/portrait")
        def portrait():
            return Response(self.portrait, media_type="image/png") if self.portrait else Response(status_code=404)

        @app.post("/cmd")
        async def cmd(c: Command):
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

        @app.get("/models/{i}.jpg")
        def model_image(i: int):
            jpg = meta_api.IMAGES.get(i)
            return Response(jpg, media_type="image/jpeg") if jpg else Response(status_code=404)

        @app.websocket("/voice")
        async def voice(sock: WebSocket):
            """Push-to-talk from the page: its mic audio goes to Muse Voice, partials come back, the final transcript
            becomes a command."""
            from airo.voice import FRAME, transcribe

            await sock.accept()
            audio: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def frames():
                buf = b""
                while (chunk := await audio.get()) is not None:
                    buf += chunk
                    while len(buf) >= 2 * FRAME:
                        yield buf[:2 * FRAME]
                        buf = buf[2 * FRAME:]

            async def receive():
                with suppress(WebSocketDisconnect):
                    while (msg := await sock.receive())["type"] != "websocket.disconnect" and msg.get("text") != "end":
                        if msg.get("bytes"):
                            audio.put_nowait(msg["bytes"])
                audio.put_nowait(None)

            def heard(text: str, final: bool):
                if final and text.strip():
                    self.commands.put_nowait(text)
                asyncio.create_task(sock.send_json({"text": text, "final": final}))

            listening = asyncio.create_task(receive())
            try:
                await transcribe(heard, frames=frames())
            except Exception as e:
                log.exception("voice request failed")
                with suppress(Exception):
                    await sock.send_json({"error": f"{type(e).__name__}: {e}"})
            finally:
                listening.cancel()
                with suppress(Exception):
                    await sock.close()

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
                    state = {**self.state, "sim": self.sim, "models": list(meta_api.FEED)}
                    await sock.send_text(json.dumps(state, default=lambda o: o.tolist()))
                    await asyncio.sleep(self.period)
            except WebSocketDisconnect:
                pass

        return app
