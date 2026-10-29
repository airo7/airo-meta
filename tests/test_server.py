import asyncio

import cv2
import numpy as np
from fastapi.testclient import TestClient

from airo.camera import to_floor
from airo.server import Dashboard, topdown


def test_index_cmd_and_ws():
    dash = Dashboard()
    client = TestClient(dash.app)

    r = client.get("/")
    assert r.status_code == 200 and "dash.js" in r.text
    assert client.get("/static/dash.js").status_code == 200

    assert client.post("/cmd", json={"text": "deliver to blue"}).json() == {"ok": True}
    assert dash.commands.get_nowait() == "deliver to blue"

    dash.publish({"t": 1.5, "pose": {"x": 0.5, "y": 0.25, "theta": 0.0}, "path": np.array([[0.0, 0.0], [1.0, 1.0]])})
    with client.websocket_connect("/ws") as ws:
        state = ws.receive_json()
    assert state["t"] == 1.5 and state["path"] == [[0.0, 0.0], [1.0, 1.0]]


def test_topdown_known_homography():
    H = np.array([[0.002, 0.0001, -0.2], [5e-5, -0.002, 1.6], [1e-5, 5e-6, 1.0]])
    img_pt = np.array([[640.0, 360.0]])
    x, y = to_floor(H, img_pt)[0]

    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[355:366, 635:646] = 255
    out = topdown(frame, H, (2.0, 1.5), px_per_m=200)

    assert out.shape == (300, 400, 3)
    v, u = np.argwhere(out[..., 0] > 128).mean(axis=0)
    assert abs(u - x * 200) < 1.5 and abs(v - (1.5 - y) * 200) < 1.5


def test_view_stream_serves_latest_frame():
    dash = Dashboard()
    assert {"/video", "/view"} <= {r.path for r in dash.app.routes}
    dash.set_frame(np.zeros((10, 20, 3), np.uint8))
    dash.set_view(np.full((90, 160, 3), 200, np.uint8))

    async def first(name):
        stream = dash._mjpeg(name)
        chunk = await anext(stream)
        await stream.aclose()
        return chunk

    chunk = asyncio.run(first("view"))
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    jpeg = chunk.split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (90, 160, 3)


def test_portrait_endpoint():
    dash = Dashboard()
    client = TestClient(dash.app)
    assert client.get("/portrait").status_code == 404
    dash.portrait = b"png"
    r = client.get("/portrait")
    assert r.status_code == 200 and r.content == b"png" and r.headers["content-type"] == "image/png"
