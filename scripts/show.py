"""Both dashboards behind one page: the MuJoCo warehouse and the live car, each its own demo.py that keeps running in
the background, so switching tabs mid-recording is instant. 1 and 2 switch tabs.

    uv run python scripts/show.py --camera http://<phone>:8080/video --floor 0.80 1.22 --marker-h 0.1 --cam-h 1.6

Every argument show.py does not know goes to the live car's demo.py (--camera, --floor, --marker-h, --cam-h,
--marker-offset, --address, --dry). The env file (model API key) is only read with --env-file .env.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHELL = ROOT / "web" / "shell.html"
STOP_S = 20


def demo(env_file: str | None, *args: str) -> list[str]:
    # --no-env-file: a UV_ENV_FILE in the environment must not load the paid key unasked
    env = ["--env-file", env_file] if env_file else ["--no-env-file"]
    return ["uv", "run", *env, "python", "scripts/demo.py", *args]


def commands(args, car_args: list[str]) -> dict[str, tuple[int, list[str]]]:
    host = ["--host", args.host]
    children = {}
    if not args.no_warehouse:
        children["warehouse"] = (args.warehouse_port, demo(args.env_file, "--site", "factory", "--busy", args.busy,
                                                           *host, "--port", str(args.warehouse_port)))
    if not args.no_car:
        children["car"] = (args.car_port, demo(args.env_file, *car_args, *host, "--port", str(args.car_port)))
    return children


def listening(host: str, port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1" if host == "0.0.0.0" else host, port), 0.2).close()
        return True
    except OSError:
        return False


class Child:
    def __init__(self, name: str, port: int, cmd: list[str], host: str):
        self.name, self.port, self.cmd, self.host = name, port, cmd, host
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque(maxlen=12)
        self.lock = threading.Lock()

    def start(self):
        self.log.clear()
        if listening(self.host, self.port):
            self.proc = None
            self.log.append(f"port {self.port} is already in use: is another demo running?")
            print(f"[{self.name}] {self.log[0]}", flush=True)
            return
        print(f"[{self.name}] {' '.join(self.cmd)}", flush=True)
        # own session: Ctrl+C in the terminal must reach show.py only, which then stops each demo exactly once
        self.proc = subprocess.Popen(self.cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                                     env=os.environ | {"PYTHONUNBUFFERED": "1"})
        threading.Thread(target=self.pump, args=(self.proc,), daemon=True).start()

    def pump(self, proc: subprocess.Popen):
        for line in proc.stdout:
            print(f"[{self.name}] {line}", end="", flush=True)
            self.log.append(line.rstrip())

    def stop(self):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        # uv run forwards SIGTERM to demo.py, which unwinds and closes BLE; a second signal would cut that short
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(STOP_S)
        except subprocess.TimeoutExpired:
            print(f"[{self.name}] did not stop in {STOP_S} s, killing it", flush=True)
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()

    def restart(self):
        with self.lock:
            self.stop()
            self.start()

    def status(self) -> dict:
        code = self.proc.poll() if self.proc else -1
        if code is not None:
            return {"port": self.port, "state": "exited", "code": code, "log": list(self.log)}
        return {"port": self.port, "state": "ready" if listening(self.host, self.port) else "starting"}


def serve(children: dict[str, Child], host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, kind: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(SHELL.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/status":
                self.send(json.dumps({k: c.status() for k, c in children.items()}).encode(), "application/json")
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            name = self.path.removeprefix("/retry/")
            if name not in children:
                return self.send(b"not found", "text/plain", 404)
            threading.Thread(target=children[name].restart, daemon=True).start()
            self.send(b"{}", "application/json")

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000, help="the page with both tabs")
    p.add_argument("--warehouse-port", type=int, default=8001)
    p.add_argument("--car-port", type=int, default=8002)
    p.add_argument("--busy", choices=("calm", "normal", "busy"), default="normal", help="the warehouse's floor life")
    p.add_argument("--env-file", help="passed to uv run, e.g. .env for the model API key; none by default")
    p.add_argument("--no-warehouse", action="store_true")
    p.add_argument("--no-car", action="store_true")
    return p.parse_known_args(argv)


def main():
    args, car_args = parse()
    children = {k: Child(k, port, cmd, args.host) for k, (port, cmd) in commands(args, car_args).items()}
    if not children:
        sys.exit("show: nothing to show without the warehouse and the car")
    server = serve(children, args.host, args.port)
    for c in children.values():
        c.start()
    print(f"show: http://{args.host}:{args.port}", flush=True)
    # a flag, not KeyboardInterrupt: uv run and the terminal may both deliver Ctrl+C, and a second one must not
    # cut the children's shutdown short
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    while not done.wait(0.5):
        pass
    print("show: stopping", flush=True)
    server.shutdown()
    stops = [threading.Thread(target=c.stop) for c in children.values()]
    for t in stops:
        t.start()
    for t in stops:
        t.join()
    server.server_close()


if __name__ == "__main__":
    main()
