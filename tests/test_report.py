import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np

from airo.body import KEY
from airo.mission import Mission
from airo.sim import SimRig

spec = importlib.util.spec_from_file_location("report", Path(__file__).parents[1] / "scripts/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

BODY = {"g_l": 0.0031, "g_r": 0.0030, "dz": 24.0, "eta": 0.31, "tau": 0.18, "lag": 0.1, "kappa": -0.04}


def _ticks(n=40, dry=False):
    """Half the ticks drive, every other tick's command goes out."""
    return [{"t": k * 0.05, "pwm_l": 100 * (k >= n // 2), "pwm_r": 100 * (k >= n // 2), "sent": not dry and k % 2 == 0,
             "x": None if k == 3 else 1.0, "y": 0.7, "theta": 0.0, "pose_t": k * 0.05} for k in range(n)]


def _notes():
    z = lambda *v: dict(zip(("forward_cm", "lateral_cm", "turn_deg"), v))
    return [
        {"t": 100.0, "kind": "command", "text": "learn", "intent": "learn"},
        {"t": 101.0, "kind": "hypothesis", "text": "pivot", "probe": "pivot"},
        {"t": 103.0, "kind": "result", "text": "pivot: ...", "z": z(0.5, -1.0, 1.5)},
        {"t": 106.0, "kind": "result", "text": "step: ...", "z": z(2.5, 0.1, 0.0)},
        {"t": 130.0, "kind": "conclusion", "text": "Bare chassis.", "n": 1, "name": "bare chassis", "body": BODY},
        {"t": 140.0, "kind": "result", "text": "Arrived at (1.00, 0.70), 2.0 cm off."},
        {"t": 150.0, "kind": "result", "text": "Arrived at (0.40, 0.40), 7.5 cm off."},
        {"t": 160.0, "kind": "result", "text": "Arrived at (1.60, 0.40), 3.0 cm off."},
        {"t": 200.0, "kind": "alert", "text": "Body changed, surprise 4.1 over the last 1.5 s: ..."},
        {"t": 204.0, "kind": "result", "text": "step: ...", "z": z(4.0, 0.2, 0.3)},
        {"t": 209.0, "kind": "result", "text": "pivot: ...", "z": z(0.2, 0.2, 0.3)},
        {"t": 212.0, "kind": "conclusion", "text": "Heavier.", "n": 2, "name": "loaded", "body": BODY, "fallback": True},
        {"t": 212.5, "kind": "conclusion", "text": "Body #2, relearned in 12.6 s (9.0 s probing)."},
        {"t": 300.0, "kind": "alert", "text": "Body changed, surprise 3.5 over the last 1.5 s: ..."},
        {"t": 303.0, "kind": "result", "text": "pivot: ...", "z": z(0.1, 0.1, 0.4)},
        {"t": 303.1, "kind": "recall", "text": "I know this one", "n": 1, "name": "bare chassis", "recall_s": 3.2},
        {"t": 303.2, "kind": "conclusion", "text": "Body #1, recalled in 3.3 s (2.1 s probing)."},
        {"t": 320.0, "kind": "alert", "text": "Safety stop: pose lost."},
    ]


def test_episodes_split_learning_relearning_and_recall():
    learn, relearn, recall = report.episodes(_notes())
    assert (learn["kind"], learn["t1"] - learn["t0"], len(learn["z"])) == ("learn", 30.0, 2)
    assert learn["name"] == "bare chassis" and not learn["scripted"]
    assert (relearn["how"], relearn["n"], relearn["seconds"], relearn["scripted"]) == ("relearned", 2, 12.6, True)
    assert relearn["t1"] - relearn["t0"] == 12.5
    assert (recall["how"], recall["n"], recall["recall_s"], len(recall["z"])) == ("recalled", 1, 3.2, 1)


def test_summary_rows():
    rows = dict(report.summary(_ticks(), _notes()))
    assert "1/2 within 2σ" in rows["learn"] and "Wonder" in rows["learn"] and "g 3.10/3.00 mm/s/PWM" in rows["learn"]
    assert rows["go-to"] == "3 arrived, 0 gave up; stop error median 3.0 cm, max 7.5 cm, 2/3 (67%) within 5 cm"
    assert rows["safety"] == "1 stops; Safety stop: pose lost"
    assert "sent 20/40 (50%)" in rows["ble"] and "driving 10/20 (50%), 10.0 Hz" in rows["ble"]
    assert "39/40" in rows["ticks"]
    surprises = [v for k, v in report.summary([], _notes()) if k == "surprise"]
    assert "relearned Body #2" in surprises[0] and "recalled Body #1" in surprises[1]


def test_tick_log_alone():
    rows = dict(report.summary(_ticks(dry=True), []))
    assert rows["ble"] == "nothing sent (dry run)" and rows["notes"] == "none"
    assert dict(report.summary([], [])) == {"ticks": "none", "notes": "none"}


def test_paths_pick_the_latest_run_with_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS", tmp_path)
    (tmp_path / "20261031-100000.jsonl").write_text(json.dumps(_ticks()[0]) + "\n")
    (tmp_path / "20261031-110000.jsonl").write_text("")
    (tmp_path / "20261031-120000.notes.jsonl").write_text("{}\n")
    assert report.paths(None)[0].name == "20261031-100000.jsonl"
    ticks, notes = report.paths("20261031-110000")
    assert (ticks.name, notes.name) == ("20261031-110000.jsonl", "20261031-110000.notes.jsonl")
    assert report.paths(str(notes))[0] == ticks


class _Scientist:
    def __init__(self):
        self.best, self.n, self.name, self.notebook, self.members = KEY, 1, "bare chassis", [], [KEY]
        self.on_update, self.updates = self.updates_append, 0

    def updates_append(self, sci):
        self.updates += 1


def test_mission_persists_the_notebook_next_to_the_tick_log(tmp_path):
    rig = SimRig(KEY, log_dir=str(tmp_path))
    sci = _Scientist()
    m = Mission(rig, sci)
    sci.notebook.append({"t": 1.0, "kind": "result", "text": "pivot", "z": {"turn_deg": np.float32(1.5)}})
    sci.on_update(sci)
    m.note("result", "Arrived at (1.00, 0.70), 2.0 cm off.")
    asyncio.run(m.handle("status"))
    asyncio.run(rig.step(0, 0))
    asyncio.run(rig.close())
    notes = report.read(rig.log_path.with_suffix(".notes.jsonl"))
    assert [n["kind"] for n in notes] == ["result", "result", "command", "result"]
    assert notes[0]["z"] == {"turn_deg": 1.5} and notes[2]["intent"] == "status" and sci.updates == 1
    assert len(report.read(rig.log_path)) == 1


def test_no_notes_without_a_tick_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = Mission(SimRig(KEY, log_dir=None), _Scientist())
    m.note("alert", "Safety stop: pose lost.")
    assert not list(tmp_path.rglob("*"))
