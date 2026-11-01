"""Numbers from one run: learning, go-to drives, surprises, safety stops and the BLE command rate.

uv run scripts/report.py [STEM] [--md]. STEM picks data/runs/STEM.jsonl, the rig's tick log, and STEM.notes.jsonl, the
mission's notebook next to it; default the latest run with ticks. Without notes only the tick log is summarised.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

RUNS = Path("data/runs")
ARRIVED = re.compile(r"Arrived at \(([-\d.]+), ([-\d.]+)\), ([\d.]+) cm off")
# Older runs carry these only in the text; newer notes say it in fields ("changed", "how").
BACK = re.compile(r"Body #(\d+), (relearned|recalled) in ([\d.]+) s")
SURPRISE = ("Body changed", "Reality left the cloud")
STOPS = ("Safety stop", "Hardware fault")
NEAR_CM = 5
WITHIN_Z = 2


def paths(stem: str | None) -> tuple[Path, Path]:
    if stem is None:
        logs = [p for p in RUNS.glob("*.jsonl") if not p.name.endswith(".notes.jsonl") and p.stat().st_size]
        if not logs:
            raise SystemExit(f"no runs in {RUNS}")
        ticks = max(logs, key=lambda p: p.name)
    else:
        ticks = Path(stem) if stem.endswith(".jsonl") else RUNS / f"{stem}.jsonl"
        ticks = ticks.with_name(ticks.name.replace(".notes.jsonl", ".jsonl"))
    return ticks, ticks.with_suffix(".notes.jsonl")


def read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a run killed mid-write leaves half a line
    return rows


def episodes(notes: list[dict]) -> list[dict]:
    """Learning runs and surprises in time order. A surprise ends at the mission's "relearned/recalled in" note, a
    learning run at the scientist's conclusion, either at a safety stop."""
    out, cur = [], None
    for e in sorted(notes, key=lambda e: e["t"]):
        kind, text = e["kind"], e["text"]
        if (kind == "command" and e.get("intent") == "learn") or (kind == "hypothesis" and cur is None):
            cur = {"kind": "learn", "t0": e["t"], "z": []}
            out.append(cur)
        elif kind == "alert" and (e.get("changed") or text.startswith(SURPRISE)) and (cur is None or
                                                                                      cur["kind"] != "surprise"):
            cur = {"kind": "surprise", "t0": e["t"], "z": []}
            out.append(cur)
        elif cur is None:
            continue
        elif kind == "result" and "z" in e:
            cur["z"].append(e["z"])
        elif kind == "recall":
            cur.update(name=e.get("name"), recall_s=e.get("recall_s"))
        elif kind == "conclusion" and "body" in e:
            cur.update(body=e["body"], name=e.get("name"), scripted=bool(e.get("fallback")),
                       uncertain=bool(e.get("uncertain")))
            if cur["kind"] == "learn":
                cur["t1"], cur = e["t"], None
        elif kind == "conclusion" and "how" in e:
            cur.update(t1=e["t"], how=e["how"], n=e["n"], seconds=e["seconds"])
            cur = None
        elif kind == "conclusion" and (m := BACK.match(text)):
            cur.update(t1=e["t"], how=m[2], n=int(m[1]), seconds=float(m[3]))
            cur = None
        elif kind == "alert" and text.startswith(STOPS):
            cur.update(t1=e["t"], stopped=text)
            cur = None
    return out


def ble(ticks: list[dict]) -> dict:
    s = ticks[-1]["t"] - ticks[0]["t"]
    moving = [r for r in ticks if r["pwm_l"] or r["pwm_r"]]
    return {"ticks": len(ticks), "s": s, "sent": sum(r["sent"] for r in ticks), "moving": len(moving),
            "moving_s": len(moving) * s / (len(ticks) - 1), "moving_sent": sum(r["sent"] for r in moving),
            "pose": sum(r["x"] is not None for r in ticks)}


def body(b: dict) -> str:
    return (f"g {b['g_l'] * 1000:.2f}/{b['g_r'] * 1000:.2f} mm/s/PWM, dz {b['dz']:.0f}, eta {b['eta']:.2f}, "
            f"tau {b['tau'] * 1000:.0f} ms, lag {b['lag'] * 1000:.0f} ms, kappa {b['kappa']:+.3f}")


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def _clock(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:02.0f}"


def _probes(ep: dict) -> str:
    ok = sum(all(abs(v) < WITHIN_Z for v in z.values()) for z in ep["z"])
    return f"{len(ep['z'])} experiments, {ok}/{len(ep['z'])} within {WITHIN_Z}σ"


def summary(ticks: list[dict], notes: list[dict]) -> list[tuple[str, str]]:
    rows = []
    if len(ticks) > 1:
        b = ble(ticks)
        rows.append(("ticks", f"{b['ticks']} over {b['s']:.0f} s, {(b['ticks'] - 1) / b['s']:.1f} Hz; "
                              f"fresh pose on {_pct(b['pose'], b['ticks'])}"))
        if not b["sent"]:
            rows.append(("ble", "nothing sent (dry run)"))
        else:
            moving = (f"; driving {_pct(b['moving_sent'], b['moving'])}, {b['moving_sent'] / b['moving_s']:.1f} Hz"
                      if b["moving_s"] else "")
            rows.append(("ble", f"sent {_pct(b['sent'], b['ticks'])} of ticks, {b['sent'] / b['s']:.1f} Hz{moving}"))
    else:
        rows.append(("ticks", "none"))
    if not notes:
        return rows + [("notes", "none")]
    start = min(e["t"] for e in notes)
    for ep in episodes(notes):
        at = _clock(ep["t0"] - start)
        took = f"{ep['t1'] - ep['t0']:.1f} s" if "t1" in ep else "unfinished"
        if "stopped" in ep:
            took = f"stopped after {took} ({ep['stopped'].rstrip('.')})"
        via = ("scripted" if ep["scripted"] else "Wonder") if "scripted" in ep else ""
        if ep["kind"] == "learn":
            got = f"; {via}: \"{ep['name']}\", {body(ep['body'])}" if "body" in ep else ""
            got += ", uncertain: the fit was discarded" if ep.get("uncertain") else ""
            rows.append(("learn", f"at {at}, {took}, {_probes(ep)}{got}"))
        else:
            how = (f"{ep['how']} Body #{ep['n']} \"{ep.get('name', '')}\" in {ep['seconds']:.1f} s"
                   if "how" in ep else "no conclusion")
            recall = f", recall probe {ep['recall_s']:.1f} s" if ep.get("recall_s") is not None else ""
            via = f" ({via})" if via else ""
            rows.append(("surprise", f"at {at}: {how}{via}; alert to conclusion {took}{recall}; {_probes(ep)}"))
    offs = [float(m[3]) for e in notes if e["kind"] == "result" and (m := ARRIVED.match(e["text"]))]
    gave_up = sum(e["kind"] == "alert" and e["text"].startswith("Gave up on (") for e in notes)
    if offs or gave_up:
        near = sum(o <= NEAR_CM for o in offs)
        rows.append(("go-to", f"{len(offs)} arrived, {gave_up} gave up; stop error median "
                              f"{statistics.median(offs) if offs else 0:.1f} cm, max {max(offs, default=0):.1f} cm, "
                              f"{_pct(near, len(offs))} within {NEAR_CM} cm"))
    stops = [e["text"].rstrip(".") for e in notes if e["kind"] == "alert" and e["text"].startswith(STOPS)]
    rows.append(("safety", "; ".join([f"{len(stops)} stops", *stops])))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stem", nargs="?", help="run stem like 20261031-205313, or a path to its tick log")
    p.add_argument("--md", action="store_true", help="a markdown table for the README")
    args = p.parse_args()
    ticks, notes = paths(args.stem)
    rows = summary(read(ticks), read(notes))
    if args.md:
        print(f"| run | {ticks.stem} |\n|---|---|")
        print("\n".join(f"| {k} | {v.replace('|', '/')} |" for k, v in rows))
    else:
        print(f"run {ticks.stem}")
        print("\n".join(f"{k:<10}{v}" for k, v in rows))


if __name__ == "__main__":
    main()
