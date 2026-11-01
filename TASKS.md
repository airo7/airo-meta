# AIRO tasks, Oct 29 – Nov 4

Deadline Nov 4 21:00 ET = Nov 5 03:00 CET. **Real cutoff: Nov 4 18:00 CET.**

Conventions:
- Every hardware run writes `data/runs/<ts>.jsonl` (t, pwm_l, pwm_r, x, y, theta). Abebe runs and shares logs; Claude refits and debugs offline.
- Every module that needs the car also runs against `airo/sim.py` (`--sim`). Claude develops in sim; Abebe validates on the car.

## D2 · Thu Oct 29 · reach a clicked point
Gate: 8/10 tries within 5 cm.
- [x] Claude `airo/rig.py`: camera thread + Car + 20 Hz loop, jsonl log, `--sim`. Abort when the pose leaves the floor inset or is lost for >300 ms.
- [x] Claude `airo/sim.py`: simulated car and camera from a `Body` (delay, noise, drift); same `drive()` as `Car`.
- [x] Claude `airo/body.py` v0: `Body` params, `step`, `predict`, `inverse(v, ω) → pwm`; prior from the hand-measured numbers.
- [x] Claude `airo/control.py`: delay-compensated pure pursuit, `goto`, `follow`; slow final approach.
- [x] Claude `scripts/goto.py` (click a point, print final error), `tests/test_control.py` in sim.
- [ ] Abebe: finish the floor (markers, measure W H), run the day-1 checks, run goto 10×, share logs. Open: the README has no hardware go-to numbers, so the 8/10 within 5 cm gate is unreported.
- [x] Abebe: office hours 16–17 CET, ask about the muse-voice realtime WS message format. Note: the realtime format is implemented in `airo/voice.py`.

## D3 · Fri Oct 30 · the car learns its body
Exit, on hardware:
- gain within 20% of 0.00316
- dead zone within ±8 PWM
- τ between 150 and 250 ms
- drift sign correct
- cloud radius shrinks across probes
Tasks:
- [x] Claude `airo/probes.py`: probe library (below), free-space check, runner returning labelled segments, `goto(centre)` between probes.
- [x] Claude `body.py` full: `fit`, `bootstrap(k=30)`, `cloud`, `surprise`; `tests/test_body.py`: sim with known params gives a fit within 10%. Note: a Laplace posterior, `ensemble`, replaced the bootstrap; the sim fit is within 7%.
- [ ] Claude `scripts/learn.py`: scripted probe schedule, saves `data/body.json`, prints the fit next to the answer key. `scripts/fit.py`: offline refit from logs. Open: learn.py is done; `scripts/fit.py` (offline refit from logs) does not exist.
- [ ] Abebe: run learn.py 3×, share logs. Open: the README has no hardware learn numbers; the D3 exit gates are unreported.
- [ ] Abebe: build add-ons: 400 g weight pack (velcro), tow box + string, V-shaped cardboard bumper, 2 coloured A4 station sheets, 3 box obstacles. Open: not verifiable from the repo.
- [ ] Abebe: X post 1: teaser gif of the cloud shrinking. Open: not verifiable from the repo.

## D4 · Sat Oct 31 · delivery A→B
Gate: 4/5 deliveries. Fallback: if push is below 3/5 by 15:00, switch to carry mode.
- [x] Claude `airo/planner.py`: SAM masks → floor grid (2 cm cells) → inflate → 8-connected A* → smoothing. `Scene` = obstacles, stations, parcel. `tests/test_planner.py`.
- [ ] Claude `airo/mission.py`: `deliver(station)`. Approach behind the parcel, push along the path, re-segment every 2 s, replan if the parcel drifts >8 cm; `--carry` fallback. Open: `deliver` with rescans and replanning is done; the real car has no `--carry` fallback (carry runs only on the factory twin).
- [x] Claude `airo/server.py` + `web/index.html` v1:
  - FastAPI `/ws` state stream at 10 Hz, `/video` MJPEG, `/cmd`
  - canvas showing the live view, path, masks and body card
- [ ] Claude `scripts/deliver.py` gate CLI. Open: no `scripts/deliver.py`; `scripts/eval.py` measures deliveries in sim only.
- [ ] Abebe: place stations, obstacles and parcel; run deliver 5×; share logs. Open: the README has no hardware delivery numbers.

## D5 · Sun Nov 1 · scientist agent + relearning
Exit: two full loops (learn → deliver → surprise → relearn → resume), one with the weight and one with the tow, screen-recorded.
- [x] Claude `airo/agent.py`: Muse Wonder tool loop, prediction-first notebook, Spark visual confirm, scripted fallback. `meta_api.wonder()`. Note: built as `airo/scientist.py`.
- [x] Claude `mission.py`: surprise monitor → pause → relearn (≤3 probes, pivots preferred) → Body #N, "relearned in X s" → replan. Note: the cap is 5 probes, `RELEARN_PROBES`, starting with a forward step.
- [x] Claude dashboard: lab notebook panel, surprise sparkline, Body #N counter, relearn timer.
- [ ] Abebe: run the full loop with the weight drop, then the tow; tune the surprise threshold from a clean log. Open: `THRESHOLD` is still 3.0, set from sim eval runs; the README has no real-car loop.
- [ ] Abebe: X post 2: 10 s clip of the surprise moment. Open: not verifiable from the repo.

## D6 · Mon Nov 2 · voice, hardening, rehearsal
No new features after tonight.
- [x] Claude `airo/voice.py`: realtime transcription → intents ("deliver to blue", "learn your body", "stop"); the text bar stays as a fallback.
- [ ] Claude hardening: Open: SAM has no timeout on the real car (8 s only in `demo.live_sam` on the twins); the Wonder budget (12 s), pose-loss hold, BLE reconnect and `demo.py` are done.
  - API timeouts (SAM 8 s, Wonder 20 s)
  - pause on pose loss
  - BLE reconnect
  - `scripts/demo.py` single entry point
- [x] Claude optional: `meta_api.image()` self-portrait body card.
- [ ] Abebe: 3 recorded dress rehearsals, B-roll of the rig and add-ons. Open: not verifiable from the repo.

## D7 · Tue Nov 3 · film and edit
- [ ] Abebe: film the shot list, record the voiceover, rough cut ≤3:00 by evening. Open: due Nov 3.
- [ ] Claude: README (measured numbers, architecture, new-work disclosure), tests green, cleanup, README gif. Open: no README gif (stills only) and no hardware numbers; architecture, sim numbers and disclosure are in, tests pass (244 passed, 19 skipped on Nov 1).
- [ ] Abebe: X post 3: self-portrait card + "a $50 robot learned its own body". Open: due Nov 3.

## D8 · Wed Nov 4 · submit by 18:00 CET
- [ ] Abebe: final cut, upload, submission form, confirmation screenshot. Open: due Nov 4.
- [ ] Abebe: X post 4: final video + vote ask. Open: due Nov 4.
- [ ] Claude: final README pass, tag `v0.1-hackathon`, fresh clone `uv sync && uv run pytest`. Open: no `v0.1-hackathon` tag yet.

**Cut order if behind:** voice, heavy parcel, Muse Image. Muse Glimmer (open-weights) is a stretch goal only if Nov 2 ends all green; it's a ~2 h base_url swap for the notebook writer.

## Body model
- u = sign(p)·max(0, |p| − dz), v_side = g_side·u
- v = (vL + vR)/2
- ω = (vR − vL)/b · (η if pivoting) + κ·v
- commands apply after τ

Fitted parameters: gL, gR, dz, η, τ, κ. The track width b is fixed. The answer key is displayed, never fed to the fit.

## Probe library
Floor inset: 0.15 m. Each probe checks its free space first. The rig aborts on leaving the inset; the firmware watchdog stops the car within 500 ms.

| Probe | Command | Measures | Space |
|---|---|---|---|
| step | p ∈ {40, 60, 80, 110, 150} for 1 s, plus reverse −60/−110 | per-side gain, dead zone, drift κ, τ | 0.8 m |
| ramp | 0 → 120 over 2 s | dead zone precisely | 0.6 m |
| pivot | ±p, p ∈ {60, 100, 150}, 1 s | pivot efficiency η, pivot dead zone | 0.3 m radius |
| arc | (120, 60) and (60, 120), 1 s | mixing law, side gains | 0.6 m |
| pulse | 100 for 0.3 s, then 0 | τ, coast distance | 0.4 m |

Initial learn: 8–10 probes. Relearn: ≤3 probes chosen by the agent.

## Surprise and relearn trigger
- Per 0.5 s window, replay the actual commands from the window's start pose through each ensemble member.
- s = max(‖Δ − mean Δ̂‖ / √(σ²_cloud + 0.01²), |Δθ − mean Δ̂θ| / √(σ²_θ + 0.03²)).
- Trigger when the median s over the last 3 windows is >3, with commands above the dead zone and no pose dropouts.
- Suppressed during the parcel approach and for 2 s after a relearn.

## Agent loop (Muse Wonder)
- Tools: `get_body`, `list_probes`, `predict(probe)`, `run_probe(probe)`, `refit`, `conclude(text)`.
- Each step:
  1. The agent picks a probe and states a hypothesis.
  2. The engine records a numeric prediction.
  3. The probe runs.
  4. A notebook entry is written: {hypothesis, predicted, measured, z, conclusion}.
- The LLM writes only the text fields; every number comes from tools. Probes outside the safety bounds are rejected.
- The relearn prompt includes the surprise window plus Spark's one-line description of the frame.
- On API error or timeout, a scripted policy runs the same schedule and the notebook marks those entries as fallback.

## Shot list
1. Overhead wide: weight dropped mid-delivery → pause → pivots → resume (the hook)
2. The same moment from a phone at floor level
3. Screen capture of the learn phase: the cloud shrinks, with the answer-key comparison
4. Voice command → SAM masks → path
5. Push delivery to a station
6. Tow string tied → surprise spike → notebook entry → Body #3 counter
7. Rig B-roll: $50 car, phone stand, markers
8. Meta models slide

## Submission checklist
- [ ] Public repo, MIT license, no secrets, fresh-clone tests pass. Open: github.com/airo7/airo is still private; the MIT `LICENSE` is in.
- [ ] README disclosure: pre-existing firmware only; commit range of the new work. Open: the firmware disclosure is in the README; the commit range is missing.
- [ ] Video ≤3:00, public. Open: due Nov 4.
- [ ] Form: Startup track, PiSpace, models used, bonuses (Best Agentic, Community Vote, Most Viral). Open: due Nov 4.
- [ ] Confirmation screenshot. Open: due Nov 4.

## Risks
| Risk | Mitigation |
|---|---|
| Push delivery unreliable | V-bumper, replan every 2 s, carry fallback |
| Pose lost when a hand drops the weight | pause on loss, resume after 1 s of stable pose |
| Wonder slow or down live | 20 s timeout → scripted policy |
| False surprise from parcel contact | threshold calibrated on clean logs, suppressed during approach |
| Floor too small | probes self-check space; relearn prefers pivots |
| BLE drop | one reconnect; firmware watchdog |
| Voice API format unknown | ask at office hours; text bar fallback; voice is cut first |

## v3 · Oct 29 – Nov 1 · memory, factory fleet, new UI
Order: recall → factory + fleet → UI → place memory → GRU first guess (last).

Workstreams (worktrees under ~/airo/airo-wt, merged by the controller):
- memory (airo-8e): `airo/memory.py` body library, recall before relearn, eval of recall vs relearn.
- fleet: factory site in MuJoCo, N cars in one physics world, per-car body changes, the spill as a place change.
- ui: the dashboard rebuilt against the state below, driven by `scripts/dash_demo.py`.
- controller: Meta model feed, fleet coordinator (shared memory, zones, command routing), integration.

### Dashboard state v3 (`/ws`, 10 Hz)
```
{
  "t": s, "mode": "sim" | "real",
  "site": {"name": "Factory", "floor": {"w", "h"}, "stations": {"packing": [x, y], ...},
           "zones": [{"kind": "slippery", "polygon": [[x, y], ...], "label": "Oil spill", "found_by": "airo-2", "t": s}]},
  "selected": "airo-1",
  "robots": [{
    "id": "airo-1", "name": "AIRO 1", "color": "#ffb340", "real": bool,
    "now": "one plain sentence: what it is doing and why",
    ... every key of today's Mission.state(): status, command, pose, cloud, path, polygons, body, surprise,
        notebook, episode, relearn ...,
    "memory": [{"n", "name", "cause", "params": [same shape as body.params], "portrait": url | null,
                "learned_t": s, "recalls": int, "active": bool, "from": robot id | null, "verified": bool}]
  }],
  "models": [{"id": int, "t": s, "robot": id | null, "model": "sam-3.1", "kind": "segment" | "describe" | "wonder"
              | "transcribe" | "image", "input": str, "output": str, "ms": int, "ok": bool,
              "image": "/models/<id>.jpg" | null, "tools": [{"name", "args", "result"}] | null}],   # newest last, ≤40
  "sim": {"camera", "preset", "paused", "speed", "presets", "parts", "changes"}
}
```
Notebook gains kind "recall": {"text", "n", "name", "predicted", "measured", "z", "recall_s"}.
Commands: "<robot> <command>" targets one robot ("airo 2 deliver to packing"); no prefix targets the selected one;
"select airo 2" switches; body changes apply to the selected robot, "spill" is a place change.

## D6+ · Mon Nov 2 · real car: a blocked way and tight gaps (code by Muse Spark via Codex CLI)
Demo beats on the real floor, all from the floor watch (no SAM needed):
1. **A box blocks the only way.** Stop → wait → no way around → Spark names it → small enough → push it aside → carry on.
2. **Too heavy.** Same with a heavy box: the push stalls → back off → "it won't move, could someone move it?" → a hand removes it → carry on.
3. **Does it fit?** Two boxes with a gap: the car measures the gap against its own width and learned steering spread
   and says why it goes through or the long way.

- [x] Muse `feat/unblock` (worktree ~/airo/airo-wt/unblock): push a blocking box aside, or ask for help.
- [x] Muse `feat/fit` (worktree ~/airo/airo-wt/fit): detour clearance from the body's uncertainty, gap narration.
- [x] Controller: review, merge both into `_around`, tests green.
- [ ] Abebe: real car, beats 1–3, share logs.
