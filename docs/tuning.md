# Tuning constants

Where each number comes from. Line numbers are as of commit 04971b6.

- **real car**: measured on the Elegoo car (Nov 1 testing unless noted).
- **by hand**: measured by hand on the car before the learner existed.
- **fitted**: comes out of a fit to data.
- **sim**: calibrated on simulated or twin runs (`scripts/eval.py`).
- **geometry**: follows from a physical dimension.
- **chosen**: set by judgement; no measurement recorded.

## Safety and the rig loop

| Constant | Value | Where | Source | Notes |
|---|---|---|---|---|
| `HZ` | 20 | `airo/rig.py:19` | chosen | Control and log rate; `DT = 1/HZ`. |
| `INSET` | 0.15 m | `airo/rig.py:21` | chosen | Floor margin that goals stay inside. TASKS.md's probe library still says 0.25 m. |
| `LOST_S` | 0.3 s | `airo/rig.py:22` | chosen | Pose age that counts as lost (D2 spec: >300 ms). |
| `HOLD_S` | 1.5 s | `airo/rig.py:23` | real car | Blind time held standing still before aborting, so brief marker blindness (glare, a hand) holds instead of aborting. |
| `Rig.overrun` | 0.1 m | `airo/rig.py:145` | real car | The real car overshoots the inset; the safety stop waits this much further so it can drive back (eb92fde). |
| `Rig.pivot_min` | 70 PWM | `airo/rig.py:148` | real car | Four fixed wheels stall when pivoting below about 70 PWM; a stalled pivot read as a body change on every patrol (a5aada3). |
| `Rig.drive_min` | 45 PWM | `airo/rig.py:149` | real car | Breakaway from rest is about 45 PWM. The fitted dead zone of about 0 comes from runs where the car is already moving (1ba2c54). |
| `WATCHDOG` | 0.5 s | `airo/world.py:19` | firmware | The twin copies the firmware watchdog that stops the car 500 ms after the last command. |
| `PWM_MAX` | 230 | `airo/protocol.py:10` | firmware | Largest PWM the protocol sends. |

## Driving

| Constant | Value | Where | Source | Notes |
|---|---|---|---|---|
| `V_MAX`, `V_MIN` | 0.25, 0.08 m/s | `airo/control.py:11-12` | chosen | Pursuit speed and final-approach floor. |
| `OM_MAX` | 2.0 rad/s | `airo/control.py:13` | chosen | Turn-rate cap. |
| `RECENTRE_PWM` | 160 | `airo/control.py:14` | real car | Caps power while recentring when the body model is wrong, so a rough refit cannot swing a heavy car (09b2e38). |
| `PIVOT`, `RESUME` | 60°, 20° | `airo/control.py:15-16` | chosen | Pivot in place above 60° heading error, drive again below 20°. |
| `SLOW_R` | 0.3 m | `airo/control.py:17` | chosen | Slow-down radius near the goal. |
| `LOOKAHEAD` | 0.15 m | `airo/control.py:18` | chosen | Pure-pursuit lookahead. |
| `PIVOT_PWM` | 70 | `airo/fleet.py:28` | chosen | Smallest turn the fleet guard lets a blocked car pivot with (Oct 29, before the real-car `pivot_min`). |
| `CARRY_PWM` | 100 | `airo/mission.py:49` | chosen | Turns stay within 100 PWM while carrying a tote (ab54d85). |
| `PATROL_IN` | 0.05 m | `airo/mission.py:63` | chosen | Patrol corners sit this far inside the inset. |

## Body model and surprise

| Constant | Value | Where | Source | Notes |
|---|---|---|---|---|
| `KEY` | g 0.00316, dz 25, eta 0.3, tau 0.2, lag 0.1, kappa −0.05 | `airo/body.py:43` | by hand | The answer key: shown next to a fit, never fed to it. |
| `PRIOR` | g 0.0025, dz 10, eta 1.0, tau 0.1, lag 0.1, kappa 0 | `airo/body.py:42` | chosen | Broad starting point; `PRIOR_SD` (line 47) sets its spread. |
| `TRACK` | 0.145 m | `airo/body.py:19` | geometry | Track width; fixed, not fitted. |
| `THRESHOLD` | 3.0 | `airo/body.py:24` | chosen | Surprise trigger: median of three windows above it. Not yet tuned on a clean real-car log (TASKS D5). |
| `INFLATE` | 14 | `airo/body.py:50` | sim | Noise inflation for ~95% predictive coverage over 30 seeds; tests/test_body.py checks it (18d968f). |
| `WINDOW` | 0.5 s | `airo/mission.py:25` | chosen | Surprise window (TASKS spec). |
| `QUIET_S` | 2.0 s | `airo/mission.py:26` | chosen | Surprise suppressed this long after a relearn (TASKS spec). |
| `DRIFT_K`, `DRIFT_H` | 1.8, 3.0 | `airo/mission.py:30` | sim | CUSUM for slow changes. Clean windows stayed under 1.6 in 40 eval runs on both rigs; a tow on a loaded car scores ~2.4 (d4240da). |
| `RELEARN_PROBES` | 5 | `airo/mission.py:43` | chosen | Probe cap per relearn (TASKS planned ≤3). |
| `FAULT_Z` | 5.0 | `airo/mission.py:44` | chosen | A relearn no body explains stops with a hardware fault (87a3270). |
| `CONFIRM_Z` | 2.0 | `airo/scientist.py:24` | chosen | A relearn counts once a predicted probe lands within 2σ. |
| `CONFIRM_TOL` | 3 cm, 3 cm, 6° | `airo/scientist.py:26` | chosen | The confirming prediction must also be tight (83c3fe1). |
| `BUMPED` | 4.0 | `airo/scientist.py:39` | chosen | A probe whose residual is 4× the median (and over 2 cm) ran into something and is left out. |
| `CHANGE_LL` | 1.0 | `airo/scientist.py:41` | chosen | Log-likelihood margin (~2.7×) for a remembered body to replace the current one. |
| `EYE_MOVE`, `EYE_ROOM`, `EYE_RMS` | 0.015, 0.15, 0.01 m | `airo/scientist.py:46` | chosen | Marker-offset check before learning (a7f8993). |
| `SLOW`, `FAST` | 60, 100 PWM | `airo/scientist.py:51` | chosen | A learn needs a straight probe at or below `SLOW` and one at or above `FAST` (e7868a4). |
| `WONDER_S` | 12 s | `airo/scientist.py:97` | chosen | Wonder's budget per relearn; was 20 s in 8463044. |

## Delivery and the real floor

| Constant | Value | Where | Source | Notes |
|---|---|---|---|---|
| `RESCAN_S` | 2.0 s | `airo/mission.py:31` | chosen | Re-segment while pushing (TASKS spec). |
| `DRIFT_M` | 0.08 m | `airo/mission.py:32` | chosen | Replan when the parcel drifts this far (TASKS spec). |
| `CONTACT` | 0.24 m | `airo/mission.py:39` | geometry | Bumper tips at 0.18 m plus the 0.1 m parcel: a closer pivot can sweep it. |
| `FLOOR_DT` | 0.2 s | `airo/mission.py:66` | chosen | Floor check period on the real car. |
| `WAY_REACH` | 0.35 m | `airo/mission.py:67` | chosen | How far along the path to look for something dropped. |
| `WAY_R` | 0.12 m | `airo/mission.py:67` | chosen | Clearance below which something is in the way. |
| `AROUND_CLEAR` | 0.15 m | `airo/mission.py:67` | chosen | Detour clearance, wider than `WAY_R` so the detour does not trip the check again. |
| `WAY_WAIT_S` | 1.5 s | `airo/mission.py:68` | chosen | Wait for the thing to move on before going around. |
| `RES` | 0.005 m | `airo/floorwatch.py:11` | chosen | Floor-image cell size. |
| `CAR_R` | 0.2 m | `airo/floorwatch.py:12` | chosen | Radius masked out around the car's pose. |
| `DIFF` | 28 | `airo/floorwatch.py:13` | chosen | Grey-level change (0–255, after gain matching) that counts as changed. |
| `MIN_AREA` | 0.0025 m² | `airo/floorwatch.py:14` | chosen | Smallest object kept (5 × 5 cm). |
| `GLOBAL` | 0.25 | `airo/floorwatch.py:16` | chosen | Above this fraction of the floor changed at once, the frame is skipped as a light or exposure change. |
| `DETECT_W` | 1280 px | `airo/camera.py:20` | chosen | Markers are found on a copy at most this wide (9bfecae). |
| `CAR_MARKER_M`, `CORNER_MARKER_M` | 0.095, 0.12 m | `airo/camera.py:16-17` | geometry | Marker edge lengths; `scripts/markers.py` prints at these sizes. |

## MuJoCo twin

| Constant | Value | Where | Source | Notes |
|---|---|---|---|---|
| `DELAY` | 0.17 s | `airo/mjsim.py:20` | chosen | Command delay; inside the D3 tau gate (150–250 ms). |
| `IMBALANCE` | 0.009 | `airo/mjsim.py:22` | chosen | Left/right motor gain mismatch that gives the twin a drift. |
| `WEIGHT_KG` | 0.8 kg | `airo/mjsim.py:21` | chosen | The twin's weight pack. |
| `KINETIC`, `BREAKAWAY` | 0.05, 0.15 rad/s | `airo/world.py:16-17` | chosen | Wheel stiction drops to 5% once a wheel turns faster than 0.15 rad/s. |
| `LOAD_DAMPING` | 2.6e-4 | `airo/world.py:18` | chosen | Wheel damping per newton of contact force, so a load slows the car. |

## Run data

`data/` is gitignored, so run logs stay on the machine that recorded them. Each rig run writes
`data/runs/<ts>.jsonl` (one row per tick) and the mission writes `<ts>.notes.jsonl` next to it (the notebook).
`uv run scripts/report.py [STEM] [--md]` turns one run into learning, go-to, surprise, safety and BLE numbers.

The README's tables come from `scripts/eval.py` in simulation, which writes `data/eval.json`, not from run files.
Neither the README nor `report.py` names the run files behind any number, so none are listed here.
