# AIRO

**Adaptive Intelligence for Robots.** A $50 robot car that learns its own body.

AIRO watches a cheap, encoder-less robot car from an overhead phone camera and works out how the car's body
behaves by running physical experiments on it. It fits a small model of its own motion and shows an ensemble of
predictions as a "ghost cloud". It delivers parcels on voice command. When the body changes (extra weight, a
towed box), it notices that reality has left the cloud, runs targeted probes, relearns and carries on.
Language models plan the experiments and explain the results, but they are never inside the control loop.

## Hackathon disclosure

Built for the Meta Global AI Developer Hackathon, Oct 26 – Nov 4 2026.

- **Built during the event:** everything in this repository.
- **Pre-existing:** the car's Arduino firmware from
  [airo7/sim2real-robot-car](https://github.com/airo7/sim2real-robot-car) (`hardware/arduino/main/main.ino`),
  used unchanged. `airo/protocol.py` reimplements its CMD/SENS byte layout so this repo can talk to it.
  Nothing else is copied.
- **Meta models:** `muse-wonder-1.0`, `muse-spark-1.3`, `sam-3.1`, `muse-voice-transcribe-1.0`.

## Hardware

Elegoo Smart Robot Car V4 (Arduino UNO, 4 TT motors, skid steer, no encoders), HM-10 BLE module ("HC-05"), and
an iPhone as an overhead camera (~1.8 m) via Continuity Camera.

## Setup

```sh
uv sync
cp .env.example .env              # add MODEL_API_KEY
uv run scripts/markers.py         # A4 pages in data/markers, print at 100%
```

Floor frame: tape corner markers 1–4 counter-clockwise (1 → 2 is +x, 1 → 4 is +y), all with their top edge
toward +y, and measure the centre-to-centre distances W and H. Tape marker 0 flat on the car, top edge forward.
On the phone, turn off Center Stage, Portrait and Studio Light; auto-cropping breaks the floor homography.

## Day-1 checks

| Check | Command | Pass |
|---|---|---|
| Pose | `uv run scripts/pose_check.py --floor W H --marker-h 0.1 --show` | ≥15 Hz, residual and still-car jitter ≤10 mm |
| Dry run | `uv run scripts/goto.py --floor W H --marker-h 0.1 --dry` | pose arrow on the car; clicks print floor coordinates that match a tape measure |
| BLE | `uv run scripts/drive_check.py` | drives straight then pivots, ack ≈66 ms, SENS ≈25 Hz |
| Meta API | `uv run --env-file .env scripts/bench_api.py --concept box` | latency printed, overlay in `data/` looks right |
| Offline | `uv run pytest` | geometry, protocol and SAM parsing tests pass |

## Architecture

- `airo/protocol.py`: CMD/SENS bytes
- `airo/car.py`: asyncio BLE link
- `airo/camera.py`: ArUco floor homography and car pose
- `airo/meta_api.py`: SAM 3.1 and Muse Spark
- `airo/mjsim.py`: MuJoCo twin of the rig: contact physics, rendered overhead camera, 3D ghost cloud (`scripts/mjshow.py`)

## Uncertainty

The ghost cloud is a Laplace posterior around the fitted body with a broad physical prior; a parameter keeps its
prior spread until some probe has excited it (eta and drift need a turn, the dead zone a command near it). The noise
variance is inflated by 14 × the residuals' autocorrelation time, calibrated on held-out probes in simulation: over
30 seeds the true outcome falls within 2σ of the prediction 87–100% of the time (93/87/100% for x/y/heading with
all probes, 100/100/97% with four), and the true parameters 87–100%, except the dead zone at 73% when only four probes
are available.

## Measured numbers

**Simulation** (`uv run scripts/eval.py --n 10 --rig sim|mujoco`, scripted scientist, no API; hardware numbers to
follow). Each seed learns once, then delivers with the body changed 0.3 m into the transit.

| | SimRig (model-exact) | MuJoCo twin (physics) |
|---|---|---|
| Learn: 10 probes, cloud radius | 0.3 cm, parameters within 6% of truth | 1.1 cm |
| Held-out probes within 2σ | 100% | 100% |
| False surprise triggers (50 runs) | 0 | 0 |
| Weight: detected / confirmed / delivered | 10/10 at 1.0 s, 9/10, 10/10 (gains −35%) | 0/10, –, 10/10 (0.4 kg is only ~−8% gain) |
| Tow: detected / confirmed / delivered | 0/10, –, 10/10 (gains −15%, lag ×2) | 10/10 at 1.6 s, 1/10, 10/10 |
| Slippery floor | 0/10 detected, 10/10 delivered | 0/10 detected, 10/10 delivered |
| Flipped motor | 10/10 detected at 1.0 s, 0/10 confirmed, safety stop 10/10 | same |
| Parcel within 10 cm of the station | 5.8 cm median (clean) | 8.2 cm median (clean) |

A towed box is a drag force the body model has no term for, and a flipped motor needs a negative gain; relearning
reports both as partial fits instead of claiming a confirmed body.

## License

MIT
