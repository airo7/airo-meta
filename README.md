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
cp .env.example .env   # add MODEL_API_KEY
```

## Day-1 checks

## Architecture

## Measured numbers

## License

MIT
