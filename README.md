# Visual Position Hold

Computer vision system for holding a drone over a fixed ground point using nadir camera, without GPS.

## How it works

The system tracks a ground marker using Lucas-Kanade optical flow, filters the noisy position estimate with an alpha-beta filter, converts pixel error to meters via camera FOV and altitude, and sends velocity commands to ArduPilot via MAVLink.

## Stack

- Python 3.10+
- OpenCV — optical flow tracking
- pymavlink — telemetry and velocity commands
- NumPy

## Hardware

- ArduPilot flight controller
- Orange Pi (onboard computer)
- Nadir (downward-facing) camera

## Project structure

| File | Description |
|------|-------------|
| `main.py` | Main loop, webcam test mode |
| `optical_flow_tracker.py` | Lucas-Kanade tracker with template matching |
| `alpha_beta_filter.py` | Alpha-beta filter for position smoothing |
| `pid.py` | PID controller |
| `visualizer.py` | HUD and trail visualization |

## Quick start

```bash
pip install opencv-python numpy
python main.py
```

Press `c` to capture the target point, `r` to reset, `q` to quit.

## Parameters

Edit the constants at the top of `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMULATED_ALTITUDE_M` | 5.0 | Simulated altitude for webcam test [m] |
| `FOV_H_DEG` | 90.0 | Camera horizontal field of view [deg] |
| `F_CUT_HZ` | 1.5 | Alpha-beta filter cutoff frequency [Hz] |
| `MAX_SPEED_MS` | 1.0 | Maximum velocity command [m/s] |
