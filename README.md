# gesture_detection

Raspberry Pi scaffold for recognising hand gestures made above an ultrasonic sensor.

## What is included

- capture/session scaffolding for HC-SR04 distance readings
- console and SSD1306 display adapters
- labelled data collection workflow that saves JSON samples by gesture and session
- dataset indexing helpers for later training/validation work
- placeholder classifier and training pipeline entry points for your own ML implementation
- a sample systemd service for boot-on-power deployment

## What is intentionally missing

The machine learning implementation is left open on purpose.

You should fill in:

- `src/gesture_detection/training/pipeline.py`
- `src/gesture_detection/ml/user_model.py`

Those files contain the places to add feature preparation, PyTorch training, validation, weight export, and a NumPy-only inference path.

## Project layout

- `src/gesture_detection/sensors/` — sensor drivers and hardware abstraction
- `src/gesture_detection/display/` — console/OLED display adapters
- `src/gesture_detection/capture.py` — sampling, rolling windows, and activity gating
- `src/gesture_detection/collector.py` — labelled capture workflow
- `src/gesture_detection/training/` — dataset indexing plus training/export stubs
- `configs/default.json` — starter runtime configuration
- `deploy/gesture-detection.service` — boot service scaffold

## Quick start

Create a starter config if you want a fresh copy:

```bash
python -m gesture_detection.cli init-config
```

Inspect the local scaffold with the mock sensor and console display:

```bash
PYTHONPATH=src python -m gesture_detection.cli --config configs/default.json run
```

Collect labelled examples locally:

```bash
PYTHONPATH=src python -m gesture_detection.cli --config configs/default.json collect swipe_left --samples 3
```

Inspect what you have captured:

```bash
PYTHONPATH=src python -m gesture_detection.cli --config configs/default.json inspect-dataset
```

## Raspberry Pi notes

- switch `sensor.driver` to `hcsr04` once `gpiozero` is installed on the Pi
- switch `display.driver` to `ssd1306` once `luma.oled` is installed on the Pi
- update `deploy/gesture-detection.service` paths to match the final install location
- keep the current activity gate as a simple pre-filter so your eventual model is not asked to classify idle buffers constantly
