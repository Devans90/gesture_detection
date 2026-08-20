# gesture_detection

A battery-powered Raspberry Pi 1 B+ that recognises hand gestures made above
an HC-SR04 ultrasonic sensor, running a neural network with **no ML runtime**
installed on the device.

---

## How it works

The HC-SR04 samples distance at ~20 Hz.  A 1.5-second sliding window of 30
readings encodes the shape of a gesture.  A small 1-D CNN trained on a laptop
is exported as raw numpy arrays, and a hand-written numpy forward pass runs on
the Pi at inference time.  The result is shown on an SSD1306 OLED.

---

## Project structure

```
gesture_detection/
├── config.py               # All tunable parameters (pins, labels, hyper-params)
├── main.py                 # Runtime loop: sample → classify → display
│
├── hardware/
│   └── sensor.py           # HC-SR04 driver (BCM GPIO, forward-fill, clamp)
│
├── capture/
│   └── sampler.py          # Continuous sampler → fixed-length window callbacks
│
├── collect/
│   └── recorder.py         # CLI tool: record labelled training examples
│
├── train/
│   ├── dataset.py          # Data loader + pre-processing (TODO: augmentation)
│   ├── model.py            # GestureCNN skeleton ← **your implementation here**
│   └── trainer.py          # Training loop + MLflow tracking
│
├── infer/
│   └── numpy_model.py      # Numpy forward pass ← **your implementation here**
│
├── display/
│   └── oled.py             # SSD1306 OLED wrapper (luma.oled)
│
├── deploy/
│   ├── gesture.service     # systemd unit file
│   └── install_service.sh  # Install & enable on the Pi
│
├── tests/
│   ├── test_capture.py     # Sensor & sampler tests (no hardware needed)
│   └── test_infer.py       # Numpy helper tests
│
├── data/
│   ├── raw/                # Session CSVs written by collect/recorder.py
│   └── processed/          # (optional) pre-processed arrays
│
├── models/
│   ├── best_model.pt       # PyTorch checkpoint (created by trainer)
│   └── weights.npz         # Numpy weights (exported by trainer, used on Pi)
│
└── requirements.txt
```

---

## Workstreams & your TODOs

### 1 · Capture  ✅ (implemented)
`hardware/sensor.py` and `capture/sampler.py` — HC-SR04 driver with forward-fill
for dropped readings and a sliding-window sampler.

### 2 · Collect  ✅ (implemented)
```bash
python -m collect.recorder --gesture swipe_left --session 1 --reps 20
```
Produces `data/raw/swipe_left_session01.csv`.  Run for each gesture and each
person, using a different `--session` number per sitting.

### 3 · Train  🟡 (scaffold ready — **you implement the model**)

Open **`train/model.py`** and implement:
- `GestureCNN.__init__` — define your Conv1d / BatchNorm / Linear layers.
- `GestureCNN.forward` — run the input through those layers and return logits.

Then run:
```bash
python -m train.trainer
```
MLflow results appear in `mlruns/` — view with `mlflow ui`.

### 4 · Validate  🟡 (scaffold ready)
`train/dataset.py` uses **session holdout** (not random split) automatically.
Tune `VALIDATION_SESSION_HOLDOUT` in `config.py`.

Optional: open `_augment` in `train/dataset.py` to add noise / time-shift.

### 5 · Deploy  🟡 (scaffold ready — **you implement numpy forward pass**)

Open **`infer/numpy_model.py`** and implement `NumpyGestureModel._forward` to
mirror your PyTorch architecture using the provided numpy helpers:
`conv1d`, `batch_norm1d`, `relu`, `max_pool1d`, `global_avg_pool`, `linear`.

Copy the exported `models/weights.npz` to the Pi, then:
```bash
python main.py
```

### 6 · Package  ✅ (implemented)
```bash
sudo bash deploy/install_service.sh
```
Enables `gesture.service` to start on boot.

---

## Quick start (off-Pi development)

```bash
pip install -r requirements.txt
# Simulate sensor + display:
python main.py --simulate

# Run tests:
pytest tests/
```

---

## Gesture labels

Edit `GESTURE_LABELS` in `config.py`:
```python
GESTURE_LABELS = ["swipe_left", "swipe_right", "push_down", "hover", "idle"]
```

---

## Hardware wiring

| HC-SR04 | Raspberry Pi (BCM) |
|---------|--------------------|
| VCC     | 5V (pin 2)         |
| GND     | GND (pin 6)        |
| TRIG    | GPIO 23 (pin 16)   |
| ECHO    | GPIO 24 (pin 18) via voltage divider |

SSD1306 OLED connects via I²C (SDA = GPIO 2, SCL = GPIO 3).
Enable I²C with `sudo raspi-config`.

---

## Known risks

* HC-SR04 tops out at ~20 Hz — very fast gestures may be unrecoverable.
  Check your gesture timing before committing to a gesture set.
* Training on one person's data will overfit to that person.
  Collect data from at least two people.
