# Gesture Detection

> **Battery-powered Raspberry Pi 1 B+ that recognises hand gestures above an
> HC-SR04 ultrasonic sensor — runs a neural network with no ML runtime
> installed.**

A cheap distance sensor produces a one-dimensional time-series; the *shape*
of that series over a second or two encodes a gesture.  Someone waves their
hand → the OLED screen names what they did.

---

## Hardware

| Component | Notes |
|-----------|-------|
| Raspberry Pi 1 B+ | ARMv6, no pip-installable PyTorch |
| HC-SR04 ultrasonic sensor | ~20 Hz, 2–400 cm range |
| SSD1306 OLED 128×64 | I²C, 3.3 V |
| Powerbank | any 5 V USB bank |

### Wiring

```
HC-SR04  TRIG  → GPIO 23 (BCM)
HC-SR04  ECHO  → GPIO 24 (BCM)  ← 5 V logic; use a voltage divider
OLED     SDA   → GPIO 2  (pin 3)
OLED     SCL   → GPIO 3  (pin 5)
```

---

## Project Layout

```
gesture_detection/
├── capture/
│   ├── sensor.py       HC-SR04 driver (distance sampling, drop handling)
│   └── buffer.py       Rolling window + gesture segmenter
│                         ← YOU implement _motion_detected()
├── collect/
│   ├── recorder.py     Labelled example recorder (numpy + CSV)
│   └── cli.py          Interactive data collection CLI
├── model/
│   ├── dataset.py      PyTorch Dataset, normalisation, session-hold-out split
│   ├── network.py      CNN shell ← YOU implement __init__ and forward()
│   ├── train.py        Training loop + MLflow tracking scaffold
│   └── validate.py     Session-hold-out evaluation + confusion matrix
├── deploy/
│   ├── inference.py    Numpy-only forward pass ← YOU implement predict()
│   ├── export.py       Export PyTorch weights → .npz (run on laptop)
│   └── predictor.py    Gesture predictor with idle rejection stub
├── display/
│   ├── oled.py         SSD1306 OLED wrapper (luma.oled)
│   └── ui.py           Display state machine (IDLE / GESTURE / ERROR)
├── systemd/
│   └── gesture.service systemd unit for boot-on-power
├── tests/              pytest suite (no hardware required)
├── main.py             Boot entry point
├── requirements.txt
└── setup.py
```

---

## Your Implementation Tasks

The scaffolding is complete.  Three things are left for you to implement:

### 1 · `capture/buffer.py` — `GestureSegmenter._motion_detected`

```python
def _motion_detected(self, window: np.ndarray) -> bool:
    """Return True if the window contains a gesture."""
    raise NotImplementedError(...)
```

**Starting point:** try `return window.std() > <threshold>`.  Print
`window.std()` for idle and gesture windows to find a good threshold.  This
is the gate that controls when data gets recorded and when predictions fire,
so it's worth tuning carefully before you move on.

---

### 2 · `model/network.py` — `GestureNet.__init__` and `forward`

```python
class GestureNet(nn.Module):
    def __init__(self, ...):
        ...  # define layers here
    def forward(self, x):
        ...  # wire them together
```

**Suggested architecture:**
```
Conv1d(1, 16, k=5, pad=2) → ReLU
Conv1d(16, 32, k=5, pad=2) → ReLU → MaxPool1d(2)
Conv1d(32, 64, k=3, pad=1) → ReLU → MaxPool1d(2)
Flatten → Linear(640, 64) → ReLU → Dropout(0.3) → Linear(64, NUM_CLASSES)
```
The window is 40 samples long.  After two `MaxPool1d(2)` layers the length
is 10, giving `64 × 10 = 640` flat features.

---

### 3 · `deploy/inference.py` — `NumpyInference.predict`

```python
def predict(self, window: np.ndarray) -> np.ndarray:
    """Numpy-only forward pass; return logits."""
    raise NotImplementedError(...)
```

Translate each layer of `GestureNet` into numpy.  The file already provides
helpers: `conv1d`, `max_pool1d`, `relu`, `softmax`.  The weight keys match
PyTorch's `state_dict` names (e.g. `"conv1.weight"`, `"fc_out.bias"`).

---

## Workstream Guide

### Step 1 — Collect data

```bash
# Install dependencies (on laptop; Pi only needs numpy + luma.oled)
pip install -e ".[pi,train,dev]"

# Implement _motion_detected first, then:
python -m collect.cli --label swipe_left  --participant alice --n 30
python -m collect.cli --label swipe_right --participant alice --n 30
python -m collect.cli --label wave        --participant alice --n 30
python -m collect.cli --label push        --participant alice --n 30
python -m collect.cli --label idle        --participant alice --n 30
# Repeat with a second participant for generalisation
```

### Step 2 — Train

```bash
# Hold out alice's last session for validation
python -m model.train \
    --data-dir data/raw \
    --holdout-sessions session_<YYYYMMDD_HHMMSS> \
    --epochs 30 --lr 1e-3

# Check MLflow UI
mlflow ui
```

### Step 3 — Validate

```bash
python -m model.validate \
    --checkpoint runs/<run_id>/best.pt \
    --holdout-sessions session_<YYYYMMDD_HHMMSS>
```

### Step 4 — Export weights and implement numpy inference

```bash
python -m deploy.export \
    --checkpoint runs/<run_id>/best.pt \
    --output deploy/weights.npz

# Copy to Pi
scp deploy/weights.npz pi@raspberrypi.local:gesture_detection/deploy/
```

Then implement `NumpyInference.predict` in `deploy/inference.py`.

### Step 5 — Deploy

```bash
# On Pi
sudo cp systemd/gesture.service /etc/systemd/system/
sudo systemctl enable gesture
sudo systemctl start gesture
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests run without hardware (GPIO and luma.oled gracefully stub out).

---

## Success Criteria

- [ ] Recognises gestures from someone who didn't contribute training data
- [ ] Stays quiet when nothing is happening (idle rejection working)
- [ ] Runs on battery, on boot, with no laptop attached

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| HC-SR04 is noisy / slow (~20 Hz) | Check `window.std()` histograms before committing to gesture set; avoid fast flicks |
| Single-person training data overfits | Recruit at least one other pair of hands before final evaluation |
| ARMv6 no PyTorch/ONNX | Numpy forward pass in `deploy/inference.py` |
