# Gesture Detection

Battery-powered Raspberry Pi 1 B+ that recognises hand gestures above an
HC-SR04 ultrasonic sensor.  A small CNN classifies the distance time-series;
inference runs via a pure-numpy forward pass (no ML runtime on the Pi).

---

## Hardware

| Part | Notes |
|------|-------|
| Raspberry Pi 1 B+ | ARMv6, no PyTorch/ONNX support |
| HC-SR04 | ~20 Hz distance sensor |
| SSD1306 128×64 OLED | I²C display |
| Powerbank | 5 V USB |

---

## Repository layout

```
capture.py          HC-SR04 sampling + noise filtering
collect.py          Interactive labelled-data collection CLI
dataset.py          Data loading, preprocessing, augmentation  ← implement
model.py            PyTorch CNN skeleton                        ← implement
train.py            MLflow-tracked training loop
validate.py         Session-held-out evaluation + rejection analysis
display.py          SSD1306 OLED helper
main.py             Boot-on-power inference loop (state machine)
deploy/
  export.py         Export trained weights → numpy .npz
  forward.py        Numpy-only forward pass for the Pi          ← implement
systemd/
  gesture.service   systemd unit for boot-on-power startup
data/raw/           Collected CSV files (git-ignored)
checkpoints/        Training checkpoints (git-ignored)
requirements.txt    Training machine deps (PyTorch, MLflow, numpy)
requirements-pi.txt Pi runtime deps (RPi.GPIO, numpy, luma.oled)
```

---

## Workstreams

### 1  Capture

`capture.py` samples the HC-SR04 at ~20 Hz, drops out-of-range readings, and
applies a small median filter.  Run standalone to verify wiring:

```bash
python capture.py
```

### 2  Collect

Record labelled examples (vary hand, speed, position):

```bash
python collect.py --label swipe_left  --sessions 20
python collect.py --label swipe_right --sessions 20
python collect.py --label push_down   --sessions 20
python collect.py --label pull_up     --sessions 20
python collect.py --label idle        --sessions 20
```

CSV files land in `data/raw/`.

### 3  Implement ML  ← **your job**

Three files have `raise NotImplementedError` stubs for you to fill in:

| File | What to implement |
|------|------------------|
| `dataset.py` → `preprocess()` | Normalise the distance time-series |
| `dataset.py` → `augment()` | Data augmentation strategy |
| `model.py` → `GestureNet` | CNN layer stack and forward pass |
| `train.py` → `choose_optimizer()` | Optimiser choice |
| `train.py` → `choose_loss()` | Loss function |
| `train.py` → `choose_scheduler()` | LR schedule (or `return None`) |
| `deploy/forward.py` → `GestureNetWeights` | Declare weight fields |
| `deploy/forward.py` → `load_model()` | Load .npz into weight struct |
| `deploy/forward.py` → `_forward()` | Numpy forward pass mirroring model.py |

### 4  Train

```bash
pip install -r requirements.txt
python train.py --epochs 40 --batch-size 16
mlflow ui    # http://localhost:5000
```

Best checkpoint saved to `checkpoints/best.pt`.

### 5  Validate

```bash
python validate.py --checkpoint checkpoints/best.pt --threshold 0.7
```

Prints per-class accuracy, confusion matrix, and rejection analysis.

### 6  Deploy to Pi

```bash
# On your laptop — export weights
python -m deploy.export --checkpoint checkpoints/best.pt --out deploy/weights.npz

# Copy to Pi
scp -r . pi@raspberrypi.local:/home/pi/gesture_detection

# On the Pi — install runtime deps only
pip install -r requirements-pi.txt

# Install systemd service
sudo cp systemd/gesture.service /etc/systemd/system/
sudo systemctl enable gesture
sudo systemctl start gesture
```

---

## Success criteria

- Recognises gestures from someone who didn't contribute training data
- Stays quiet during idle motion (confidence threshold gating)
- Runs on battery, on boot, with no laptop attached

---

## Known risks

- HC-SR04 is noisy (~20 Hz); fast gestures may be unrecoverable — test your
  gesture set before committing to collecting a large dataset.
- Single-person training data will overfit; recruit at least one other pair
  of hands to improve generalisation.
