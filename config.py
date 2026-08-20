"""
Central configuration for the gesture-detection project.
Adjust pins, sample rates, and gesture labels here.
"""

# ---------------------------------------------------------------------------
# HC-SR04 GPIO pins (BCM numbering)
# ---------------------------------------------------------------------------
TRIG_PIN = 23
ECHO_PIN = 24

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 20          # HC-SR04 max reliable rate
WINDOW_SECONDS = 1.5         # length of one gesture window
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)  # 30 samples

MIN_DISTANCE_CM = 2.0        # below this → clamp / treat as noise
MAX_DISTANCE_CM = 40.0       # above this → treat as "no hand present"
IDLE_THRESHOLD_CM = 35.0     # distances above this are considered idle

# ---------------------------------------------------------------------------
# Gesture labels
# Extend or rename to suit your gesture set.
# ---------------------------------------------------------------------------
GESTURE_LABELS = [
    "swipe_left",
    "swipe_right",
    "push_down",
    "hover",
    "idle",
]
NUM_CLASSES = len(GESTURE_LABELS)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODEL_WEIGHTS_PATH = "models/weights.npz"
MLFLOW_TRACKING_URI = "mlruns"

# ---------------------------------------------------------------------------
# Training
# (hyper-parameters intentionally left here so you can tune them)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
VALIDATION_SESSION_HOLDOUT = 2   # number of recording sessions to hold out

# ---------------------------------------------------------------------------
# OLED (SSD1306)
# ---------------------------------------------------------------------------
OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_I2C_ADDRESS = 0x3C
