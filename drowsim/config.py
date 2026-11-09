"""Single source of truth for every tunable in DrowSim.

Both the client and the server import this module, so the two tiers can never
disagree about a threshold.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------- classes ---
# Class order is fixed here and reused by the YOLO dataset YAML, the BiLSTM
# head and the rule engine. Changing the order invalidates trained weights.
CLASSES = ("alert", "microsleep", "yawning")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_ALERT, IDX_MICROSLEEP, IDX_YAWNING = 0, 1, 2

# ------------------------------------------------------------ driver state ---
STATE_NORMAL = "NORMAL"
STATE_WARNING = "WARNING"
STATE_CRITICAL = "CRITICAL"
STATE_NO_FACE = "NO_FACE"

# ------------------------------------------------------------------ vision ---
# YOLOv8n, not YOLO11n. Published benchmarks favour the newer model, but on
# this task a like-for-like comparison (scripts/compare_models.py, identical
# data, hyperparameters and seed) measured v8n ahead on every quality metric
# at the same latency - and by nearly 3 points of F1 on microsleep, the class
# that matters most here. See runs/model_comparison.json.
YOLO_WEIGHTS = "models/drowsim_yolov8n.pt"
YOLO_IMG_SIZE = 320
YOLO_CONF = 0.25
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
# 10 Hz, matching the training sampling rate: FL3D video is 30 fps and
# prepare_yolo.py keeps every 3rd frame, so a 32-frame BiLSTM window spans
# 3.2 s there. Running the client faster would shrink that window to a
# different span of real time than the model was trained on.
TARGET_FPS = 10.0

# ----------------------------------------------------------------- temporal ---
SEQ_LEN = 32          # frames fed to the BiLSTM
VOTE_WINDOW = 9       # majority-vote smoothing on model predictions
FEATURE_DIM = 7       # see features.py

# ------------------------------------------------- rule-based safety path ---
# Hysteresis on the eyes-closed evidence, so a borderline frame cannot make the
# state flicker. Enter "closed" above CLOSE_ENTER, leave it below CLOSE_EXIT.
CLOSE_ENTER = 0.50
CLOSE_EXIT = 0.35
MICROSLEEP_MIN_S = 0.80    # sustained closure that is no longer a blink
BLINK_MAX_S = 0.40         # anything shorter is counted as a blink, not a risk

YAWN_ENTER = 0.50
YAWN_EXIT = 0.35
YAWN_MIN_S = 1.50          # talking/laughing opens the mouth only briefly
YAWN_REFRACTORY_S = 3.0    # one yawn must not be counted three times

# ------------------------------------------------------------------ PERCLOS ---
PERCLOS_WINDOW_S = 60.0    # the industry-standard one-minute sliding window
PERCLOS_MIN_S = 20.0       # do not trust the score until the window has filled
PERCLOS_WARN = 0.12
PERCLOS_CRITICAL = 0.15

# --------------------------------------------------------------- yawn rate ---
YAWN_RATE_WINDOW_S = 300.0
YAWN_RATE_WARN = 3         # yawns within the window that justify a warning

# ------------------------------------------------------------------- alerts ---
TONE_CRITICAL_HZ, TONE_CRITICAL_MS = 1500, 600
TONE_WARNING_HZ, TONE_WARNING_MS = 900, 350
ALERT_REFRACTORY_S = 2.0   # minimum gap between two audible alarms

# --------------------------------------------------------------- networking ---
SERVER_URL = "http://127.0.0.1:8000"
INFER_PATH = "/infer"
REQUEST_TIMEOUT_S = 1.5
TELEMETRY_HZ = 5.0         # aggregated telemetry rate, not the frame rate


@dataclass(frozen=True)
class Thresholds:
    """Snapshot of the tunables, attached to every evaluation result so a
    recorded run can always be traced back to the settings that produced it."""

    close_enter: float = CLOSE_ENTER
    close_exit: float = CLOSE_EXIT
    microsleep_min_s: float = MICROSLEEP_MIN_S
    perclos_warn: float = PERCLOS_WARN
    perclos_critical: float = PERCLOS_CRITICAL
    yawn_min_s: float = YAWN_MIN_S
