from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"

SR = 16000
N_MELS = 128
FIXED_FRAMES = 128
