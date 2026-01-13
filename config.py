# config.py

TARGET_SR = 16000

WINDOW_SEC = 3.0
HOP_SEC = 1.5

N_MELS = 128
TARGET_FRAMES = 126
EPS = 1e-8

NUM_CLASSES = 11

CLASS_NAMES = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio", "voi"
]
