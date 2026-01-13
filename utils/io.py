# utils/io.py

import json
from config import CLASS_NAMES, WINDOW_SEC, HOP_SEC

def intensity_to_json(
    wav_name, times, intensities,
    aggregation, smoothing, threshold
):
    return {
        "audio_file": wav_name,
        "segment_duration_sec": WINDOW_SEC,
        "hop_duration_sec": HOP_SEC,
        "aggregation": aggregation,
        "smoothing": smoothing,
        "threshold": threshold,
        "classes": CLASS_NAMES,
        "timeline": [
            {
                "time_sec": float(t),
                "intensity": {
                    cls: float(vals[i])
                    for i, cls in enumerate(CLASS_NAMES)
                }
            }
            for t, vals in zip(times, intensities)
        ]
    }
