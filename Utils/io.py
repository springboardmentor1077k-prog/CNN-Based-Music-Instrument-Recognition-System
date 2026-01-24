import json
from config import CLASS_NAMES, WINDOW_SEC, HOP_SEC
import numpy as np

def intensity_to_json(
    filename,
    times,
    smoothed_probs,
    aggregation_method,
    smoothing_window,
    threshold
):
    num_classes = smoothed_probs.shape[1]

    # ----------------------------
    # Per-hop activity
    # ----------------------------
    per_hop_activity = []

    for t, probs in zip(times, smoothed_probs):
        active = {
            CLASS_NAMES[i]: float(probs[i] * 100)
            for i in range(num_classes)
            if probs[i] >= threshold
        }

        per_hop_activity.append({
            "time": float(t),
            "active_instruments": active
        })

    # ----------------------------
    # Overall (final) aggregation
    # ----------------------------
    if aggregation_method == "mean":
        final_probs = np.mean(smoothed_probs, axis=0)
    else:
        final_probs = np.max(smoothed_probs, axis=0)

    overall_detected = {
        CLASS_NAMES[i]: float(final_probs[i] * 100)
        for i in range(num_classes)
        if final_probs[i] >= threshold
    }

    return {
        "file": filename,
        "aggregation_method": aggregation_method,
        "smoothing_window": smoothing_window,
        "threshold": threshold,
        "overall_detected_instruments": overall_detected,
        "per_hop_activity": per_hop_activity
    }
