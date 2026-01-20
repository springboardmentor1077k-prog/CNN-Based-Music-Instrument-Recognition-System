import librosa
import numpy as np
from pathlib import Path

from backend.inference import audio_to_mel, predict_segments
from backend.utils import load_labels


def run_pipeline(
    audio_path: str,
    original_filename: str,
    threshold: float,
    aggregation: str
):


    # ---- Load audio ----
    y, sr = librosa.load(audio_path, sr=16000)
    duration = round(len(y) / sr, 2)

    # ---- Segment audio ----
    segment_len = sr * 2      # 2 seconds
    hop = sr                  # 1 second hop

    segments = []
    for start in range(0, len(y) - segment_len, hop):
        seg = y[start:start + segment_len]
        mel = audio_to_mel(seg, sr)
        segments.append(mel)

    if not segments:
        raise ValueError("Audio too short for segmentation")

    mel_segments = np.array(segments)

    # ---- Predict per segment ----
    segment_probs = predict_segments(mel_segments)
    # shape: (num_segments, num_classes)

    # ---- Load labels ----
    LABELS = load_labels()

    # ---- SAFETY CHECK (CRITICAL) ----
    if segment_probs.shape[1] != len(LABELS):
        raise ValueError(
            f"Model outputs {segment_probs.shape[1]} classes, "
            f"but {len(LABELS)} labels were provided."
        )

    # ---- Aggregate (TOP-K MEAN) ----
    if aggregation == "topk_mean":
        k = min(5, segment_probs.shape[0])
        sorted_probs = np.sort(segment_probs, axis=0)[-k:]
        agg_probs = sorted_probs.mean(axis=0)

    elif aggregation == "mean":
        agg_probs = segment_probs.mean(axis=0)

    elif aggregation == "max":
        agg_probs = segment_probs.max(axis=0)

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # ---- Confidence scores ----
    confidence_scores = {
        LABELS[i]: float(agg_probs[i])
        for i in range(len(agg_probs))
    }

    # ---- Final instruments ----
    final_instruments = [
        label
        for label, score in confidence_scores.items()
        if score >= threshold
    ]

    # ---- Result object ----
    return {
        "metadata": {
            "file_name": original_filename,
            "duration_sec": duration,
            "sample_rate": sr
        },

        "model": {
            "name": "Multilabel CNN",
            "version": "v1.0"
        },
        "predictions": {
            "final_instruments": final_instruments,
            "confidence_scores": confidence_scores,
            "timelines": segment_probs.tolist()
        }
    }
