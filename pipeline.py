import numpy as np
from Utils.audio import *
from Utils.features import *
from Utils.segmentation import *
from Utils.aggregation import *
from Utils.io import *

from config import TARGET_SR, CLASS_NAMES


# ==================================================
# MAIN INFERENCE PIPELINE
# ==================================================
def run_inference(
    audio_path,
    model,
    aggregation_method,
    threshold,
    smoothing_window
):
    # load_audio returns ONLY y
    y = load_audio(audio_path, TARGET_SR)
    sr = TARGET_SR

    segment_probs = []
    times = []

    # sliding_windows expects ONLY y
    for window, t in sliding_windows(y):
        window = peak_normalize(trim_silence(window))
        window = fix_duration(window, sr, duration=3.0)

        mel = generate_log_mel(window, sr)
        mel = fix_mel_frames(mel)
        mel = mel[np.newaxis, ..., np.newaxis]

        probs = model.predict(mel, verbose=0)[0]
        segment_probs.append(probs)
        times.append(t)

    # Safety fallback
    if len(segment_probs) == 0:
        segment_probs = np.zeros((1, len(CLASS_NAMES)))
        times = [0.0]

    segment_probs = np.array(segment_probs)
    times = np.array(times)

    smoothed = moving_average(segment_probs, smoothing_window)
    aggregated = aggregate(smoothed, aggregation_method)

    json_output = intensity_to_json(
        audio_path.split("/")[-1],
        times.tolist(),
        smoothed,
        aggregation_method,
        smoothing_window,
        threshold
    )

    return smoothed, times, aggregated, json_output


