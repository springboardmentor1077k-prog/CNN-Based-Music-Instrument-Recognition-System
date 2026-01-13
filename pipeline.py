# pipeline.py

import numpy as np
from utils.audio import *
from utils.features import *
from utils.segmentation import *
from utils.aggregation import *
from utils.io import *
from config import TARGET_SR

def run_inference(
    audio_path,
    model,
    aggregation_method,
    threshold,
    smoothing_window
):
    y = load_audio(audio_path, TARGET_SR)

    segment_probs = []
    times = []

    for window, t in sliding_windows(y):
        window = peak_normalize(trim_silence(window))
        window = fix_duration(window, TARGET_SR, duration=3.0)

        mel = fix_mel_frames(generate_log_mel(window, TARGET_SR))
        mel = mel[np.newaxis, ..., np.newaxis]

        probs = model.predict(mel, verbose=0)[0]
        segment_probs.append(probs)
        times.append(t)

    segment_probs = np.array(segment_probs)

    smoothed = moving_average(segment_probs, smoothing_window)
    aggregated = aggregate(smoothed, aggregation_method)

    json_output = intensity_to_json(
        audio_path.split("/")[-1],
        times,
        smoothed,
        aggregation_method,
        smoothing_window,
        threshold
    )

    return smoothed, times, aggregated, json_output
