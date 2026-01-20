import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------------------------
# 🎨 Spectrogram
# -------------------------------------------------
def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")

    return fig


# -------------------------------------------------
# 🔥 Intensity Bars
# -------------------------------------------------
def plot_intensity_bars(top_predictions):
    names = [p["instrument"] for p in top_predictions]
    confidences = [p["confidence"] * 100 for p in top_predictions]

    fig = go.Figure(
        go.Bar(
            x=confidences,
            y=names,
            orientation="h",
            marker=dict(
                color=confidences,
                colorscale="Turbo"
            )
        )
    )

    fig.update_layout(
        title="Instrument Intensity",
        xaxis_title="Confidence (%)",
        yaxis_title="Instrument",
        template="plotly_dark",
        height=300
    )

    return fig


# -------------------------------------------------
# ⏱ Timeline
# -------------------------------------------------
def plot_timeline(timeline_data):
    fig = go.Figure()

    for inst, values in timeline_data.items():
        fig.add_trace(
            go.Scatter(
                y=values,
                mode="lines",
                name=inst
            )
        )

    fig.update_layout(
        title="Instrument Presence Over Time",
        xaxis_title="Segment Index",
        yaxis_title="Confidence",
        template="plotly_dark",
        height=300
    )

    return fig
