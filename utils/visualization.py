# utils/visualization.py

import matplotlib.pyplot as plt
from config import CLASS_NAMES

def plot_intensity(times, intensities, threshold):
    plt.figure(figsize=(12, 4))
    for i, cls in enumerate(CLASS_NAMES):
        plt.plot(times, intensities[:, i], label=cls)
    plt.axhline(threshold, linestyle="--", color="red", alpha=0.4)
    plt.xlabel("Time (sec)")
    plt.ylabel("Intensity")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    return plt.gcf()
