import numpy as np

def aggregate(probs, method="mean"):
    if method == "mean":
        return np.mean(probs, axis=0)
    if method == "max":
        return np.max(probs, axis=0)
    if method == "voting":
        return (probs >= 0.5).mean(axis=0)
    raise ValueError("Invalid aggregation method")

def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    out = np.zeros_like(x)
    for i in range(x.shape[1]):
        out[:, i] = np.convolve(x[:, i], kernel, mode="same")[:x.shape[0]]
    return out