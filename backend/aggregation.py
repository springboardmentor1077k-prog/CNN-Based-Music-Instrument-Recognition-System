import numpy as np

def topk_mean(probs, k=3):
    topk = np.sort(probs, axis=0)[-k:]
    return topk.mean(axis=0)

def aggregate(probs, method="mean"):
    if method == "max":
        return probs.max(axis=0)
    return topk_mean(probs)
