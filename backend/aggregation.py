import numpy as np

def aggregate_predictions(preds, method="mean"):
    """
    preds: (num_segments, num_classes)
    returns: (num_classes,)
    """
    if method == "mean":
        return preds.mean(axis=0)

    if method == "max":
        return preds.max(axis=0)

    if method == "vote":
        votes = np.argmax(preds, axis=1)
        counts = np.bincount(votes, minlength=preds.shape[1])
        return counts / counts.sum()

    raise ValueError("Invalid aggregation method")
