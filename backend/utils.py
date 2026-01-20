import pandas as pd
from pathlib import Path


def load_labels():
    labels_csv = Path(__file__).resolve().parent.parent / "data" / "multilabel_labels.csv"

    df = pd.read_csv(labels_csv)

    # Assume first column is filename, rest are labels
    labels = list(df.columns[1:])

    return labels
