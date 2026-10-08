"""Loading and splitting the OASIS-1 MRI slice dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset_info():
    with open(DATA_DIR / "dataset_info.json") as f:
        return json.load(f)


def load_labels():
    return pd.read_csv(DATA_DIR / "labels.csv")


def load_slices():
    with np.load(DATA_DIR / "slices.npz") as d:
        return d["slices"]


def load_split(split):
    """Return (images, labels, patient_ids) for one of 'train'/'val'/'test'.

    Row order in labels.csv matches slice order in slices.npz.
    Images are float32 in [0, 1] with a trailing channel dimension.
    """
    labels_df = load_labels()
    slices = load_slices()

    mask = (labels_df["split"] == split).to_numpy()
    images = slices[mask].astype("float32") / 255.0
    images = images[..., np.newaxis]
    y = labels_df.loc[mask, "label"].to_numpy()
    patient_ids = labels_df.loc[mask, "patient_id"].to_numpy()
    return images, y, patient_ids


def class_names():
    return load_dataset_info()["classes"]


def balanced_train_dataset(images, labels, batch_size, seed):
    """Training dataset with each class equally likely per batch.

    A class_weight-weighted loss was tried first but repeatedly made the
    optimizer collapse to always predicting one class early in training
    (a known failure mode of aggressive loss reweighting on a small,
    imbalanced dataset) rather than learning real features. Balancing via
    oversampling instead avoids skewing the loss landscape: every batch is
    already class-balanced, so there's no "cheap" single-class solution.
    """
    num_classes = labels.max() + 1
    per_class = [
        tf.data.Dataset.from_tensor_slices(
            (images[labels == c], labels[labels == c])
        )
        .shuffle(int((labels == c).sum()), seed=seed)
        .repeat()
        for c in range(num_classes)
    ]
    ds = tf.data.Dataset.sample_from_datasets(
        per_class, weights=[1 / num_classes] * num_classes, seed=seed
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
