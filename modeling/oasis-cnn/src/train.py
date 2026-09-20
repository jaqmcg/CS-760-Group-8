"""Train the CNN on the OASIS-1 slice dataset and report per-slice and
patient-level (majority vote) metrics on the held-out test split."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from dataset import balanced_train_dataset, class_names, load_split
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.keras"
HISTORY_PLOT_PATH = ROOT / "outputs" / "training_history.png"
CONFUSION_PLOT_PATH = ROOT / "outputs" / "confusion_matrix_test.png"

BATCH_SIZE = 64
EPOCHS = 40
SEED = 123


def make_dataset(images, labels, training, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class BalancedAccuracy(tf.keras.callbacks.Callback):
    """Adds val_balanced_acc (mean per-class recall) to the logs each epoch.

    Plain val_loss/val_accuracy picked the very first epoch as "best" here,
    since that's when the model still leaned toward the majority class and
    so had the lowest confidently-wrong penalty on this imbalanced,
    34-patient validation split -- even though later epochs were the ones
    actually learning to recognize all three classes. Balanced accuracy
    scores per-class recall equally, so it doesn't reward that shortcut.
    """

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        logs["val_balanced_acc"] = balanced_accuracy_score(self.y_val, preds)


def patient_level_predictions(probs, patient_ids, true_labels):
    """Majority-vote the per-slice softmax predictions up to one label per patient."""
    pred_labels = probs.argmax(axis=1)
    unique_patients = np.unique(patient_ids)

    patient_true, patient_pred = [], []
    for pid in unique_patients:
        mask = patient_ids == pid
        votes = np.bincount(pred_labels[mask], minlength=probs.shape[1])
        patient_pred.append(votes.argmax())
        # all slices for a patient share one label
        patient_true.append(true_labels[mask][0])
    return np.array(patient_true), np.array(patient_pred)


def plot_history(history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    HISTORY_PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(HISTORY_PLOT_PATH)
    plt.close(fig)


def plot_confusion(cm, labels):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    CONFUSION_PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(CONFUSION_PLOT_PATH)
    plt.close(fig)


def main():
    tf.random.set_seed(SEED)

    print("Loading data...")
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, patient_ids_test = load_split("test")
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    train_ds = balanced_train_dataset(X_train, y_train, BATCH_SIZE, SEED)
    steps_per_epoch = len(X_train) // BATCH_SIZE
    val_ds = make_dataset(X_val, y_val, training=False)
    test_ds = make_dataset(X_test, y_test, training=False)

    model = build_model(input_shape=X_train.shape[1:], num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    callbacks = [
        BalancedAccuracy(X_val, y_val),  # must run first to populate logs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_acc", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_balanced_acc", mode="max", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH, monitor="val_balanced_acc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    plot_history(history)

    print("\nEvaluating on test split (per-slice)...")
    test_probs = model.predict(test_ds)
    test_pred = test_probs.argmax(axis=1)
    names = class_names()

    print(classification_report(y_test, test_pred, target_names=names))
    cm_slice = confusion_matrix(y_test, test_pred)
    print("Per-slice confusion matrix:\n", cm_slice)

    print("\nEvaluating on test split (patient-level majority vote)...")
    patient_true, patient_pred = patient_level_predictions(
        test_probs, patient_ids_test, y_test
    )
    print(classification_report(patient_true, patient_pred, target_names=names))
    cm_patient = confusion_matrix(patient_true, patient_pred)
    print("Patient-level confusion matrix:\n", cm_patient)
    plot_confusion(cm_patient, names)

    print(f"\nBest model saved to {CHECKPOINT_PATH}")
    print(f"Training curves saved to {HISTORY_PLOT_PATH}")
    print(f"Patient-level confusion matrix saved to {CONFUSION_PLOT_PATH}")


if __name__ == "__main__":
    main()
