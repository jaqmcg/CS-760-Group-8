"""Train diverse architectures on the OASIS-1 slice dataset and report 
per-slice and patient-level (majority vote) metrics on the held-out test split.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from dataset import balanced_train_dataset, class_names, load_split
from model import (
    build_model,                  # Standard CNN
    build_vit_model,              # Vision Transformer
    build_cnn_svm_model,          # CNN-SVM Hybrid
    build_hybrid_cnn_transformer, # CNN-Transformer Hybrid
)

ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
EPOCHS = 40
SEED = 123

# Dictionary mapping model keys to their building functions
MODEL_FACTORY = {
    "cnn": build_model,
    "vit": build_vit_model,
    "cnn_svm": build_cnn_svm_model,
    "hybrid": build_hybrid_cnn_transformer,
}


def make_dataset(images, labels, training, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class BalancedAccuracy(tf.keras.callbacks.Callback):
    """Adds val_balanced_acc (mean per-class recall) to the logs each epoch."""

    def __init__(self, X_val, y_val, is_svm=False):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.is_svm = is_svm

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        logs["val_balanced_acc"] = balanced_accuracy_score(self.y_val, preds)


def patient_level_predictions(probs, patient_ids, true_labels):
    """Majority-vote the per-slice predictions up to one label per patient."""
    pred_labels = probs.argmax(axis=1)
    unique_patients = np.unique(patient_ids)

    patient_true, patient_pred = [], []
    for pid in unique_patients:
        mask = patient_ids == pid
        votes = np.bincount(pred_labels[mask], minlength=probs.shape[1])
        patient_pred.append(votes.argmax())
        patient_true.append(true_labels[mask][0])
    return np.array(patient_true), np.array(patient_pred)


def plot_history(history, save_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accommodate metric names if using categorical/hinge accuracy
    acc_key = "accuracy" if "accuracy" in history.history else "categorical_accuracy"
    val_acc_key = f"val_{acc_key}"
    
    if acc_key in history.history:
        axes[1].plot(history.history[acc_key], label="train")
        axes[1].plot(history.history[val_acc_key], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()

    fig.tight_layout()
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion(cm, labels, save_path):
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
    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train MRI Dementia Classifiers")
    parser.add_argument(
        "--model", 
        type=str, 
        default="cnn",  # <-- Allows running without parameters (defaults to cnn)
        choices=list(MODEL_FACTORY.keys()),
        help="Architecture type to train (default: cnn)"
    )
    args = parser.parse_args()

    # Dynamic Model-Specific Output Paths
    checkpoint_path = ROOT / "checkpoints" / f"best_model_{args.model}.keras"
    history_plot_path = ROOT / "outputs" / f"training_history_{args.model}.png"
    confusion_plot_path = ROOT / "outputs" / f"confusion_matrix_test_{args.model}.png"

    tf.random.set_seed(SEED)

    print("Loading data...")
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, patient_ids_test = load_split("test")
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Specific Configuration adjustments for SVM Head
    is_svm = args.model == "cnn_svm"
    if is_svm:
        # SVM Multi-class hinge loss expects one-hot encoded targets
        y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=3)
        y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=3)
        
        loss_fn = "squared_hinge"
        metrics = ["categorical_accuracy"]
        
        train_ds = balanced_train_dataset(X_train, y_train, BATCH_SIZE, SEED)
        train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=3)))
        
        val_ds = make_dataset(X_val, y_val_encoded, training=False)
        test_ds = make_dataset(X_test, y_test_encoded, training=False)
    else:
        loss_fn = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
        train_ds = balanced_train_dataset(X_train, y_train, BATCH_SIZE, SEED)
        val_ds = make_dataset(X_val, y_val, training=False)
        test_ds = make_dataset(X_test, y_test, training=False)

    steps_per_epoch = len(X_train) // BATCH_SIZE

    print(f"\nBuilding architecture profile: {args.model}...")
    model = MODEL_FACTORY[args.model](input_shape=X_train.shape[1:], num_classes=3)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4, clipnorm=1.0),
        loss=loss_fn,
        metrics=metrics,
    )
    model.summary()

    checkpoint_path.parent.mkdir(exist_ok=True)
    callbacks = [
        BalancedAccuracy(X_val, y_val, is_svm=is_svm), 
        tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_acc", mode="max", patience=12, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_balanced_acc", mode="max", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_balanced_acc", mode="max", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    plot_history(history, history_plot_path)

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
    plot_confusion(cm_patient, names, confusion_plot_path)

    print(f"\nBest model saved to {checkpoint_path}")
    print(f"Training curves saved to {history_plot_path}")
    print(f"Patient-level confusion matrix saved to {confusion_plot_path}")


if __name__ == "__main__":
    main()

