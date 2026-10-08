"""Train diverse architectures on the OASIS-1 slice dataset and report
per-slice and patient-level (majority vote) metrics on the held-out test split.

With --cv, runs 5-fold cross-validation over the cv_fold column instead and
reports mean +/- std across folds (see dataset.load_cv_fold).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
)

from dataset import balanced_train_dataset, class_names, load_cv_fold, load_split
from model import (
    build_model,                  # Standard CNN
    build_vit_model,              # Vision Transformer
    build_cnn_svm_model,          # CNN-SVM Hybrid
    build_hybrid_cnn_transformer, # CNN-Transformer Hybrid
)

ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
EPOCHS = 60
NUM_FOLDS = 5
EARLY_STOP_START = 15
SEED = 123

# Dictionary mapping model keys to their building functions
MODEL_FACTORY = {
    "cnn": build_model,
    "vit": build_vit_model,
    "cnn_svm": build_cnn_svm_model,
    "hybrid": build_hybrid_cnn_transformer,
}

# Per-architecture learning-rate schedule. At lr=5e-4 the ViT often sat at a
# constant prediction (train loss ~0.698) from epoch 1 and only escaped once
# ReduceLROnPlateau had dropped it to ~1e-4 -- in one CV fold it never did.
# Transformers usually want a lower peak lr with warmup, and LR cuts held off
# until they've had a chance to start learning.
# The same candidate schedules are available to every model (--schedule), and
# each model's default below was picked by mean validation balanced accuracy
# across the CV folds (never by test score).
SCHEDULES = {
    "default": {"lr": 5e-4, "warmup_epochs": 0, "lr_decay_start": 0},
    "warmup": {"lr": 5e-4, "warmup_epochs": 5, "lr_decay_start": EARLY_STOP_START},
    "low_lr_warmup": {"lr": 1e-4, "warmup_epochs": 5, "lr_decay_start": EARLY_STOP_START},
}
# Validation picked "warmup" for all four (outputs/cv_results_<model>_<schedule>.csv).
MODEL_SCHEDULE = {
    "cnn": "warmup",
    "vit": "warmup",
    "cnn_svm": "warmup",
    "hybrid": "warmup",
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


class LinearWarmup(tf.keras.callbacks.Callback):
    """Ramps the learning rate linearly up to peak_lr over the first warmup_epochs."""

    def __init__(self, peak_lr, warmup_epochs):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            self.model.optimizer.learning_rate = self.peak_lr * (epoch + 1) / self.warmup_epochs


class DelayedReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau that doesn't start counting until start_epoch."""

    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            logs = logs or {}
            logs["learning_rate"] = float(np.asarray(self.model.optimizer.learning_rate))
            return
        super().on_epoch_end(epoch, logs)


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


def train_and_evaluate(model_key, data, tag, verbose=1, schedule_name=None, train_seed=SEED):
    """Train one model on data = {'train'|'val'|'test': (X, y, patient_ids)}
    and return a dict of per-slice and patient-level test metrics.

    tag names the output files: checkpoints/best_model_<tag>.keras etc.
    """
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, patient_ids_test = data["test"]
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    names = class_names()
    num_classes = len(names)

    checkpoint_path = ROOT / "checkpoints" / f"best_model_{tag}.keras"
    history_plot_path = ROOT / "outputs" / f"training_history_{tag}.png"
    confusion_plot_path = ROOT / "outputs" / f"confusion_matrix_test_{tag}.png"

    # Specific Configuration adjustments for SVM Head
    is_svm = model_key == "cnn_svm"
    train_ds = balanced_train_dataset(X_train, y_train, BATCH_SIZE, train_seed)
    if is_svm:
        # SVM Multi-class hinge loss expects one-hot encoded targets
        y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

        loss_fn = "squared_hinge"
        metrics = ["categorical_accuracy"]
        train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)))
        val_ds = make_dataset(X_val, y_val_encoded, training=False)
        test_ds = make_dataset(X_test, y_test_encoded, training=False)
    else:
        loss_fn = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
        val_ds = make_dataset(X_val, y_val, training=False)
        test_ds = make_dataset(X_test, y_test, training=False)

    steps_per_epoch = len(X_train) // BATCH_SIZE

    print(f"\nBuilding architecture profile: {model_key}...")
    model = MODEL_FACTORY[model_key](input_shape=X_train.shape[1:], num_classes=num_classes)

    schedule_name = schedule_name or MODEL_SCHEDULE[model_key]
    schedule = SCHEDULES[schedule_name]
    print(f"LR schedule '{schedule_name}': {schedule}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(schedule["lr"], clipnorm=1.0),
        loss=loss_fn,
        metrics=metrics,
    )
    if verbose:
        model.summary()

    checkpoint_path.parent.mkdir(exist_ok=True)
    callbacks = [
        BalancedAccuracy(X_val, y_val, is_svm=is_svm),
        # start_from_epoch: the ViT/hybrid sit at chance for the first ~10 epochs
        # before they start learning, so stopping 12 epochs after an early lucky
        # epoch killed some CV folds before training had really begun.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_acc", mode="max", patience=12,
            restore_best_weights=True, start_from_epoch=EARLY_STOP_START,
        ),
        LinearWarmup(schedule["lr"], schedule["warmup_epochs"]),
        DelayedReduceLROnPlateau(
            start_epoch=schedule["lr_decay_start"],
            monitor="val_balanced_acc", mode="max", factor=0.5, patience=5, min_lr=1e-6,
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
        verbose=verbose,
    )
    plot_history(history, history_plot_path)
    # EarlyStopping only tracks its best weights from EARLY_STOP_START on, so
    # reload the checkpoint (best over all epochs) to evaluate exactly what's saved
    model.load_weights(checkpoint_path)

    print("\nEvaluating on test split (per-slice)...")
    test_probs = model.predict(test_ds, verbose=0)
    test_pred = test_probs.argmax(axis=1)

    print(classification_report(y_test, test_pred, target_names=names, zero_division=0))
    print("Per-slice confusion matrix:\n", confusion_matrix(y_test, test_pred))

    print("\nEvaluating on test split (patient-level majority vote)...")
    patient_true, patient_pred = patient_level_predictions(
        test_probs, patient_ids_test, y_test
    )
    print(classification_report(patient_true, patient_pred, target_names=names, zero_division=0))
    cm_patient = confusion_matrix(patient_true, patient_pred, labels=range(num_classes))
    print("Patient-level confusion matrix:\n", cm_patient)
    plot_confusion(cm_patient, names, confusion_plot_path)

    print(f"\nBest model saved to {checkpoint_path}")
    print(f"Training curves saved to {history_plot_path}")
    print(f"Patient-level confusion matrix saved to {confusion_plot_path}")

    val_bal = history.history["val_balanced_acc"]
    results = {
        "epochs_run": len(val_bal),
        "best_epoch": int(np.argmax(val_bal)) + 1,
        "val_balanced_acc": max(val_bal),
        "slice_acc": accuracy_score(y_test, test_pred),
        "slice_balanced_acc": balanced_accuracy_score(y_test, test_pred),
        "slice_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "patient_acc": accuracy_score(patient_true, patient_pred),
        "patient_balanced_acc": balanced_accuracy_score(patient_true, patient_pred),
        "patient_macro_f1": f1_score(patient_true, patient_pred, average="macro", zero_division=0),
    }
    patient_recall = cm_patient.diagonal() / cm_patient.sum(axis=1).clip(min=1)
    for name, recall in zip(names, patient_recall):
        results[f"patient_recall_{name}"] = recall
    return results


def run_cross_validation(model_key, schedule_name=None, run_name=None, train_seed=SEED):
    """Train/evaluate on each of the NUM_FOLDS cv_fold splits, then summarize.

    run_name (default: model_key) names the output files. train_seed sets the
    weight init and batch sampling; the fold/val split always uses SEED, so runs
    with different train_seeds are evaluated on identical patients.
    """
    run_name = run_name or model_key
    rows = []
    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 20} {model_key}: fold {fold + 1}/{NUM_FOLDS} {'=' * 20}")
        tf.keras.backend.clear_session()
        tf.random.set_seed(train_seed + fold)
        data = load_cv_fold(fold, seed=SEED)
        results = train_and_evaluate(
            model_key, data, f"{run_name}_fold{fold}", verbose=2,
            schedule_name=schedule_name, train_seed=train_seed,
        )
        rows.append({"fold": fold, **results})

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != "fold"]
    summary = pd.DataFrame(
        [{"fold": "mean", **df[metric_cols].mean()}, {"fold": "std", **df[metric_cols].std()}]
    )
    df = pd.concat([df, summary], ignore_index=True)
    out_path = ROOT / "outputs" / f"cv_results_{run_name}.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.round(4).to_csv(out_path, index=False)

    print(f"\n{'=' * 20} {run_name}: {NUM_FOLDS}-fold CV summary {'=' * 20}")
    for col in metric_cols:
        if col in ("epochs_run", "best_epoch"):
            continue
        print(f"{col:>32}: {df.loc[df.fold == 'mean', col].item():.3f} "
              f"+/- {df.loc[df.fold == 'std', col].item():.3f}")
    print(f"Per-fold results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MRI Dementia Classifiers")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",  # <-- Allows running without parameters (defaults to cnn)
        choices=list(MODEL_FACTORY.keys()),
        help="Architecture type to train (default: cnn)"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help=f"Run {NUM_FOLDS}-fold cross-validation over the cv_fold column "
             "instead of the single train/val/test split",
    )
    parser.add_argument(
        "--schedule",
        choices=list(SCHEDULES.keys()),
        help="LR schedule to use instead of the model's default (MODEL_SCHEDULE); "
             "outputs are then named <model>_<schedule>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Training seed (weight init and batch sampling) instead of {SEED}; "
             "outputs are then named <model>_<schedule>_seed<seed>",
    )
    args = parser.parse_args()
    run_name = f"{args.model}_{args.schedule}" if args.schedule else args.model
    if args.seed is not None:
        run_name = f"{args.model}_{args.schedule or MODEL_SCHEDULE[args.model]}_seed{args.seed}"
    train_seed = SEED if args.seed is None else args.seed

    if args.cv:
        run_cross_validation(args.model, args.schedule, run_name, train_seed)
        return

    tf.random.set_seed(train_seed)
    print("Loading data...")
    data = {split: load_split(split) for split in ("train", "val", "test")}
    train_and_evaluate(args.model, data, run_name, schedule_name=args.schedule, train_seed=train_seed)


if __name__ == "__main__":
    main()
