"""Run Grad-CAM / LIME / SHAP on the cross-validated models.

For each model, uses its selected LR schedule (train.MODEL_SCHEDULE) and the
CV fold checkpoint whose test fold contains the patient being explained, so
every explanation comes from a model that never saw that patient in training.
Picks --per-class patients per class (middle slice of each) and writes one
figure per model/patient to outputs/xai_<model>_<class>_<patient>.png.

Rebuilds each architecture from model.py and loads the trained weights into it,
rather than deserializing the .keras file, since the ViT's custom layers don't
implement get_config.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf

from dataset import class_names, load_cv_fold, load_labels
from train import MODEL_FACTORY, MODEL_SCHEDULE, SEED
from xai_explainer import main_explain

ROOT = Path(__file__).resolve().parent.parent
SHAP_BACKGROUND = 50


def pick_patients(labels_df, per_class, seed=SEED):
    """per_class patients from each class, as (patient_id, label, cv_fold)."""
    patients = labels_df.drop_duplicates("patient_id")
    rng = np.random.default_rng(seed)
    picked = []
    for label, group in patients.groupby("label"):
        rows = group.iloc[rng.choice(len(group), per_class, replace=False)]
        picked.extend(zip(rows["patient_id"], rows["label"], rows["cv_fold"]))
    return picked


def gradcam_layer(model):
    """Last stride-1 Conv2D: full-resolution features. Skips the hybrid's 16x16
    patch-projection conv, whose 13x11 map is too coarse to localize anything."""
    convs = [
        l for l in model.layers
        if isinstance(l, tf.keras.layers.Conv2D) and tuple(l.strides) == (1, 1)
    ]
    return convs[-1].name if convs else None


def main():
    parser = argparse.ArgumentParser(description="Run XAI on the cross-validated models")
    parser.add_argument(
        "--model", nargs="+", default=list(MODEL_FACTORY.keys()),
        choices=list(MODEL_FACTORY.keys()),
    )
    parser.add_argument("--per-class", type=int, default=2)
    args = parser.parse_args()

    names = class_names()
    labels_df = load_labels()
    patients = pick_patients(labels_df, args.per_class)
    (ROOT / "outputs").mkdir(exist_ok=True)

    for fold in sorted({f for _, _, f in patients}):
        data = load_cv_fold(fold, seed=SEED)
        X_train = data["train"][0]
        X_test, _, pid_test = data["test"]
        rng = np.random.default_rng(SEED)
        background = X_train[rng.choice(len(X_train), SHAP_BACKGROUND, replace=False)]

        for key in args.model:
            run_name = f"{key}_{MODEL_SCHEDULE[key]}"
            ckpt = ROOT / "checkpoints" / f"best_model_{run_name}_fold{fold}.keras"
            if not ckpt.exists():
                print(f"[skip] {run_name} fold {fold}: no checkpoint at {ckpt}")
                continue
            tf.keras.backend.clear_session()
            model = MODEL_FACTORY[key](input_shape=X_test.shape[1:], num_classes=len(names))
            model.load_weights(ckpt)

            for pid, label, f in patients:
                if f != fold:
                    continue
                idx = np.where(pid_test == pid)[0]
                i = idx[len(idx) // 2]
                out = ROOT / "outputs" / f"xai_{key}_{names[label]}_{pid}.png"
                print(f"[{run_name} fold {fold}] true={names[label]} patient={pid} -> {out.name}")
                tf.random.set_seed(SEED)
                np.random.seed(SEED)
                main_explain(
                    model, X_test[i], background_batch=background,
                    last_conv_layer_name=gradcam_layer(model),
                    class_names=names, save_path=out,
                )


if __name__ == "__main__":
    main()
