# OASIS-1 CNN classifier

CNN that classifies OASIS-1 (dataset 2 — see `../../data_preparation/DATASET.md`)
T88_111 masked_gfc MRI slices (208x176, grayscale) into `Non_Demented`,
`Very_Mild_Demented`, or `Mild_Demented`.

## Data

**Not included here** — same reason as `data_preparation/OASIS-1/Data/`: the
OASIS data use agreement covers the prepped output too, so it can't go in
this public repo. Get `02_oasis_preparation.ipynb`'s output from the group
OneDrive (or generate it yourself per `EXTRACT_OASIS-1_DATA_HERE.md`), then
place these three files in `data/` (gitignored here, not tracked):

- `slices.npz` — array `slices`, shape `(4660, 208, 176)`, uint8
- `labels.csv` — one row per slice (same order as `slices.npz`), with
  `patient_id`, `label`, `class_name`, `split` (train/val/test, split by
  patient), `cv_fold`, and metadata (`age`, `sex`, `mmse`, `nwbv`, ...)
- `dataset_info.json` — class names and per-class loss weights

Note: this script trains on the single `train`/`val`/`test` split, not the
`cv_fold` column — see "Limitations" below.

## Usage

```bash
pip install -r requirements.txt
python src/train.py
```

Or run [notebook.ipynb](notebook.ipynb) for the same pipeline step by step
with inline visualizations (sample slices, training curves, confusion
matrices).

Trains the CNN using the provided `train`/`val`/`test` split (patient-level,
no leakage). Classes are imbalanced (2700/1400/560 slices) — this is
handled by oversampling each class equally per training batch, not by
`class_weight` loss reweighting, since the latter repeatedly made the
optimizer collapse to predicting a single class early in training.
Checkpointing/early-stopping monitor a custom balanced-accuracy metric
(mean per-class recall) rather than `val_loss`, since `val_loss` on this
small validation split favored a model that leaned toward the majority
class over ones that had actually learned to tell the three classes apart.

Reports both per-slice metrics and patient-level metrics (majority vote
across a patient's slices), since patient-level accuracy is the more
clinically meaningful number. Outputs:

- `checkpoints/best_model.keras` — best model by validation balanced accuracy
  (gitignored — not committed, since it's trained directly on the restricted
  data)
- `outputs/training_history.png` — loss/accuracy curves
- `outputs/confusion_matrix_test.png` — patient-level confusion matrix

`src/overnight_runs.py` reruns training with 40 different seeds to measure
result stability rather than trusting one run; `outputs/experiment_log.csv`
and `outputs/best_overall_info.json` are the results of that (safe to
commit — aggregate metrics only, no patient data).

## Results (40 independent runs, different seeds)

Patient-level macro-F1: mean 0.468, median 0.465, std dev 0.061, range
0.28–0.61. Per-class mean recall: `Mild_Demented` 96%, `Non_Demented` 69%,
`Very_Mild_Demented` 16% — this last one is the consistent weak point
across every run, not a one-off.

## Limitations

- Trains on the fixed `train`/`val`/`test` split only — doesn't use the
  `cv_fold` column, so it isn't the 5-fold cross-validation study
  `DATASET.md` mentions ("made 5 cross-validation folds ... so we can
  report an average instead of trusting one small test set"). The 40-seed
  rerun above is a different kind of robustness check (same split,
  different init) and isn't a substitute for that.
- 2D single-slice classification — no 3D spatial context, no atlas
  alignment used for interpretability.
- No XAI/explainability (Grad-CAM, SHAP, etc.) implemented in this piece.
- Small test set (36 patients, 4 in the smallest class) — individual run
  metrics carry real sampling noise, hence the 40-run study above.
