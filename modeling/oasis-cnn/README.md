# OASIS-1 dementia classifiers

Four models that classify OASIS-1 (dataset 2 — see `../../data_preparation/DATASET.md`)
T88_111 masked_gfc MRI slices (208x176, grayscale) as `Non_Demented` (CDR 0)
or `Demented` (CDR 0.5 and 1 merged), plus Grad-CAM / LIME / SHAP
explanations:

- `cnn` — 3-block CNN
- `vit` — small Vision Transformer (16x16 patches)
- `cnn_svm` — the same CNN with a linear SVM head (squared hinge loss)
- `hybrid` — CNN branch + self-attention branch, fused

## Data

**Not included here** — same reason as `data_preparation/OASIS-1/Data/`: the
OASIS data use agreement covers the prepped output too, so it can't go in
this public repo. Get `02_oasis_preparation.ipynb`'s output from the group
OneDrive (or generate it yourself per `EXTRACT_OASIS-1_DATA_HERE.md`), then
place these three files in `data/` (gitignored here, not tracked):

- `slices.npz` — array `slices`, shape `(4660, 208, 176)`, uint8
- `labels.csv` — one row per slice (same order as `slices.npz`), with
  `patient_id`, `label`, `class_name`, `severity`, `split` (train/val/test,
  split by patient), `cv_fold`, and metadata (`age`, `sex`, `mmse`, `nwbv`, ...).
  The notebook writes this as `data_final.csv` — rename it.
- `dataset_info.json` — class names and per-class loss weights

## Usage

```bash
pip install -r requirements.txt
python src/train.py --model cnn --cv     # 5-fold cross-validation (what we report)
python src/train.py --model cnn          # single train/val/test split
python src/run_xai.py                    # Grad-CAM / LIME / SHAP on the CV models
```

`--model` is one of `cnn`, `vit`, `cnn_svm`, `hybrid`. `train_all.bat` runs
5-fold CV for all four in turn. Other options:

- `--schedule {default,warmup,low_lr_warmup}` — try a different LR schedule
  (outputs are then named `<model>_<schedule>`)
- `--seed N` — retrain with a different training seed; folds and val
  patients stay the same, so seeds are compared on identical patients
- `src/seed_sweep.py --deadline <ISO time>` — reruns CV for every model over
  several seeds unattended; `src/summarize_seeds.py` writes
  `outputs/seed_summary.csv`

TensorFlow dropped native GPU support on Windows after 2.10, so to train on
the GPU run it under WSL2 with `tensorflow[and-cuda]`. On older (Pascal) cards
cuDNN 9.27 fails on convolutions — pin `nvidia-cudnn-cu12==9.3.0.75`. It does
run on the CPU, just ~3x slower.

[notebook.ipynb](notebook.ipynb) is the original step-by-step version of the
single-split pipeline and still predates the binary/CV changes.

## How it trains

- **5-fold CV.** Fold k's patients (the `cv_fold` column, ~47 patients,
  class-stratified) are the test set; a stratified 15% of the rest (28
  patients) is held out for early stopping and checkpointing; the remaining
  ~158 patients train. Every patient is tested exactly once, nobody is in two
  sets, and results are reported as the mean ± std over folds.
- **Class balance** is handled by oversampling each class equally per
  training batch, not by `class_weight` loss reweighting, since the latter
  repeatedly made the optimizer collapse to predicting a single class early
  in training.
- **Checkpointing/early stopping** monitor validation balanced accuracy
  (mean per-class recall) rather than `val_loss`, since `val_loss` on this
  small validation split favoured models that leaned toward the majority
  class.
- **Schedule.** Up to 60 epochs. All models use the `warmup` schedule: Adam at
  5e-4 with a 5-epoch linear warmup, and neither early stopping nor
  ReduceLROnPlateau allowed to act before epoch 15. Without that, the ViT and
  hybrid sometimes sat at chance for the first ~10 epochs and got stopped (or
  had their LR cut to nothing) before they'd started learning, and the CNN was
  being stopped early too. `warmup` was picked over `default` and
  `low_lr_warmup` for all four models by mean CV *validation* balanced
  accuracy — not by test score.
- **Metrics** are per-slice and patient-level (majority vote over a
  patient's 20 slices); patient-level is the clinically meaningful one.

Outputs (all gitignored apart from `seed_summary.csv`, since they're trained
directly on, or show, the restricted data):

- `checkpoints/best_model_<run>_fold<k>.keras` — per-fold best checkpoints
- `outputs/cv_results_<run>.csv` — per-fold metrics plus mean/std rows
- `outputs/training_history_*.png`, `outputs/confusion_matrix_test_*.png`
- `outputs/xai_<model>_<class>_<patient>.png` — XAI figures

## Results

5-fold CV, `warmup` schedule, 5 training seeds per model (123, 200, 210, 310,
410), patient level. Mean over the 25 (seed, fold) runs; brackets are the
range of the per-seed means. Full table in `outputs/seed_summary.csv`.

| Model | Balanced acc | Macro-F1 | Sensitivity (Demented) | Specificity (Non_Demented) |
|---|---|---|---|---|
| CNN | 0.723 [0.705–0.739] | 0.698 | 0.853 | 0.593 |
| ViT | 0.714 [0.702–0.730] | 0.692 | 0.819 | 0.609 |
| CNN-SVM | 0.715 [0.708–0.725] | 0.688 | 0.865 | 0.564 |
| Hybrid | 0.725 [0.708–0.740] | 0.700 | 0.848 | 0.601 |

Always guessing `Non_Demented` scores 0.50 balanced accuracy.

- **The four models are statistically tied.** Every pairwise gap is ≤0.011
  and every 95% bootstrap interval includes zero. Which patients land in the
  test fold accounts for essentially all the variation; model and seed
  account for almost none. Every model scores ~0.59–0.65 on folds 0 and 3
  and ~0.80–0.83 on fold 4.
- **All models over-predict Demented** — high sensitivity, but roughly 40% of
  healthy patients get flagged.
- **The single 36-patient test split was misleading.** It put the hybrid and
  ViT on top (0.75–0.77); under CV that ranking didn't hold.
- **XAI is what separates them.** The CNN and CNN-SVM Grad-CAMs highlight
  the ventricles and sulci (plausible atrophy markers). The hybrid's Grad-CAM
  mostly lights up the background, and the ViT has no conv layer for
  Grad-CAM at all. So the CNN is the recommended model — for its
  explanations, not its accuracy.

`outputs/experiment_log.csv` and `outputs/best_overall_info.json` are from the
earlier **3-class** 40-seed study (`src/overnight_runs.py`) and aren't
comparable with the binary results above.

## Limitations

- 2D single-slice classification — no 3D spatial context.
- **Age confound not yet checked.** Demented patients average ~7.5 years
  older (see `DATASET.md`); nobody has checked whether predictions track age
  more than diagnosis.
- Per-patient CV predictions aren't saved, only aggregate metrics, which the
  age check and the hard-fold analysis both need.
- XAI is qualitative only so far (a few patients, middle slice). There's no
  brain-mask or atlas overlap measure, and the LIME and SHAP setups need
  work: LIME blacks out regions, which looks like atrophy to the model, and
  the SHAP panel's colour scale hides negative attributions.
- The validation sets are small (28 patients), so the schedule selection is
  noisy too.
