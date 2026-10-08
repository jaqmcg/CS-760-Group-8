"""Summarize the multi-seed CV runs from seed_sweep.py.

For each model (at its MODEL_SCHEDULE schedule) collects the original CV run
(seed 123, cv_results_<model>_<schedule>.csv) and every
cv_results_<model>_<schedule>_seed<N>.csv, then reports:
  - each seed's 5-fold mean,
  - the mean and std of those per-seed means (seed-to-seed noise),
  - the mean over seeds for each fold (which folds are hard regardless of seed).
Writes outputs/seed_summary.csv (one row per model/seed, plus mean/std rows).
"""
import re
from pathlib import Path

import pandas as pd

from train import MODEL_FACTORY, MODEL_SCHEDULE, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
METRICS = [
    "patient_balanced_acc", "patient_macro_f1", "patient_acc",
    "patient_recall_Demented", "patient_recall_Non_Demented", "slice_balanced_acc",
]


def seed_runs(model):
    run = f"{model}_{MODEL_SCHEDULE[model]}"
    paths = {}
    if (OUT / f"cv_results_{run}.csv").exists():
        paths[SEED] = OUT / f"cv_results_{run}.csv"
    for p in OUT.glob(f"cv_results_{run}_seed*.csv"):
        paths[int(re.search(r"_seed(\d+)\.csv$", p.name).group(1))] = p
    return run, dict(sorted(paths.items()))


def main():
    rows, fold_tables = [], {}
    for model in MODEL_FACTORY:
        run, paths = seed_runs(model)
        if not paths:
            continue
        per_fold = []
        for seed, path in paths.items():
            df = pd.read_csv(path)
            folds = df[~df["fold"].isin(["mean", "std"])].copy()
            folds["fold"] = folds["fold"].astype(int)
            folds["seed"] = seed
            per_fold.append(folds)
            rows.append({"model": model, "schedule": MODEL_SCHEDULE[model], "seed": seed,
                         **folds[METRICS].mean()})
        per_fold = pd.concat(per_fold)
        fold_tables[run] = per_fold.groupby("fold")["patient_balanced_acc"].agg(["mean", "std"])

        seeds = pd.DataFrame([r for r in rows if r["model"] == model])
        for stat in ("mean", "std"):
            rows.append({"model": model, "schedule": MODEL_SCHEDULE[model], "seed": stat,
                         **getattr(seeds[METRICS], stat)()})

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(OUT / "seed_summary.csv", index=False)

    print("Per-seed 5-fold means; 'mean'/'std' rows are across seeds.\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
        print("\nPatient balanced accuracy per fold, mean (std) across seeds:")
        for run, t in fold_tables.items():
            cells = "  ".join(f"f{f}: {m:.3f} ({s:.3f})" for f, (m, s) in t.iterrows())
            print(f"  {run:<28} {cells}")
    print(f"\nSaved to {OUT / 'seed_summary.csv'}")


if __name__ == "__main__":
    main()
