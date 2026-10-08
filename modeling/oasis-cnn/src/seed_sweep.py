"""Re-run 5-fold CV for every model with extra training seeds, unattended.

Each model uses its selected schedule (train.MODEL_SCHEDULE, resolved by
train.py at launch). Only the training seed changes (weight init and batch
sampling); folds and val splits stay fixed, so every seed is scored on the same
patients and the spread across seeds is pure training noise.

Runs seed-major (all models for one seed, then the next seed), so if the
deadline cuts the sweep short every model has the same number of seeds.
Waits until the GPU is free before starting, skips runs whose
cv_results_*.csv already exists (so it can be restarted), and writes
outputs/seed_summary.csv at the end (see summarize_seeds.py).

Usage: python src/seed_sweep.py --deadline 2026-10-09T08:30
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "outputs"
STATUS_LOG = OUT / "seed_sweep_status.log"

MODELS = ["cnn_svm", "vit", "hybrid", "cnn"]
SEEDS = [200, 300, 400, 500, 600, 700]  # seed 123 (the original CV runs) counts as the first
GPU_FREE_MIB = 1500
GPU_FREE_CHECKS = 10  # consecutive free checks, 30 s apart, before starting
INITIAL_RUN_ESTIMATE = timedelta(minutes=28)
# The tuning runs that must finish before the GPU is ours
PREREQS = ["cv_results_hybrid_low_lr_warmup.csv", "cv_results_vit_warmup.csv"]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(STATUS_LOG, "a") as f:
        f.write(line + "\n")


def gpu_memory_used():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(deadline):
    log("Waiting for the tuning runs to finish and the GPU to be free...")
    free_streak = 0
    while datetime.now() < deadline:
        prereqs_done = all((OUT / p).exists() for p in PREREQS)
        mem = gpu_memory_used()
        free_streak = free_streak + 1 if prereqs_done and mem is not None and mem < GPU_FREE_MIB else 0
        if free_streak >= GPU_FREE_CHECKS:
            log(f"GPU free ({mem} MiB used); starting.")
            return True
        time.sleep(30)
    return False


def schedule_for(model):
    """MODEL_SCHEDULE as it is in train.py right now (it may change after tuning)."""
    out = subprocess.run(
        [sys.executable, "-c", f"from train import MODEL_SCHEDULE; print(MODEL_SCHEDULE['{model}'])"],
        cwd=SRC, capture_output=True, text=True, check=True,
    ).stdout
    return out.strip().splitlines()[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", required=True, help="Don't start runs that would end after this (ISO time)")
    parser.add_argument("--no-wait", action="store_true", help="Start without waiting for the GPU")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    OUT.mkdir(exist_ok=True)

    log(f"Seed sweep started (seeds {args.seeds}); deadline {deadline:%Y-%m-%d %H:%M}")
    if not args.no_wait and not wait_for_gpu(deadline):
        log("Deadline reached while waiting for the GPU; nothing run.")
        return

    durations = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    stop = False
    for seed in args.seeds:
        for model in MODELS:
            schedule = schedule_for(model)
            run_name = f"{model}_{schedule}_seed{seed}"
            if (OUT / f"cv_results_{run_name}.csv").exists():
                log(f"[skip] {run_name}: already done")
                continue
            estimate = max(durations, default=INITIAL_RUN_ESTIMATE)
            if datetime.now() + estimate > deadline:
                log(f"Stopping: {run_name} (~{estimate.seconds // 60} min) wouldn't finish before the deadline.")
                stop = True
                break

            log(f"[start] {run_name}")
            started = datetime.now()
            with open(OUT / f"train_cv_{run_name}.log", "w") as f:
                code = subprocess.run(
                    [sys.executable, "train.py", "--model", model, "--cv", "--seed", str(seed)],
                    cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=env,
                ).returncode
            took = datetime.now() - started
            if code == 0:
                durations.append(took)
                log(f"[done]  {run_name} in {took.seconds // 60} min")
            else:
                log(f"[FAIL]  {run_name} exited {code} after {took.seconds // 60} min; see train_cv_{run_name}.log")
        if stop:
            break

    log("Writing seed summary...")
    subprocess.run([sys.executable, "summarize_seeds.py"], cwd=SRC, env=env,
                   stdout=open(OUT / "seed_summary.txt", "w"), stderr=subprocess.STDOUT)
    log("Seed sweep finished.")


if __name__ == "__main__":
    main()
