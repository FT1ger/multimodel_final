"""Aggregate metrics, bootstrap intervals and McNemar tests."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = FINAL / "results"
LABELS = list(range(5))
LABEL_NAMES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]


def macro_f1(y, pred):
    """Macro-F1 over the five fixed labels."""
    return f1_score(
        y, pred, average="macro", labels=LABELS, zero_division=0
    )


def bootstrap_diff(y, pred_a, pred_b, n=2000, seed=0):
    """Return the 95% interval of macro-F1(a) minus macro-F1(b)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = np.array([
        macro_f1(y[i], pred_a[i]) - macro_f1(y[i], pred_b[i])
        for i in idx
    ])
    return np.percentile(diffs, [2.5, 97.5])


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test on paired correct and incorrect predictions."""
    correct_a = pred_a == y
    correct_b = pred_b == y
    n01 = int(((~correct_a) & correct_b).sum())
    n10 = int((correct_a & (~correct_b)).sum())
    total = n01 + n10
    if total == 0:
        return 1.0
    return binomtest(min(n01, n10), total, 0.5).pvalue


def load_preds(results, model, seed):
    """Load one prediction file, using seed 0 for deterministic baselines."""
    path = results / "preds" / f"test_{model}_seed{seed}.csv"
    if not path.exists() and model.startswith("logreg"):
        path = results / "preds" / f"test_{model}_seed0.csv"
    return pd.read_csv(path)


def per_event_table(results, models, seeds):
    """Return per-event macro-F1 for each model and seed."""
    rows = []
    for model in models:
        for seed in seeds:
            df = load_preds(results, model, seed)
            for event, group in df.groupby("event_name"):
                rows.append({
                    "model": model,
                    "seed": seed,
                    "event": event,
                    "n": len(group),
                    "macro_f1": macro_f1(
                        group["label"].values, group["pred"].values
                    ),
                })
    return pd.DataFrame(rows)


def main():
    """Read saved predictions and print aggregate comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--models",
        default=(
            "logreg_image,logreg_text,logreg_early,logreg_late,"
            "mlp_early,late,cross"
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    frames = [pd.read_csv(args.results / "metrics_runs.csv")]
    baseline_path = args.results / "metrics_baseline.csv"
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path))
    metrics = pd.concat(frames, ignore_index=True)
    if "parameters" not in metrics:
        metrics["parameters"] = np.nan
    summary = metrics.groupby("model").agg(
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        accuracy_mean=("test_accuracy", "mean"),
        params=("parameters", "first"),
    )
    print("model summary")
    print(summary.round(3).to_string())

    rows = []
    for model_a, model_b in combinations(models, 2):
        for seed in seeds:
            a = load_preds(args.results, model_a, seed)
            b = load_preds(args.results, model_b, seed)
            y = a["label"].values
            lo, hi = bootstrap_diff(
                y, a["pred"].values, b["pred"].values, seed=seed
            )
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "seed": seed,
                "macro_f1_a": macro_f1(y, a["pred"].values),
                "macro_f1_b": macro_f1(y, b["pred"].values),
                "diff": macro_f1(y, a["pred"].values)
                - macro_f1(y, b["pred"].values),
                "ci_low": lo,
                "ci_high": hi,
                "mcnemar_p": mcnemar(
                    y, a["pred"].values, b["pred"].values
                ),
            })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(args.results / "pairwise.csv", index=False)
    print("\npairwise comparisons")
    print(pairwise.round(3).to_string(index=False))

    events = per_event_table(args.results, models, seeds)
    events.to_csv(args.results / "per_event.csv", index=False)
    print("\nper-event macro-F1")
    print(
        events.groupby(["model", "event"])["macro_f1"]
        .mean().unstack().round(3).to_string()
    )


if __name__ == "__main__":
    main()
