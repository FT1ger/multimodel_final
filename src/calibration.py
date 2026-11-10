"""Calibration and affected-class threshold analysis."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = FINAL / "results_std"
DEFAULT_R50 = FINAL / "results_r50_std"
OUT = FINAL / "results" / "report"
MODELS = [
    "logreg_late", "mlp_early", "late", "token_late", "cross",
    "r50_mlp_early", "r50_late",
]


def load_pred(model, seed, results, r50):
    """Load one prediction file."""
    if model.startswith("r50_"):
        stem = model.replace("r50_", "")
        path = r50 / "preds" / f"test_{stem}_seed{seed}.csv"
    else:
        path = results / "preds" / f"test_{model}_seed{seed}.csv"
    if not path.exists() and model.startswith("logreg"):
        path = results / "preds" / f"test_{model}_seed0.csv"
    return pd.read_csv(path) if path.exists() else None


def ece(frame, n_bins=10):
    """Expected calibration error for the five-class predictions."""
    probs = frame[[c for c in frame if c.startswith("prob_")]].values
    labels = frame["label"].values
    confidence = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(
                confidence[mask].mean() - correct[mask].mean()
            )
    return total


def threshold_table(frame):
    """Precision and recall for the affected class at thresholds."""
    probs = frame["prob_affected_individuals"].values
    labels = (frame["label"].values == 0).astype(int)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (probs >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                labels, pred, zero_division=0
            ),
            "recall": recall_score(labels, pred, zero_division=0),
            "positive_rate": pred.mean(),
        })
    return pd.DataFrame(rows)


def main():
    """Write calibration and threshold tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--r50", type=Path, default=DEFAULT_R50)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    threshold_rows = []
    for model in MODELS:
        for seed in [0, 1, 2]:
            frame = load_pred(model, seed, args.results, args.r50)
            if frame is None:
                continue
            rows.append({
                "model": model,
                "seed": seed,
                "ece": ece(frame),
            })
            if seed == 0 and model in {
                "logreg_late", "r50_mlp_early", "late", "r50_late"
            }:
                table = threshold_table(frame)
                table.insert(0, "model", model)
                threshold_rows.append(table)
    pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(
        OUT / "affected_thresholds.csv", index=False
    )
    print(pd.DataFrame(rows).groupby("model").ece.mean().round(3))


if __name__ == "__main__":
    main()
