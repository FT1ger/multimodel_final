"""Recompute test metrics after removing duplicate-flagged test images."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = FINAL / "results_std"
DEFAULT_R50 = FINAL / "results_r50_std"
FLAGS = FINAL / "results" / "report" / "dedup_flags.csv"
OUT = FINAL / "results" / "report"

MODELS = [
    "logreg_image", "logreg_text", "logreg_early", "logreg_late",
    "mlp_image", "mlp_text", "mlp_early", "late", "token_late", "cross",
    "r50_mlp_image", "r50_mlp_early", "r50_late",
]


def macro_f1(frame):
    """Macro-F1 on the five fixed labels."""
    return f1_score(
        frame["label"], frame["pred"], average="macro",
        labels=range(5), zero_division=0,
    )


def load_pred(model, seed, results, r50):
    """Load a prediction file, falling back to seed 0 for baselines."""
    if model.startswith("r50_"):
        stem = model.replace("r50_", "")
        path = r50 / "preds" / f"test_{stem}_seed{seed}.csv"
    else:
        path = results / "preds" / f"test_{model}_seed{seed}.csv"
    if not path.exists() and model.startswith("logreg"):
        if model.startswith("r50_"):
            path = r50 / "preds" / f"test_{stem}_seed0.csv"
        else:
            path = results / "preds" / f"test_{model}_seed0.csv"
    return pd.read_csv(path) if path.exists() else None


def main():
    """Compare full and deduplicated test scores for all models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--r50", type=Path, default=DEFAULT_R50)
    args = parser.parse_args()
    flags = pd.read_csv(FLAGS)
    rows = []
    for model in MODELS:
        for seed in [0, 1, 2]:
            frame = load_pred(model, seed, args.results, args.r50)
            if frame is None:
                continue
            for flag in ["phash", "dhash", "union", "text"]:
                ids = set(flags.loc[flags[flag] == 1, "image_id"])
                kept = frame[~frame["image_id"].isin(ids)]
                rows.append({
                    "model": model,
                    "seed": seed,
                    "flag": flag,
                    "flagged": len(ids),
                    "n_full": len(frame),
                    "n_kept": len(kept),
                    "macro_f1_full": macro_f1(frame),
                    "macro_f1_deduplicated": macro_f1(kept),
                    "delta": macro_f1(kept) - macro_f1(frame),
                })
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "sensitivity_all.csv", index=False)
    print(table.groupby("flag").delta.agg(["mean", "min", "max"]).round(4))
    print("\nmax absolute delta")
    print(
        table.loc[table.delta.abs().idxmax()].to_dict()
    )


if __name__ == "__main__":
    main()
