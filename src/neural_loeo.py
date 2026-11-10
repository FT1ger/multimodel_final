"""Leave-one-event-out evaluation for neural fusion models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from data import LABELS
from dataset import PooledDataset
from models import EarlyFusionMLP, LateFusionModel
from preprocess import class_weights, set_seed
from train import evaluate_split, make_loader, train_model

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = FINAL / "data" / "manifest_agreed.csv"
DEFAULT_CACHE = FINAL / "cache"
DEFAULT_OUT = FINAL / "results" / "report"
B0 = FINAL / "results_std" / "preds"
R50 = FINAL / "results_r50_std" / "preds"


def in_distribution(event, name, seed):
    """Return in-distribution predictions for one held-out event."""
    if name == "r50_late":
        path = R50 / f"test_late_seed{seed}.csv"
    else:
        path = B0 / f"test_mlp_early_seed{seed}.csv"
    frame = pd.read_csv(path)
    return frame[frame.event_name == event]


def evaluate(y, pred):
    """Return accuracy, fixed-label macro-F1 and present-label macro-F1."""
    return (
        float((np.asarray(y) == np.asarray(pred)).mean()),
        f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
        f1_score(y, pred, average="macro", labels=np.unique(y),
                 zero_division=0),
    )


def main():
    """Run neural LOEO for B0 early fusion and R50 late fusion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    splits = {
        name: manifest[manifest.split == name].reset_index(drop=True)
        for name in ["train", "dev", "test"]
    }
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [
        ("b0_early", "b0", "mlp_early", EarlyFusionMLP, "pooled"),
        ("r50_late", "r50", "late", LateFusionModel, "late"),
    ]
    for event in sorted(splits["test"]["event_name"].unique()):
        train_rows = splits["train"][
            splits["train"].event_name != event
        ].reset_index(drop=True)
        dev_rows = splits["dev"][
            splits["dev"].event_name != event
        ].reset_index(drop=True)
        test_rows = splits["test"][
            splits["test"].event_name == event
        ].reset_index(drop=True)
        weights = class_weights(train_rows["label"], LABELS)
        for name, image_key, stem, model_cls, kind in specs:
            train = PooledDataset(train_rows, image_key=image_key)
            dev = PooledDataset(dev_rows, image_key=image_key)
            test = PooledDataset(test_rows, image_key=image_key)
            stats = train.fit_stats()
            for dataset in [train, dev, test]:
                dataset.apply_stats(stats)
            train_loader = make_loader(train, args.batch, True)
            dev_loader = make_loader(dev, args.batch, False)
            test_loader = make_loader(test, args.batch, False)
            for seed in [int(s) for s in args.seeds.split(",")]:
                set_seed(seed)
                if kind == "late":
                    model = model_cls(train.image_dim, train.text_dim)
                else:
                    model = model_cls(train.image_dim, train.text_dim)
                model, _, best_epoch, _ = train_model(
                    model, train_loader, dev_loader, device, kind,
                    weights, args.epochs, args.lr, args.patience,
                )
                criterion = (
                    torch.nn.NLLLoss(weight=weights.to(device))
                    if kind == "late"
                    else torch.nn.CrossEntropyLoss(weight=weights.to(device))
                )
                result = evaluate_split(
                    model, test_loader, criterion, kind, device
                )
                acc, fixed, present = evaluate(
                    result["labels"], result["preds"]
                )
                ind = in_distribution(event, stem, seed)
                ind = ind.set_index("image_id").loc[
                    test_rows["image_id"]
                ]
                ind_f1 = f1_score(
                    result["labels"], ind["pred"].values,
                    average="macro", labels=range(5), zero_division=0,
                )
                rows.append({
                    "event": event,
                    "model": name,
                    "seed": seed,
                    "n_test": len(test_rows),
                    "best_epoch": best_epoch,
                    "accuracy": acc,
                    "macro_f1": fixed,
                    "present_macro_f1": present,
                    "delta": fixed - ind_f1,
                })
                print(
                    f"{event} {name} seed {seed}: "
                    f"{fixed:.3f} delta {fixed - ind_f1:+.3f}"
                )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "neural_loeo.csv", index=False)
    summary = table.groupby("model").agg(
        n_events=("event", "nunique"),
        macro_f1=("macro_f1", "mean"),
        present_macro_f1=("present_macro_f1", "mean"),
        delta=("delta", "mean"),
        delta_std=("delta", "std"),
    )
    summary.to_csv(args.out / "neural_loeo_summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
