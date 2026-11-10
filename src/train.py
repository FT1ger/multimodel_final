"""Train and evaluate pooled and token-level fusion models."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from data import LABELS
from dataset import PooledDataset, TokenDataset, TokenPooledDataset
from models import (
    CrossAttentionFusion,
    DeepSingleModalityMLP,
    EarlyFusionMLP,
    LateFusionModel,
    SingleModalityMLP,
    count_parameters,
)
from preprocess import class_weights, set_seed

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = FINAL / "data" / "manifest_agreed.csv"
DEFAULT_OUT = FINAL / "results"


def make_loader(dataset, batch_size, shuffle):
    """Build a DataLoader with deterministic order when not shuffling."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


def forward_batch(model, batch, kind, device):
    """Run one batch and return logits or log-probabilities."""
    if kind == "token":
        image, image_mask, text, text_mask, label = batch
        image = image.to(device)
        image_mask = image_mask.to(device)
        text = text.to(device)
        text_mask = text_mask.to(device)
        return model(image, image_mask, text, text_mask), label.to(device)
    if kind == "image":
        image, _, _, label = batch
        return model(image.to(device)), label.to(device)
    if kind == "text":
        _, text, _, label = batch
        return model(text.to(device)), label.to(device)
    image, text, missing, label = batch
    output = model(image.to(device), text.to(device), missing.to(device))
    return output, label.to(device)


def evaluate_split(model, loader, criterion, kind, device):
    """Return labels, predictions, probabilities and mean loss."""
    model.eval()
    labels = []
    preds = []
    probs = []
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            output, target = forward_batch(model, batch, kind, device)
            if kind in {"late", "token_pooled"}:
                log_probs, _, _ = output
                loss = criterion(log_probs, target)
                prob = log_probs.exp()
            else:
                loss = criterion(output, target)
                prob = torch.softmax(output, dim=1)
            total += loss.item() * target.size(0)
            count += target.size(0)
            labels.extend(target.cpu().tolist())
            preds.extend(prob.argmax(1).cpu().tolist())
            probs.extend(prob.cpu().tolist())
    return {
        "labels": np.array(labels),
        "preds": np.array(preds),
        "probs": np.array(probs),
        "loss": total / max(count, 1),
    }


def train_model(
    model, train_loader, dev_loader, device, kind, weights, epochs,
    lr, patience,
):
    """Train with early stopping on validation macro-F1."""
    model = model.to(device)
    weight = weights.to(device)
    if kind in {"late", "token_pooled"}:
        criterion = nn.NLLLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            output, target = forward_batch(model, batch, kind, device)
            if kind in {"late", "token_pooled"}:
                log_probs, _, _ = output
                loss = criterion(log_probs, target)
            else:
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * target.size(0)
            count += target.size(0)
        dev = evaluate_split(model, dev_loader, criterion, kind, device)
        dev_f1 = f1_score(
            dev["labels"], dev["preds"], average="macro",
            labels=range(len(LABELS)), zero_division=0,
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": total / max(count, 1),
            "dev_loss": dev["loss"],
            "dev_macro_f1": dev_f1,
        })
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_f1


def metrics_dict(result, prefix="test"):
    """Return aggregate metrics for one split."""
    labels = result["labels"]
    preds = result["preds"]
    return {
        f"{prefix}_accuracy": accuracy_score(labels, preds),
        f"{prefix}_macro_f1": f1_score(
            labels, preds, average="macro", labels=range(len(LABELS)),
            zero_division=0,
        ),
        f"{prefix}_weighted_f1": f1_score(
            labels, preds, average="weighted", labels=range(len(LABELS)),
            zero_division=0,
        ),
        f"{prefix}_loss": result["loss"],
    }


def per_class_table(result):
    """Return per-class precision, recall and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        result["labels"], result["preds"], labels=range(len(LABELS))
    )
    return pd.DataFrame({
        "label": LABELS,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def build_model(
    name, pooled_dataset, token_dataset, token_pooled_dataset, seed, args
):
    """Build one model and its data loaders."""
    if name == "mlp_early":
        model = EarlyFusionMLP(
            pooled_dataset.image_dim, pooled_dataset.text_dim,
            dropout=args.dropout,
        )
        return model, "pooled", pooled_dataset
    if name == "late":
        model = LateFusionModel(
            pooled_dataset.image_dim, pooled_dataset.text_dim,
            dropout=args.dropout,
        )
        return model, "late", pooled_dataset
    if name == "mlp_image":
        model = SingleModalityMLP(
            pooled_dataset.image_dim, dropout=args.dropout
        )
        return model, "image", pooled_dataset
    if name == "mlp_text":
        model = SingleModalityMLP(
            pooled_dataset.text_dim, dropout=args.dropout
        )
        return model, "text", pooled_dataset
    if name == "mlp_image_deep":
        model = DeepSingleModalityMLP(
            pooled_dataset.image_dim, hidden1=768, hidden2=256,
            dropout=args.dropout,
        )
        return model, "image", pooled_dataset
    if name == "mlp_text_deep":
        model = DeepSingleModalityMLP(
            pooled_dataset.text_dim, hidden1=1152, hidden2=256,
            dropout=args.dropout,
        )
        return model, "text", pooled_dataset
    if name == "cross":
        model = CrossAttentionFusion(dropout=args.dropout)
        return model, "token", token_dataset
    if name == "token_late":
        model = LateFusionModel(
            token_pooled_dataset.image_dim,
            token_pooled_dataset.text_dim,
            dropout=args.dropout,
        )
        return model, "token_pooled", token_pooled_dataset
    raise ValueError(f"unknown model {name}")


def main():
    """Train the requested models for all seeds and save metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", default="mlp_early,late,cross")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image-key", default="b0")
    parser.add_argument("--shuffle-text", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    train_rows = manifest[manifest.split == "train"].reset_index(drop=True)
    dev_rows = manifest[manifest.split == "dev"].reset_index(drop=True)
    test_rows = manifest[manifest.split == "test"].reset_index(drop=True)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preds").mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(parents=True, exist_ok=True)

    pooled_train = PooledDataset(
        train_rows, image_key=args.image_key,
        shuffle_text=args.shuffle_text,
    )
    pooled_dev = PooledDataset(
        dev_rows, image_key=args.image_key,
        shuffle_text=args.shuffle_text, seed=1,
    )
    pooled_test = PooledDataset(
        test_rows, image_key=args.image_key,
        shuffle_text=args.shuffle_text, seed=2,
    )
    pooled_stats = pooled_train.fit_stats()
    for dataset in [pooled_train, pooled_dev, pooled_test]:
        dataset.apply_stats(pooled_stats)

    model_names = args.models.split(",")
    needs_token = any(
        name in {"cross", "token_late"} for name in model_names
    )
    if needs_token:
        token_train = TokenDataset(train_rows)
        token_dev = TokenDataset(dev_rows)
        token_test = TokenDataset(test_rows)
        token_stats = token_train.fit_stats()
        for dataset in [token_train, token_dev, token_test]:
            dataset.apply_stats(token_stats)
    else:
        token_train = token_dev = token_test = None

    if "token_late" in model_names:
        token_pooled_train = TokenPooledDataset(train_rows)
        token_pooled_dev = TokenPooledDataset(dev_rows)
        token_pooled_test = TokenPooledDataset(test_rows)
        token_pooled_stats = token_pooled_train.fit_stats()
        for dataset in [
            token_pooled_train, token_pooled_dev, token_pooled_test
        ]:
            dataset.apply_stats(token_pooled_stats)
    else:
        token_pooled_train = token_pooled_dev = token_pooled_test = None

    weights = class_weights(train_rows["label"], LABELS)
    if args.no_class_weights:
        weights = torch.ones(len(LABELS))

    print("train/dev/test", len(train_rows), len(dev_rows), len(test_rows))
    print("device", device)
    rows = []
    for name in model_names:
        for seed in [int(s) for s in args.seeds.split(",")]:
            set_seed(seed)
            model, kind, _ = build_model(
                name, pooled_train, token_train, token_pooled_train,
                seed, args,
            )
            if kind == "token":
                train_ds, dev_ds, test_ds = (
                    token_train, token_dev, token_test
                )
            elif kind == "token_pooled":
                train_ds, dev_ds, test_ds = (
                    token_pooled_train,
                    token_pooled_dev,
                    token_pooled_test,
                )
            else:
                train_ds, dev_ds, test_ds = (
                    pooled_train, pooled_dev, pooled_test
                )
            train_loader = make_loader(train_ds, args.batch, True)
            dev_loader = make_loader(dev_ds, args.batch, False)
            test_loader = make_loader(test_ds, args.batch, False)
            start = time.time()
            model, history, best_epoch, best_dev = train_model(
                model, train_loader, dev_loader, device, kind, weights,
                args.epochs, args.lr, args.patience,
            )
            train_seconds = time.time() - start
            criterion = (
                nn.NLLLoss(weight=weights.to(device))
                if kind in {"late", "token_pooled"}
                else nn.CrossEntropyLoss(weight=weights.to(device))
            )
            test = evaluate_split(
                model, test_loader, criterion, kind, device
            )
            row = {
                "model": name,
                "seed": seed,
                "image_key": args.image_key,
                "best_epoch": best_epoch,
                "best_dev_macro_f1": best_dev,
                "parameters": count_parameters(model),
                "train_seconds": train_seconds,
            }
            row.update(metrics_dict(test, "test"))
            row.update(metrics_dict(
                evaluate_split(model, train_loader, criterion, kind, device),
                "train",
            ))
            rows.append(row)
            pd.DataFrame(history).to_csv(
                args.out / f"history_{name}_seed{seed}.csv", index=False
            )
            per_class_table(test).to_csv(
                args.out / f"per_class_{name}_seed{seed}.csv", index=False
            )
            pred = pd.DataFrame({
                "image_id": test_rows["image_id"].values,
                "event_name": test_rows["event_name"].values,
                "label": test["labels"],
                "pred": test["preds"],
            })
            for i, label in enumerate(LABELS):
                pred[f"prob_{label}"] = test["probs"][:, i]
            pred.to_csv(
                args.out / "preds" / f"test_{name}_seed{seed}.csv",
                index=False,
            )
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                args.out / "models" / f"{name}_seed{seed}.pt",
            )
            print(
                f"{name} seed {seed}: dev {best_dev:.3f} "
                f"test {row['test_macro_f1']:.3f} "
                f"params {row['parameters']}"
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "metrics_runs.csv", index=False)
    print("\nmean and standard deviation by model")
    summary = table.groupby("model").agg(
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        accuracy_mean=("test_accuracy", "mean"),
        params=("parameters", "first"),
    )
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
