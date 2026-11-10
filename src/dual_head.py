"""Train a lightweight two-head model for text and image labels."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from data import LABELS
from dataset import load_index
from models import DualHeadModel, count_parameters
from preprocess import class_weights, set_seed

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = FINAL / "data" / "manifest_all.csv"
DEFAULT_CACHE = FINAL / "cache"
DEFAULT_OUT = FINAL / "results"


class DualHeadDataset(Dataset):
    """Pooled features with separate text and image labels."""

    def __init__(self, rows, cache):
        self.image_map, self.image_feat = load_index(
            cache / "image_b0.npz", "image_id"
        )
        self.text_map, self.text_feat = load_index(
            cache / "text_bert.npz", "tweet_id"
        )
        self.rows = rows.reset_index(drop=True)
        self.image_dim = self.image_feat.shape[1]
        self.text_dim = self.text_feat.shape[1]
        self.image_mean = None
        self.image_std = None
        self.text_mean = None
        self.text_std = None

    def fit_stats(self):
        """Return train-set means and standard deviations."""
        image_sum = torch.zeros(self.image_dim)
        image_sq = torch.zeros(self.image_dim)
        text_sum = torch.zeros(self.text_dim)
        text_sq = torch.zeros(self.text_dim)
        n_image = 0
        n_text = 0
        for index in range(len(self.rows)):
            row = self.rows.iloc[index]
            if float(row["image_missing"]) == 0:
                image = torch.from_numpy(
                    self.image_feat[self.image_map[str(row["image_id"])]]
                ).float()
                image_sum += image
                image_sq += image ** 2
                n_image += 1
            text = torch.from_numpy(
                self.text_feat[self.text_map[str(row["tweet_id"])]]
            ).float()
            text_sum += text
            text_sq += text ** 2
            n_text += 1
        image_mean = image_sum / max(n_image, 1)
        text_mean = text_sum / max(n_text, 1)
        image_std = (
            image_sq / max(n_image, 1) - image_mean ** 2
        ).clamp(min=1e-6).sqrt()
        text_std = (
            text_sq / max(n_text, 1) - text_mean ** 2
        ).clamp(min=1e-6).sqrt()
        return {
            "image_mean": image_mean,
            "image_std": image_std,
            "text_mean": text_mean,
            "text_std": text_std,
        }

    def apply_stats(self, stats):
        """Store train-set scaler statistics for this dataset."""
        self.image_mean = stats["image_mean"]
        self.image_std = stats["image_std"]
        self.text_mean = stats["text_mean"]
        self.text_std = stats["text_std"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        missing = float(row["image_missing"])
        image = torch.zeros(self.image_dim)
        if missing == 0:
            image = torch.from_numpy(
                self.image_feat[self.image_map[str(row["image_id"])]]
            ).float()
            if self.image_mean is not None:
                image = (image - self.image_mean) / self.image_std
        text = torch.from_numpy(
            self.text_feat[self.text_map[str(row["tweet_id"])]]
        ).float()
        if self.text_mean is not None:
            text = (text - self.text_mean) / self.text_std
        text_label = LABELS.index(row["label_text_5"])
        image_label = LABELS.index(row["label_image_5"])
        return (
            image, text, torch.tensor(missing),
            torch.tensor(text_label), torch.tensor(image_label),
        )


def evaluate(model, loader, device):
    """Return predictions for both heads."""
    model.eval()
    text_true, image_true = [], []
    text_pred, image_pred = [], []
    with torch.no_grad():
        for image, text, missing, ty, iy in loader:
            text_logits, image_logits = model(
                image.to(device), text.to(device), missing.to(device)
            )
            text_true.extend(ty.tolist())
            image_true.extend(iy.tolist())
            text_pred.extend(text_logits.argmax(1).cpu().tolist())
            image_pred.extend(image_logits.argmax(1).cpu().tolist())
    return (
        np.array(text_true), np.array(image_true),
        np.array(text_pred), np.array(image_pred),
    )


def train(model, train_loader, dev_loader, device, text_w, image_w,
          epochs, lr, patience):
    """Train the shared model with two classification losses."""
    model = model.to(device)
    text_loss = nn.CrossEntropyLoss(weight=text_w.to(device))
    image_loss = nn.CrossEntropyLoss(weight=image_w.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for image, text, missing, ty, iy in train_loader:
            image = image.to(device)
            text = text.to(device)
            missing = missing.to(device)
            ty = ty.to(device)
            iy = iy.to(device)
            text_logits, image_logits = model(image, text, missing)
            loss = text_loss(text_logits, ty)
            loss = loss + image_loss(image_logits, iy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ty)
            count += len(ty)
        tt, it, tp, ip = evaluate(model, dev_loader, device)
        text_f1 = f1_score(
            tt, tp, average="macro", labels=range(5), zero_division=0
        )
        image_f1 = f1_score(
            it, ip, average="macro", labels=range(5), zero_division=0
        )
        score = 0.5 * (text_f1 + image_f1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total / max(count, 1),
            "dev_text_macro_f1": text_f1,
            "dev_image_macro_f1": image_f1,
            "dev_mean_macro_f1": score,
        })
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_score


def main():
    """Train and evaluate the two-head model for three seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    splits = {
        name: manifest[manifest.split == name].reset_index(drop=True)
        for name in ["train", "dev", "test"]
    }
    datasets = {
        name: DualHeadDataset(rows, args.cache)
        for name, rows in splits.items()
    }
    stats = datasets["train"].fit_stats()
    for dataset in datasets.values():
        dataset.apply_stats(stats)
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch, shuffle=(name == "train")
        )
        for name, dataset in datasets.items()
    }
    text_w = class_weights(splits["train"]["label_text_5"], LABELS)
    image_w = class_weights(splits["train"]["label_image_5"], LABELS)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preds").mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        set_seed(seed)
        model = DualHeadModel(
            datasets["train"].image_dim, datasets["train"].text_dim
        )
        model, history, best_epoch, best_score = train(
            model, loaders["train"], loaders["dev"], device,
            text_w, image_w, args.epochs, args.lr, args.patience,
        )
        tt, it, tp, ip = evaluate(model, loaders["test"], device)
        text_f1 = f1_score(
            tt, tp, average="macro", labels=range(5), zero_division=0
        )
        image_f1 = f1_score(
            it, ip, average="macro", labels=range(5), zero_division=0
        )
        disagree = tt != it
        row = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_dev_mean": best_score,
            "text_macro_f1": text_f1,
            "image_macro_f1": image_f1,
            "label_agreement": float((tt == it).mean()),
            "pred_agreement": float((tp == ip).mean()),
            "n_disagree": int(disagree.sum()),
            "text_acc_on_disagree": float(
                (tp[disagree] == tt[disagree]).mean()
            ) if disagree.any() else float("nan"),
            "image_acc_on_disagree": float(
                (ip[disagree] == it[disagree]).mean()
            ) if disagree.any() else float("nan"),
            "parameters": count_parameters(model),
        }
        rows.append(row)
        pd.DataFrame(history).to_csv(
            args.out / f"history_dual_seed{seed}.csv", index=False
        )
        pd.DataFrame({
            "image_id": splits["test"]["image_id"].values,
            "event_name": splits["test"]["event_name"].values,
            "text_label": tt,
            "image_label": it,
            "text_pred": tp,
            "image_pred": ip,
        }).to_csv(
            args.out / "preds" / f"test_dual_seed{seed}.csv", index=False
        )
        print(
            f"dual seed {seed}: text {text_f1:.3f} "
            f"image {image_f1:.3f} pred agree {row['pred_agreement']:.3f}"
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "metrics_dual.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
