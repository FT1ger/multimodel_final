"""Measure training time and inference latency for neural models."""
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import PooledDataset, TokenDataset, TokenPooledDataset
from models import (
    CrossAttentionFusion,
    EarlyFusionMLP,
    LateFusionModel,
    count_parameters,
)

FINAL = Path(__file__).resolve().parents[1]
MANIFEST = FINAL / "data" / "manifest_agreed.csv"
OUT = FINAL / "results" / "report"
B0 = FINAL / "results_std"
R50 = FINAL / "results_r50_std"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def model_spec(name, train_ds, token_train, token_pooled_train):
    """Return model, kind and test dataset for one model name."""
    if name in {"mlp_early", "late"}:
        return {
            "model": (
                EarlyFusionMLP(train_ds.image_dim, train_ds.text_dim)
                if name == "mlp_early"
                else LateFusionModel(train_ds.image_dim, train_ds.text_dim)
            ),
            "kind": "pooled",
            "train": train_ds,
        }
    if name in {"r50_mlp_early", "r50_late"}:
        return {
            "model": (
                EarlyFusionMLP(train_ds.image_dim, train_ds.text_dim)
                if name == "r50_mlp_early"
                else LateFusionModel(train_ds.image_dim, train_ds.text_dim)
            ),
            "kind": "pooled",
            "train": train_ds,
        }
    if name == "token_late":
        return {
            "model": LateFusionModel(
                token_pooled_train.image_dim, token_pooled_train.text_dim
            ),
            "kind": "token_pooled",
            "train": token_pooled_train,
        }
    if name == "cross":
        return {
            "model": CrossAttentionFusion(),
            "kind": "token",
            "train": token_train,
        }
    raise ValueError(name)


def load_state(model, path):
    """Load a saved CPU state dict."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def make_test(name, train_rows, test_rows):
    """Build the matching standardized test dataset."""
    if name in {"r50_mlp_early", "r50_late"}:
        train = PooledDataset(train_rows, image_key="r50")
        test = PooledDataset(test_rows, image_key="r50")
        return train, test
    if name in {"mlp_early", "late"}:
        train = PooledDataset(train_rows, image_key="b0")
        test = PooledDataset(test_rows, image_key="b0")
        return train, test
    if name == "token_late":
        train = TokenPooledDataset(train_rows)
        test = TokenPooledDataset(test_rows)
        return train, test
    if name == "cross":
        train = TokenDataset(train_rows)
        test = TokenDataset(test_rows)
        return train, test
    raise ValueError(name)


def forward(model, batch, kind):
    """Run one batch without labels."""
    if kind == "token":
        image, image_mask, text, text_mask, _ = batch
        return model(
            image.to(DEVICE), image_mask.to(DEVICE),
            text.to(DEVICE), text_mask.to(DEVICE),
        )
    image, text, missing, _ = batch
    return model(image.to(DEVICE), text.to(DEVICE), missing.to(DEVICE))


def main():
    """Write train time, inference latency and throughput."""
    manifest = pd.read_csv(MANIFEST, dtype=str)
    train_rows = manifest[manifest.split == "train"].reset_index(drop=True)
    test_rows = manifest[manifest.split == "test"].reset_index(drop=True)
    specs = {
        "mlp_early": (
            B0, "mlp_early", EarlyFusionMLP
        ),
        "late": (
            B0, "late", LateFusionModel
        ),
        "token_late": (
            B0, "token_late", LateFusionModel
        ),
        "cross": (
            B0, "cross", CrossAttentionFusion
        ),
        "r50_mlp_early": (
            R50, "mlp_early", EarlyFusionMLP
        ),
        "r50_late": (
            R50, "late", LateFusionModel
        ),
    }
    metrics_b0 = pd.read_csv(B0 / "metrics_runs.csv")
    metrics_r50 = pd.read_csv(R50 / "metrics_runs.csv")
    rows = []
    for name, (folder, stem, _) in specs.items():
        train, test = make_test(name, train_rows, test_rows)
        stats = train.fit_stats()
        train.apply_stats(stats)
        test.apply_stats(stats)
        spec = model_spec(name, train, train, train)
        model = spec["model"].to(DEVICE).eval()
        path = folder / "models" / f"{stem}_seed0.pt"
        if not path.exists():
            continue
        load_state(model, path)
        loader = DataLoader(test, batch_size=32, shuffle=False)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                forward(model, batch, spec["kind"])
                if i >= 2:
                    break
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                forward(model, batch, spec["kind"])
        torch.mps.synchronize() if torch.backends.mps.is_available() else None
        elapsed = time.perf_counter() - start
        train_seconds = None
        source = metrics_b0 if folder == B0 else metrics_r50
        match = source[
            (source.model == stem) & (source.seed == 0)
        ]
        if len(match):
            train_seconds = float(match.train_seconds.iloc[0])
        rows.append({
            "model": name,
            "parameters": count_parameters(model),
            "train_seconds": train_seconds,
            "test_seconds": elapsed,
            "ms_per_sample": elapsed * 1000 / len(test),
            "samples_per_second": len(test) / elapsed,
        })
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "efficiency.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
