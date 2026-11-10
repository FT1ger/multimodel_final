"""Logistic regression baselines on frozen pooled features."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from data import LABELS
from dataset import load_index

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = FINAL / "data" / "manifest_agreed.csv"
DEFAULT_CACHE = FINAL / "cache"
DEFAULT_OUT = FINAL / "results"
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def arrays(rows, image_map, image_feat, text_map, text_feat):
    """Build pooled feature matrices and labels for one split."""
    image = np.zeros((len(rows), image_feat.shape[1]), dtype=np.float32)
    text = np.zeros((len(rows), text_feat.shape[1]), dtype=np.float32)
    missing = rows["image_missing"].astype(float).values
    for i, (_, row) in enumerate(rows.iterrows()):
        if missing[i] == 0:
            image[i] = image_feat[image_map[str(row["image_id"])]]
        text[i] = text_feat[text_map[str(row["tweet_id"])]]
    label = np.array([LABELS.index(x) for x in rows["label"]])
    early = np.hstack([image, text, missing[:, None]])
    return image, text, early, label, missing


def fit_best(x_train, y_train, x_dev, y_dev):
    """Fit logistic regression for each C and keep the best dev macro-F1."""
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xd = scaler.transform(x_dev)
    best = None
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value, max_iter=3000, class_weight="balanced"
        ).fit(xt, y_train)
        score = f1_score(
            y_dev, model.predict(xd), average="macro", labels=range(5),
            zero_division=0,
        )
        if best is None or score > best[0]:
            best = (score, c_value, model)
    return scaler, best[1], best[2]


def scores(y, pred):
    """Return accuracy, macro-F1 and weighted F1."""
    return (
        accuracy_score(y, pred),
        f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
        f1_score(
            y, pred, average="weighted", labels=range(5), zero_division=0
        ),
    )


def main():
    """Fit four logistic baselines and save metrics and predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    splits = {
        name: manifest[manifest.split == name].reset_index(drop=True)
        for name in ["train", "dev", "test"]
    }
    image_map, image_feat = load_index(
        args.cache / "image_b0.npz", "image_id"
    )
    text_map, text_feat = load_index(
        args.cache / "text_bert.npz", "tweet_id"
    )
    data = {
        name: arrays(
            rows, image_map, image_feat, text_map, text_feat
        )
        for name, rows in splits.items()
    }
    rows = []
    preds = {}
    probs = {}
    for name, index in [("image", 0), ("text", 1), ("early", 2)]:
        scaler, c_value, model = fit_best(
            data["train"][index], data["train"][3],
            data["dev"][index], data["dev"][3],
        )
        for split in ["train", "dev", "test"]:
            x = scaler.transform(data[split][index])
            probs[name, split] = model.predict_proba(x)
            preds[name, split] = model.predict(x)
        acc, mf1, wf1 = scores(data["test"][3], preds[name, "test"])
        rows.append({
            "model": f"logreg_{name}",
            "setting": f"C={c_value:g}",
            "test_accuracy": acc,
            "test_macro_f1": mf1,
            "test_weighted_f1": wf1,
            "train_macro_f1": scores(
                data["train"][3], preds[name, "train"]
            )[1],
            "dev_macro_f1": scores(data["dev"][3], preds[name, "dev"])[1],
        })

    best_w = max(
        np.linspace(0, 1, 11),
        key=lambda w: f1_score(
            data["dev"][3],
            (
                w * (1 - data["dev"][4])[:, None]
                * probs["image", "dev"]
                + (1 - w * (1 - data["dev"][4])[:, None])
                * probs["text", "dev"]
            ).argmax(1),
            average="macro", labels=range(5), zero_division=0,
        ),
    )
    for split in ["train", "dev", "test"]:
        weight = best_w * (1 - data[split][4])[:, None]
        prob = (
            weight * probs["image", split]
            + (1 - weight) * probs["text", split]
        )
        probs["late", split] = prob
        preds["late", split] = prob.argmax(1)
    acc, mf1, wf1 = scores(data["test"][3], preds["late", "test"])
    rows.append({
        "model": "logreg_late",
        "setting": f"w_image={best_w:.1f}",
        "test_accuracy": acc,
        "test_macro_f1": mf1,
        "test_weighted_f1": wf1,
        "train_macro_f1": scores(
            data["train"][3], preds["late", "train"]
        )[1],
        "dev_macro_f1": scores(data["dev"][3], preds["late", "dev"])[1],
    })

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preds").mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "metrics_baseline.csv", index=False)
    for name in ["image", "text", "early", "late"]:
        test = splits["test"]
        frame = pd.DataFrame({
            "image_id": test["image_id"].values,
            "event_name": test["event_name"].values,
            "label": data["test"][3],
            "pred": preds[name, "test"],
        })
        for i, label in enumerate(LABELS):
            frame[f"prob_{label}"] = probs[name, "test"][:, i]
        frame.to_csv(
            args.out / "preds" / f"test_logreg_{name}_seed{args.seed}.csv",
            index=False,
        )
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
