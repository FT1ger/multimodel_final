"""Summaries of the class weight control and the logistic error analysis.

Expects train.py outputs in results_rerun/weighted and
results_rerun/unweighted (the second run with --no-class-weights).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from baseline import DEFAULT_CACHE, DEFAULT_MANIFEST, FINAL, arrays
from dataset import load_index

RERUN = FINAL / "results_rerun"
REPORT = FINAL / "results" / "report"
FIXED_C = {"image": 1e-3, "text": 1e-3, "early": 1e-2}
IMAGE_WEIGHT = 0.5
OTHER = 4


def macro_f1(y, pred):
    """Return macro-F1 over the five labels."""
    return f1_score(
        y, pred, average="macro", labels=range(5), zero_division=0
    )


def class_weight_summary():
    """Average the weighted and unweighted runs over seeds."""
    rows = []
    for condition in ["weighted", "unweighted"]:
        for model in ["late", "mlp_early"]:
            runs = []
            for seed in range(3):
                frame = pd.read_csv(
                    RERUN / condition / "preds"
                    / f"test_{model}_seed{seed}.csv"
                )
                y, pred = frame["label"].values, frame["pred"].values
                runs.append({
                    "macro_f1": macro_f1(y, pred),
                    "accuracy": (y == pred).mean(),
                    "affected_recall": recall_score(
                        y, pred, labels=[0], average="macro",
                        zero_division=0,
                    ),
                    "affected_precision": precision_score(
                        y, pred, labels=[0], average="macro",
                        zero_division=0,
                    ),
                    "missed_humanitarian": (
                        (y != OTHER) & (pred == OTHER)
                    ).sum(),
                    "false_alarms": ((y == OTHER) & (pred != OTHER)).sum(),
                })
            runs = pd.DataFrame(runs)
            row = {"model": model, "condition": condition}
            row.update(runs.mean().to_dict())
            row["macro_f1_std"] = runs["macro_f1"].std()
            rows.append(row)
    return pd.DataFrame(rows)


def logistic_rerun():
    """Refit the logistic baselines with the C values of the main run."""
    manifest = pd.read_csv(DEFAULT_MANIFEST, dtype=str)
    image_map, image_feat = load_index(
        DEFAULT_CACHE / "image_b0.npz", "image_id"
    )
    text_map, text_feat = load_index(
        DEFAULT_CACHE / "text_bert.npz", "tweet_id"
    )
    data = {
        name: arrays(
            manifest[manifest.split == name].reset_index(drop=True),
            image_map, image_feat, text_map, text_feat,
        )
        for name in ["train", "dev", "test"]
    }
    probs = {}
    for index, (name, c_value) in enumerate(FIXED_C.items()):
        scaler = StandardScaler().fit(data["train"][index])
        model = LogisticRegression(
            C=c_value, max_iter=3000, class_weight="balanced"
        ).fit(scaler.transform(data["train"][index]), data["train"][3])
        for split in data:
            probs[name, split] = model.predict_proba(
                scaler.transform(data[split][index])
            )
    for split in data:
        weight = IMAGE_WEIGHT * (1 - data[split][4])[:, None]
        probs["late", split] = (
            weight * probs["image", split]
            + (1 - weight) * probs["text", split]
        )
    gaps = []
    for name in ["text", "image", "early", "late"]:
        score = {
            split: macro_f1(data[split][3], probs[name, split].argmax(1))
            for split in data
        }
        gaps.append({
            "model": f"logreg_{name}",
            "train_macro_f1": score["train"],
            "dev_macro_f1": score["dev"],
            "test_macro_f1": score["test"],
            "gap": score["train"] - score["test"],
        })
    return pd.DataFrame(gaps), data["test"][3], {
        name: probs[name, "test"] for name in ["image", "text", "late"]
    }


def error_tables(y, probs):
    """Cross errors of late fusion with single models and confidence."""
    image_ok = probs["image"].argmax(1) == y
    text_ok = probs["text"].argmax(1) == y
    pred = probs["late"].argmax(1)
    late_ok = pred == y
    modalities = pd.crosstab(
        [pd.Series(image_ok, name="image_ok"),
         pd.Series(text_ok, name="text_ok")],
        pd.Series(late_ok, name="late_ok"),
    ).reset_index()
    confidence = probs["late"].max(1)
    bands = pd.cut(
        confidence, [0, 0.5, 0.8, 1.0],
        labels=["below 0.5", "0.5 to 0.8", "above 0.8"],
    )
    by_confidence = (
        pd.DataFrame({"confidence": bands, "error": ~late_ok})
        .groupby("confidence", observed=True)["error"]
        .agg(n="size", errors="sum")
        .reset_index()
    )
    kinds = pd.DataFrame([{
        "errors": (~late_ok).sum(),
        "missed_humanitarian": ((y != OTHER) & (pred == OTHER)).sum(),
        "false_alarms": ((y == OTHER) & (pred != OTHER)).sum(),
    }])
    return modalities, by_confidence, kinds


def main():
    """Write the four summary tables."""
    REPORT.mkdir(parents=True, exist_ok=True)
    weights = class_weight_summary()
    weights.to_csv(REPORT / "class_weight_summary.csv", index=False)
    gaps, y, probs = logistic_rerun()
    gaps.to_csv(REPORT / "logreg_gap.csv", index=False)
    modalities, by_confidence, kinds = error_tables(y, probs)
    modalities.to_csv(REPORT / "error_modalities.csv", index=False)
    by_confidence.to_csv(REPORT / "error_confidence.csv", index=False)
    for table in [weights, gaps, modalities, by_confidence, kinds]:
        print(table.round(3).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
