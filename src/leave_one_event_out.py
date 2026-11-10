"""Leave-one-event-out evaluation with logistic fusion baselines."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baseline import arrays, fit_best
from dataset import load_index

FINAL = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = FINAL / "data" / "manifest_agreed.csv"
DEFAULT_CACHE = FINAL / "cache"
DEFAULT_OUT = FINAL / "results"
REPORT = FINAL / "results" / "report"


def evaluate(y, pred):
    """Return accuracy, fixed-label F1 and present-label F1."""
    labels = np.unique(y)
    fixed = f1_score(
        y, pred, average="macro", labels=range(5), zero_division=0
    )
    present = f1_score(
        y, pred, average="macro", labels=labels, zero_division=0
    )
    return accuracy_score(y, pred), fixed, present


def bootstrap_ci(y, pred, n=2000, seed=0):
    """Bootstrap interval for fixed-label macro-F1."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    values = np.array([
        f1_score(
            y[i], pred[i], average="macro", labels=range(5),
            zero_division=0,
        )
        for i in idx
    ])
    return np.percentile(values, [2.5, 97.5])


def bootstrap_delta(y, pred_a, pred_b, n=2000, seed=0):
    """Bootstrap interval for the paired macro-F1 difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    values = np.array([
        f1_score(
            y[i], pred_a[i], average="macro", labels=range(5),
            zero_division=0,
        )
        - f1_score(
            y[i], pred_b[i], average="macro", labels=range(5),
            zero_division=0,
        )
        for i in idx
    ])
    return np.percentile(values, [2.5, 97.5])


def in_distribution(event, model, test_rows):
    """Return in-distribution predictions aligned to the event test rows."""
    path = DEFAULT_OUT / "preds" / f"test_{model}_seed0.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.event_name == event].set_index("image_id")
    return frame.loc[test_rows["image_id"], "pred"].values


def main():
    """Run leave-one-event-out and save predictions and intervals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
    rows = []
    predictions = []
    for event in sorted(splits["test"]["event_name"].unique()):
        train_rows = splits["train"][
            splits["train"]["event_name"] != event
        ].reset_index(drop=True)
        dev_rows = splits["dev"][
            splits["dev"]["event_name"] != event
        ].reset_index(drop=True)
        test_rows = splits["test"][
            splits["test"]["event_name"] == event
        ].reset_index(drop=True)
        if len(dev_rows) == 0 or len(test_rows) == 0:
            continue
        train = arrays(
            train_rows, image_map, image_feat, text_map, text_feat
        )
        dev = arrays(
            dev_rows, image_map, image_feat, text_map, text_feat
        )
        test = arrays(
            test_rows, image_map, image_feat, text_map, text_feat
        )
        probs = {}
        for name, index in [("image", 0), ("text", 1), ("early", 2)]:
            scaler, _, model = fit_best(
                train[index], train[3], dev[index], dev[3]
            )
            probs[name, "dev"] = model.predict_proba(
                scaler.transform(dev[index])
            )
            probs[name, "test"] = model.predict_proba(
                scaler.transform(test[index])
            )
            pred = probs[name, "test"].argmax(1)
            acc, fixed, present = evaluate(test[3], pred)
            lo, hi = bootstrap_ci(test[3], pred)
            model_name = f"logreg_{name}"
            ind_pred = in_distribution(event, model_name, test_rows)
            dlo, dhi = bootstrap_delta(test[3], pred, ind_pred)
            rows.append({
                "event": event,
                "model": model_name,
                "n_test": len(test_rows),
                "accuracy": acc,
                "macro_f1": fixed,
                "present_macro_f1": present,
                "ci_low": lo,
                "ci_high": hi,
                "delta": fixed - f1_score(
                    test[3], ind_pred, average="macro", labels=range(5),
                    zero_division=0,
                ),
                "delta_low": dlo,
                "delta_high": dhi,
            })
            predictions.append(pd.DataFrame({
                "event": event,
                "model": model_name,
                "image_id": test_rows["image_id"].values,
                "label": test[3],
                "pred": pred,
            }))
        best_w = max(
            np.linspace(0, 1, 11),
            key=lambda w: f1_score(
                dev[3],
                (
                    w * (1 - dev[4])[:, None]
                    * probs["image", "dev"]
                    + (1 - w * (1 - dev[4])[:, None])
                    * probs["text", "dev"]
                ).argmax(1),
                average="macro", labels=range(5), zero_division=0,
            ),
        )
        weight = best_w * (1 - test[4])[:, None]
        late = (
            weight * probs["image", "test"]
            + (1 - weight) * probs["text", "test"]
        )
        pred = late.argmax(1)
        acc, fixed, present = evaluate(test[3], pred)
        lo, hi = bootstrap_ci(test[3], pred)
        ind_pred = in_distribution(event, "logreg_late", test_rows)
        dlo, dhi = bootstrap_delta(test[3], pred, ind_pred)
        rows.append({
            "event": event,
            "model": "logreg_late",
            "n_test": len(test_rows),
            "accuracy": acc,
            "macro_f1": fixed,
            "present_macro_f1": present,
            "ci_low": lo,
            "ci_high": hi,
            "delta": fixed - f1_score(
                test[3], ind_pred, average="macro", labels=range(5),
                zero_division=0,
            ),
            "delta_low": dlo,
            "delta_high": dhi,
        })
        predictions.append(pd.DataFrame({
            "event": event,
            "model": "logreg_late",
            "image_id": test_rows["image_id"].values,
            "label": test[3],
            "pred": pred,
        }))
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "leave_one_event_out.csv", index=False)
    table.to_csv(REPORT / "loeo_adjusted.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        REPORT / "loeo_preds.csv", index=False
    )

    summary = []
    for model, group in table.groupby("model"):
        weights = group["n_test"] / group["n_test"].sum()
        summary.append({
            "model": model,
            "unweighted": group["macro_f1"].mean(),
            "weighted": (group["macro_f1"] * weights).sum(),
            "trimmed": group["macro_f1"].sort_values().iloc[
                1:-1
            ].mean(),
            "delta_unweighted": group["delta"].mean(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(REPORT / "loeo_summary.csv", index=False)
    print(table.round(3).to_string(index=False))
    print("\nsummary")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
