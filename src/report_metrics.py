"""Compute the additional aggregate tables used in the revised report."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score

FINAL = Path(__file__).resolve().parents[1]
RESULTS = FINAL / "results_std"
R50 = FINAL / "results_r50_std"
OUT = FINAL / "results" / "report"
LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian",
]
MODELS = [
    "logreg_image", "logreg_text", "logreg_early", "logreg_late",
    "mlp_image", "mlp_text", "mlp_early", "late", "token_late", "cross",
    "r50_mlp_image", "r50_mlp_early", "r50_late",
]


def load_pred(model, seed):
    """Load test predictions for a main or R50 model."""
    if model.startswith("r50_"):
        stem = model.replace("r50_", "")
        path = R50 / "preds" / f"test_{stem}_seed{seed}.csv"
    else:
        path = RESULTS / "preds" / f"test_{model}_seed{seed}.csv"
    if not path.exists() and model.startswith("logreg"):
        path = RESULTS / "preds" / f"test_{model}_seed0.csv"
    return pd.read_csv(path)


def macro_f1(y, pred):
    """Macro-F1 over the five fixed labels."""
    return f1_score(
        y, pred, average="macro", labels=range(5), zero_division=0
    )


def bootstrap_diff(y, pred_a, pred_b, n=2000, seed=0):
    """Percentile interval for macro-F1(a) minus macro-F1(b)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = np.array([
        macro_f1(y[i], pred_a[i]) - macro_f1(y[i], pred_b[i])
        for i in idx
    ])
    return np.percentile(diffs, [2.5, 97.5])


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar p-value on paired predictions."""
    correct_a = pred_a == y
    correct_b = pred_b == y
    n01 = int(((~correct_a) & correct_b).sum())
    n10 = int((correct_a & (~correct_b)).sum())
    total = n01 + n10
    if total == 0:
        return 1.0
    return float(binomtest(min(n01, n10), total, 0.5).pvalue)


def pairwise_extra():
    """Compare selected models with bootstrap and McNemar."""
    pairs = [
        ("logreg_late", "logreg_image"),
        ("logreg_late", "mlp_image"),
        ("logreg_late", "mlp_text"),
        ("logreg_late", "mlp_early"),
        ("logreg_late", "late"),
        ("logreg_late", "token_late"),
        ("logreg_late", "cross"),
        ("mlp_early", "mlp_image"),
        ("mlp_early", "mlp_text"),
        ("late", "token_late"),
        ("late", "cross"),
        ("cross", "mlp_early"),
        ("r50_mlp_early", "r50_mlp_image"),
        ("r50_mlp_early", "r50_late"),
        ("r50_mlp_early", "mlp_early"),
        ("r50_mlp_early", "logreg_late"),
        ("r50_late", "r50_mlp_image"),
        ("r50_late", "logreg_late"),
        ("r50_mlp_image", "logreg_image"),
        ("mlp_early", "r50_mlp_image"),
        ("logreg_late", "r50_mlp_image"),
    ]
    rows = []
    for model_a, model_b in pairs:
        for seed in [0, 1, 2]:
            a = load_pred(model_a, seed)
            b = load_pred(model_b, seed)
            y = a["label"].values
            pa = a["pred"].values
            pb = b["pred"].values
            lo, hi = bootstrap_diff(y, pa, pb, seed=seed)
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "seed": seed,
                "macro_f1_a": macro_f1(y, pa),
                "macro_f1_b": macro_f1(y, pb),
                "diff": macro_f1(y, pa) - macro_f1(y, pb),
                "ci_low": lo,
                "ci_high": hi,
                "mcnemar_p": mcnemar(y, pa, pb),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "pairwise_extra.csv", index=False)
    return table


def affected_metrics():
    """Report affected-class metrics for every model and seed."""
    rows = []
    for model in MODELS:
        for seed in [0, 1, 2]:
            try:
                frame = load_pred(model, seed)
            except FileNotFoundError:
                continue
            y = frame["label"].values
            pred = frame["pred"].values
            rows.append({
                "model": model,
                "seed": seed,
                "recall": recall_score(
                    y, pred, labels=[0], average=None, zero_division=0
                )[0],
                "precision": precision_score(
                    y, pred, labels=[0], average=None, zero_division=0
                )[0],
                "f1": f1_score(
                    y, pred, labels=[0], average=None, zero_division=0
                )[0],
                "support": int((y == 0).sum()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "affected_all.csv", index=False)
    table.to_csv(OUT / "affected_class.csv", index=False)
    return table.groupby("model")[["recall", "precision", "f1"]].mean()


def affected_scores(y, pred):
    """Return affected-class recall, precision and F1."""
    tp = float(((y == 0) & (pred == 0)).sum())
    fp = float(((y != 0) & (pred == 0)).sum())
    fn = float(((y == 0) & (pred != 0)).sum())
    recall = tp / max(tp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
    return recall, precision, f1


def affected_ci(n_boot=2000):
    """Bootstrap intervals for affected-class metrics, per model and seed."""
    rows = []
    for model in MODELS:
        for seed in [0, 1, 2]:
            frame = load_pred(model, seed)
            y = frame["label"].values
            pred = frame["pred"].values
            point = affected_scores(y, pred)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(y), size=(n_boot, len(y)))
            yb = y[idx]
            pb = pred[idx]
            tp = ((yb == 0) & (pb == 0)).sum(axis=1)
            fp = ((yb != 0) & (pb == 0)).sum(axis=1)
            fn = ((yb == 0) & (pb != 0)).sum(axis=1)
            values = np.column_stack([
                tp / np.maximum(tp + fn, 1),
                tp / np.maximum(tp + fp, 1),
                2 * tp / np.maximum(2 * tp + fp + fn, 1),
            ])
            low, high = np.percentile(values, [2.5, 97.5], axis=0)
            for metric, p, lo, hi in zip(
                ["recall", "precision", "f1"], point, low, high
            ):
                rows.append({
                    "model": model,
                    "seed": seed,
                    "metric": metric,
                    "value": p,
                    "ci_low": lo,
                    "ci_high": hi,
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "affected_ci.csv", index=False)
    return table


def hierarchical_diff(model_a, model_b, n_boot=2000, seed=0):
    """Bootstrap seeds and test rows, then average the seed-level diff."""
    frames = {
        s: (load_pred(model_a, s), load_pred(model_b, s))
        for s in [0, 1, 2]
    }
    rng = np.random.default_rng(seed)
    y0 = frames[0][0]["label"].values
    values = []
    for _ in range(n_boot):
        chosen = rng.integers(0, 3, size=3)
        idx = rng.integers(0, len(y0), size=len(y0))
        diffs = []
        for s in chosen:
            a, b = frames[int(s)]
            diffs.append(
                macro_f1(y0[idx], a["pred"].values[idx])
                - macro_f1(y0[idx], b["pred"].values[idx])
            )
        values.append(np.mean(diffs))
    return np.percentile(values, [2.5, 97.5]), float(np.mean(values))


def hierarchical_pairwise():
    """Compare key fusion models to the best single modality."""
    pairs = [
        ("r50_late", "r50_mlp_image"),
        ("r50_mlp_early", "r50_mlp_image"),
        ("mlp_early", "r50_mlp_image"),
        ("logreg_late", "r50_mlp_image"),
        ("r50_mlp_early", "r50_late"),
    ]
    rows = []
    for model_a, model_b in pairs:
        (lo, hi), _ = hierarchical_diff(model_a, model_b)
        seed_diffs = []
        for s in [0, 1, 2]:
            a = load_pred(model_a, s)
            b = load_pred(model_b, s)
            y = a["label"].values
            seed_diffs.append(
                macro_f1(y, a["pred"].values)
                - macro_f1(y, b["pred"].values)
            )
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "mean_diff": float(np.mean(seed_diffs)),
            "ci_low": lo,
            "ci_high": hi,
            "seed_0": seed_diffs[0],
            "seed_1": seed_diffs[1],
            "seed_2": seed_diffs[2],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "hierarchical_pairwise.csv", index=False)
    return table


def affected_diff_ci(n_boot=2000):
    """Paired bootstrap intervals for affected-class F1 differences."""
    pairs = [
        ("r50_late", "logreg_late"),
        ("r50_late", "late"),
        ("r50_late", "r50_mlp_early"),
    ]
    rows = []
    for model_a, model_b in pairs:
        for seed in [0, 1, 2]:
            a = load_pred(model_a, seed)
            b = load_pred(model_b, seed)
            y = a["label"].values
            pa = a["pred"].values
            pb = b["pred"].values
            diff = affected_scores(y, pa)[2] - affected_scores(y, pb)[2]
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(y), size=(n_boot, len(y)))
            vals = []
            for i in idx:
                vals.append(
                    affected_scores(y[i], pa[i])[2]
                    - affected_scores(y[i], pb[i])[2]
                )
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "seed": seed,
                "diff": diff,
                "ci_low": np.percentile(vals, 2.5),
                "ci_high": np.percentile(vals, 97.5),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "affected_diff_ci.csv", index=False)
    return table


def absolute_ci(model="logreg_late", seed=0):
    """Bootstrap interval for the absolute macro-F1 of one model."""
    frame = load_pred(model, seed)
    y = frame["label"].values
    pred = frame["pred"].values
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(2000, len(y)))
    values = np.array([macro_f1(y[i], pred[i]) for i in idx])
    row = {
        "model": model,
        "seed": seed,
        "macro_f1": macro_f1(y, pred),
        "ci_low": np.percentile(values, 2.5),
        "ci_high": np.percentile(values, 97.5),
    }
    pd.DataFrame([row]).to_csv(
        OUT / "absolute_ci.csv", index=False
    )
    return row


def majority_baseline():
    """Return the majority-class baseline on the agreed test split."""
    manifest = pd.read_csv(
        FINAL / "data" / "manifest_agreed.csv", dtype=str
    )
    test = manifest[manifest.split == "test"]
    y = np.array([LABELS.index(x) for x in test.label])
    majority = np.bincount(y).argmax()
    pred = np.full_like(y, majority)
    row = {
        "class": LABELS[majority],
        "accuracy": float((pred == y).mean()),
        "macro_f1": macro_f1(y, pred),
    }
    pd.DataFrame([row]).to_csv(
        OUT / "majority_baseline.csv", index=False
    )
    return row


def loeo_adjusted():
    """Return the enhanced LOEO table written by the LOEO script."""
    return pd.read_csv(OUT / "loeo_adjusted.csv")


def dual_agreed():
    """Evaluate the dual-head model on the agreed test subset."""
    manifest = pd.read_csv(
        FINAL / "data" / "manifest_agreed.csv", dtype=str
    )
    test = manifest[manifest.split == "test"]
    agreed_ids = set(test.image_id)
    rows = []
    for seed in [0, 1, 2]:
        frame = pd.read_csv(
            FINAL / "results" / "preds" / f"test_dual_seed{seed}.csv"
        )
        frame = frame[frame.image_id.isin(agreed_ids)]
        text_y = frame.text_label.values
        image_y = frame.image_label.values
        text_p = frame.text_pred.values
        image_p = frame.image_pred.values
        rows.append({
            "seed": seed,
            "n": len(frame),
            "text_macro_f1": f1_score(
                text_y, text_p, average="macro", labels=range(5),
                zero_division=0,
            ),
            "image_macro_f1": f1_score(
                image_y, image_p, average="macro", labels=range(5),
                zero_division=0,
            ),
            "label_agreement": float((text_y == image_y).mean()),
            "pred_agreement": float((text_p == image_p).mean()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dual_agreed.csv", index=False)
    return table


def dual_per_class():
    """Per-class F1 and support for the two dual-head outputs."""
    rows = []
    for seed in [0, 1, 2]:
        frame = pd.read_csv(
            FINAL / "results" / "preds" / f"test_dual_seed{seed}.csv"
        )
        for side in ["text", "image"]:
            y = frame[f"{side}_label"].values
            pred = frame[f"{side}_pred"].values
            scores = f1_score(
                y, pred, average=None, labels=range(5), zero_division=0
            )
            support = np.bincount(y, minlength=5)
            for index, label in enumerate(LABELS):
                rows.append({
                    "seed": seed,
                    "head": side,
                    "class": label,
                    "support": int(support[index]),
                    "f1": scores[index],
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "dual_per_class.csv", index=False)
    return table.groupby(["head", "class"])[["support", "f1"]].mean()


def r50_gap():
    """Report R50 train-test gaps."""
    runs = pd.read_csv(R50 / "metrics_runs.csv")
    table = runs[[
        "model", "seed", "train_macro_f1", "test_macro_f1",
        "best_epoch",
    ]].copy()
    table.to_csv(OUT / "r50_gap.csv", index=False)
    return table


def controls_summary():
    """Combine capacity and shuffled-pair control results."""
    sources = [
        ("deep_b0", FINAL / "results_controls_std"),
        ("deep_r50", FINAL / "results_controls_r50_std"),
        ("shuffled_b0", FINAL / "results_shuffled_std"),
        ("shuffled_r50", FINAL / "results_shuffled_r50_std"),
    ]
    rows = []
    for control, folder in sources:
        path = folder / "metrics_runs.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table["control"] = control
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(OUT / "controls.csv", index=False)
    return table.groupby(["control", "model"]).agg(
        macro_f1=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
        accuracy=("test_accuracy", "mean"),
        parameters=("parameters", "first"),
    )


def main():
    """Write all additional aggregate tables and print summaries."""
    OUT.mkdir(parents=True, exist_ok=True)
    print("pairwise extra")
    print(pairwise_extra().round(3).to_string(index=False))
    print("\naffected class")
    print(affected_metrics())
    print("\naffected class intervals")
    print(affected_ci().round(3).to_string(index=False))
    print("\naffected F1 differences")
    print(affected_diff_ci().round(3).to_string(index=False))
    print("\nhierarchical pairwise")
    print(hierarchical_pairwise().round(3).to_string(index=False))
    print("\nabsolute CI")
    print(absolute_ci())
    print("\nmajority baseline")
    print(majority_baseline())
    print("\nLOEO adjusted")
    print(loeo_adjusted().round(3).to_string(index=False))
    print("\ndual head on agreed test")
    print(dual_agreed().round(3).to_string(index=False))
    print("\ndual head per class")
    print(dual_per_class().round(3).to_string())
    print("\nR50 gap")
    print(r50_gap().round(3).to_string(index=False))
    print("\ncontrols")
    print(controls_summary().round(3).to_string())


if __name__ == "__main__":
    main()
