"""Preliminary comparison: unimodal vs early vs late fusion, logistic regression.

Usage: python classify.py SPLITS_DIR CACHE_DIR
Uses the agreed-label splits of the informative and humanitarian tasks.
Hyperparameters and late fusion weights are chosen on dev; test is used once.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SPLITS = Path(sys.argv[1]) / "crisismmd_datasplit_agreed_label"
CACHE = Path(sys.argv[2])
C_GRID = [0.001, 0.01, 0.1, 1.0]
RNG = np.random.default_rng(0)


def load_features():
    """Image features keyed by image_id and text features keyed by tweet_id."""
    img = np.load(CACHE / "image_b0.npz", allow_pickle=True)
    txt = np.load(CACHE / "text_bert.npz", allow_pickle=True)
    img_map = dict(zip(img["image_id"], range(len(img["image_id"]))))
    txt_map = dict(zip(txt["tweet_id"], range(len(txt["tweet_id"]))))
    return (img["feat"].astype(np.float32), img_map,
            txt["feat"].astype(np.float32), txt_map)


def load_task(task, feats):
    """Feature matrices and labels for the train, dev and test splits."""
    img, img_map, txt, txt_map = feats
    out = {}
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(SPLITS / f"task_{task}_text_img_agreed_lab_{split}"
                         ".tsv", sep="\t", dtype=str, quoting=3)
        out[split] = {
            "image": img[[img_map[i] for i in df.image_id]],
            "text": txt[[txt_map[i] for i in df.tweet_id]],
            "y": df.label.values}
    return out


def fit_best(x_train, y_train, x_dev, y_dev):
    """Fit logistic regression for each C and keep the best on dev macro-F1."""
    scaler = StandardScaler().fit(x_train)
    xt, xd = scaler.transform(x_train), scaler.transform(x_dev)
    best = None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=3000,
                                   class_weight="balanced").fit(xt, y_train)
        score = f1_score(y_dev, model.predict(xd), average="macro")
        if best is None or score > best[0]:
            best = (score, c, model)
    return scaler, best[2], best[1]


def scores(y, pred):
    """Accuracy, macro-F1 and weighted F1."""
    return (accuracy_score(y, pred), f1_score(y, pred, average="macro"),
            f1_score(y, pred, average="weighted"))


def bootstrap_diff(y, pred_a, pred_b, n=1000):
    """95% interval of macro-F1(a) minus macro-F1(b) over test resamples."""
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    diffs = [f1_score(y[i], pred_a[i], average="macro")
             - f1_score(y[i], pred_b[i], average="macro") for i in idx]
    return np.percentile(diffs, [2.5, 97.5])


def run(task, feats):
    """Train the four models on one task and print the comparison."""
    d = load_task(task, feats)
    for s in d.values():
        s["early"] = np.hstack([s["image"], s["text"]])
    rows, preds, probs = [], {}, {}
    for name in ["image", "text", "early"]:
        scaler, model, c = fit_best(d["train"][name], d["train"]["y"],
                                    d["dev"][name], d["dev"]["y"])
        for split in ["train", "dev", "test"]:
            x = scaler.transform(d[split][name])
            probs[name, split] = model.predict_proba(x)
            preds[name, split] = model.predict(x)
        classes = model.classes_
        rows.append([name, c])

    # late fusion: weighted mean of unimodal probabilities, weight on dev
    best_w = max(np.linspace(0, 1, 11), key=lambda w: f1_score(
        d["dev"]["y"], classes[(w * probs["image", "dev"] + (1 - w)
                                * probs["text", "dev"]).argmax(1)],
        average="macro"))
    for split in ["train", "dev", "test"]:
        p = best_w * probs["image", split] + (1 - best_w) * probs[
            "text", split]
        preds["late", split] = classes[p.argmax(1)]
    rows.append(["late", f"w_image={best_w:.1f}"])

    table = []
    for name, setting in rows:
        acc, mf1, wf1 = scores(d["test"]["y"], preds[name, "test"])
        tr = f1_score(d["train"]["y"], preds[name, "train"],
                      average="macro")
        dv = f1_score(d["dev"]["y"], preds[name, "dev"], average="macro")
        table.append([name, setting, round(tr, 3), round(dv, 3),
                      round(acc, 3), round(mf1, 3), round(wf1, 3)])
    print(f"\n=== {task} (agreed labels) train {len(d['train']['y'])}, "
          f"dev {len(d['dev']['y'])}, test {len(d['test']['y'])} ===")
    print(pd.DataFrame(table, columns=[
        "model", "setting", "train mF1", "dev mF1", "test acc",
        "test mF1", "test wF1"]).to_string(index=False))

    y = d["test"]["y"]
    uni = max(["image", "text"], key=lambda m: f1_score(
        d["dev"]["y"], preds[m, "dev"], average="macro"))
    for fused in ["early", "late"]:
        lo, hi = bootstrap_diff(y, preds[fused, "test"], preds[uni, "test"])
        print(f"{fused} minus {uni} test macro-F1, 95% bootstrap interval "
              f"[{lo:.3f}, {hi:.3f}]")
    print("\nper-class test F1")
    per = {m: f1_score(y, preds[m, "test"], average=None, labels=classes)
           for m in ["image", "text", "early", "late"]}
    print(pd.DataFrame(per, index=classes).round(3).to_string())


feats = load_features()
for task in ["informative", "humanitarian"]:
    run(task, feats)
