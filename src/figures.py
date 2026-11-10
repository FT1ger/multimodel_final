"""Generate aggregate report figures from saved predictions and metrics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score

FINAL = Path(__file__).resolve().parents[1]
RESULTS = FINAL / "results_std"
R50 = FINAL / "results_r50_std"
OUT = FINAL / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#898781"
GRID = "#e1e0d9"
TEXT = "#0b0b0b"
SECONDARY = "#52514e"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"],
)

MODELS = [
    "logreg_image", "logreg_text", "logreg_early", "logreg_late",
    "mlp_image", "mlp_text", "mlp_early", "late", "token_late", "cross",
    "r50_mlp_image", "r50_mlp_early", "r50_late",
]
NAMES = {
    "logreg_image": "Image only (LR)",
    "logreg_text": "Text only (LR)",
    "logreg_early": "Early fusion (LR)",
    "logreg_late": "Late fusion (LR)",
    "mlp_image": "Image only (MLP, B0)",
    "mlp_text": "Text only (MLP)",
    "mlp_early": "Early fusion (MLP, B0)",
    "late": "Late fusion (MLP, B0)",
    "token_late": "Late fusion (tokens, B0)",
    "cross": "Cross-attention (B0)",
    "r50_mlp_image": "Image only (MLP, R50)",
    "r50_mlp_early": "Early fusion (MLP, R50)",
    "r50_late": "Late fusion (MLP, R50)",
}
CLASSES = [
    "Affected", "Infrastructure", "Rescue", "Other", "Not human.",
]
EVENTS = [
    "hurricane_harvey", "hurricane_maria", "hurricane_irma",
    "srilanka_floods", "mexico_earthquake", "california_wildfires",
    "iraq_iran_earthquake",
]
EVENT_NAMES = {
    "hurricane_harvey": "Harvey",
    "hurricane_maria": "Maria",
    "hurricane_irma": "Irma",
    "srilanka_floods": "Sri Lanka",
    "mexico_earthquake": "Mexico",
    "california_wildfires": "California",
    "iraq_iran_earthquake": "Iraq-Iran",
}
SEEDS = [0, 1, 2]


def setup():
    """Set a restrained report style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TEXT,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "text.color": TEXT,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_pred(model, seed):
    """Load a prediction file, using seed 0 for logistic baselines."""
    if model.startswith("r50_"):
        folder = R50 / "preds"
        stem = model.replace("r50_", "")
        path = folder / f"test_{stem}_seed{seed}.csv"
    else:
        path = RESULTS / "preds" / f"test_{model}_seed{seed}.csv"
    if not path.exists() and model.startswith("logreg"):
        path = RESULTS / "preds" / f"test_{model}_seed0.csv"
    return pd.read_csv(path)


def model_summary():
    """Return mean and standard deviation of test macro-F1."""
    runs = pd.read_csv(RESULTS / "metrics_runs.csv")
    base = pd.read_csv(RESULTS / "metrics_baseline.csv")
    r50 = pd.read_csv(R50 / "metrics_runs.csv")
    r50["model"] = "r50_" + r50["model"]
    runs = pd.concat([runs, r50], ignore_index=True)
    rows = []
    for model in MODELS:
        if model.startswith("logreg"):
            value = base.loc[
                base.model == model, "test_macro_f1"
            ].iloc[0]
            rows.append((model, value, 0.0))
        else:
            sub = runs.loc[runs.model == model, "test_macro_f1"]
            rows.append((model, sub.mean(), sub.std()))
    return pd.DataFrame(rows, columns=["model", "mean", "std"])


def figure_model_comparison():
    """Dot plot of test macro-F1 with seed spread."""
    data = model_summary().sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for yi, (_, row) in enumerate(data.iterrows()):
        color = GRAY if row["model"].startswith("logreg") else BLUE
        ax.errorbar(
            row["mean"], yi, xerr=row["std"], fmt="o",
            color=color, ms=6, elinewidth=0.9, capsize=3,
        )
        ax.text(
            row["mean"] + row["std"] + 0.008, yi,
            f"{row['mean']:.3f}", va="center", fontsize=7.5,
            color=TEXT,
        )
    ax.axvline(0.80, color="#c3c2b7", lw=0.9, zorder=0)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([NAMES[m] for m in data.model])
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlim(0.60, 0.86)
    ax.set_xlabel("Test macro-F1")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color=GRAY, lw=0, label="Logistic"),
            plt.Line2D([], [], marker="o", color=BLUE, lw=0, label="Neural"),
        ],
        loc="lower right", frameon=False,
    )
    fig.savefig(OUT / "model_comparison.pdf")
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)


def figure_training_curves():
    """Dev macro-F1 and the cross-attention loss gap."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    ax = axes[0]
    colors = {"mlp_early": BLUE, "late": ORANGE, "cross": AQUA}
    for model, color in colors.items():
        frames = [
            pd.read_csv(RESULTS / f"history_{model}_seed{s}.csv")
            for s in SEEDS
        ]
        hist = pd.concat(frames, ignore_index=True)
        agg = hist.groupby("epoch")["dev_macro_f1"].agg(
            ["mean", "std"]
        ).reset_index()
        ax.plot(
            agg["epoch"], agg["mean"], color=color, lw=1.6,
            marker="o", ms=3, label=NAMES[model],
        )
        ax.fill_between(
            agg["epoch"], agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            color=color, alpha=0.12, linewidth=0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0.45, 0.85)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    hist = pd.read_csv(RESULTS / "history_cross_seed0.csv")
    ax.plot(
        hist["epoch"], hist["train_loss"], color=BLUE, lw=1.6,
        marker="o", ms=3, label="Train loss",
    )
    ax.plot(
        hist["epoch"], hist["dev_loss"], color=ORANGE, lw=1.6,
        marker="o", ms=3, label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.pdf")
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)


def per_class_matrix():
    """Average per-class F1 for each model across seeds."""
    values = []
    for model in MODELS:
        scores = []
        for seed in SEEDS:
            df = load_pred(model, seed)
            scores.append(
                f1_score(
                    df["label"], df["pred"], average=None,
                    labels=range(5), zero_division=0,
                )
            )
        values.append(np.mean(scores, axis=0))
    return np.array(values)


def figure_per_class():
    """Heatmap of per-class F1."""
    values = per_class_matrix()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    image = ax.imshow(values, cmap=SEQ, vmin=0.2, vmax=0.95, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([NAMES[m] for m in MODELS])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.7 else TEXT
            ax.text(
                j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color=color,
            )
    ax.set_xlabel("Humanitarian class")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "per_class_f1.pdf")
    fig.savefig(OUT / "per_class_f1.png")
    plt.close(fig)


def figure_per_event():
    """Heatmap of per-event macro-F1."""
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            frame = load_pred(model, seed)
            for event, group in frame.groupby("event_name"):
                score = f1_score(
                    group["label"], group["pred"], average="macro",
                    labels=range(5), zero_division=0,
                )
                rows.append({
                    "model": model, "event": event, "macro_f1": score,
                })
    events = pd.DataFrame(rows)
    pivot = events.groupby(["model", "event"])["macro_f1"].mean().unstack()
    pivot = pivot.reindex(index=MODELS, columns=EVENTS)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    image = ax.imshow(
        pivot.values, cmap=SEQ, vmin=0.3, vmax=0.9, aspect="auto"
    )
    ax.set_xticks(range(len(EVENTS)))
    ax.set_xticklabels([EVENT_NAMES[e] for e in EVENTS], rotation=20,
                       ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([NAMES[m] for m in MODELS])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            color = "white" if value > 0.65 else TEXT
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=7, color=color,
            )
    ax.set_xlabel("Disaster event")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Macro-F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "per_event_f1.pdf")
    fig.savefig(OUT / "per_event_f1.png")
    plt.close(fig)


def figure_confusion():
    """Confusion matrix for the best logistic late-fusion model."""
    df = load_pred("logreg_late", 0)
    matrix = confusion_matrix(
        df["label"], df["pred"], labels=range(5)
    )
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    image = ax.imshow(matrix, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(5):
        for j in range(5):
            color = "white" if matrix[i, j] > matrix.max() * 0.55 else TEXT
            ax.text(
                j, i, str(matrix[i, j]), ha="center", va="center",
                fontsize=7.5, color=color,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Images", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "confusion_matrix.pdf")
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)


def figure_leave_one_event_out():
    """Heatmap of leave-one-event-out macro-F1 over present classes."""
    table = pd.read_csv(
        FINAL / "results" / "leave_one_event_out.csv"
    )
    models = [
        "logreg_image", "logreg_text", "logreg_early", "logreg_late"
    ]
    pivot = table.pivot_table(
        index="model", columns="event", values="present_macro_f1"
    ).reindex(index=models, columns=EVENTS)
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    image = ax.imshow(
        pivot.values, cmap=SEQ, vmin=0.35, vmax=0.95, aspect="auto"
    )
    ax.set_xticks(range(len(EVENTS)))
    ax.set_xticklabels([EVENT_NAMES[e] for e in EVENTS], rotation=20,
                       ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([NAMES[m] for m in models])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            color = "white" if value > 0.6 else TEXT
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=7, color=color,
            )
    ax.set_xlabel("Held-out event")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Macro-F1", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "leave_one_event_out.pdf")
    fig.savefig(OUT / "leave_one_event_out.png")
    plt.close(fig)


def figure_pipeline():
    """Schematic of the text-image fusion pipeline."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, color, size=7.5):
        patch = plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8, edgecolor="#c3c2b7", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=TEXT,
        )

    box(0.01, 0.66, 0.15, 0.16, "Tweet text", "#e8f1fd")
    box(0.20, 0.66, 0.17, 0.16, "BERT", "#cde2fb")
    box(0.41, 0.66, 0.17, 0.16, "Text features", "#cde2fb")
    box(0.01, 0.18, 0.15, 0.16, "Image", "#fde8df")
    box(0.20, 0.18, 0.17, 0.16, "EfficientNet-B0\nor ResNet-50", "#fbd5c7")
    box(0.41, 0.18, 0.17, 0.16, "Image features", "#fbd5c7")
    box(
        0.64, 0.40, 0.18, 0.20,
        "Fusion\nearly / late /\ncross-attention", "#d9f2e7",
    )
    box(0.86, 0.40, 0.13, 0.20, "5-class\noutput", "#e8f1fd")

    for xy in [
        ((0.16, 0.74), (0.20, 0.74)),
        ((0.37, 0.74), (0.41, 0.74)),
        ((0.16, 0.26), (0.20, 0.26)),
        ((0.37, 0.26), (0.41, 0.26)),
        ((0.58, 0.74), (0.64, 0.54)),
        ((0.58, 0.26), (0.64, 0.46)),
        ((0.82, 0.50), (0.86, 0.50)),
    ]:
        ax.annotate(
            "", xy=xy[1], xytext=xy[0],
            arrowprops=dict(arrowstyle="-|>", lw=0.8,
                            color="#898781", shrinkA=0, shrinkB=0),
        )
    fig.savefig(OUT / "pipeline.pdf")
    fig.savefig(OUT / "pipeline.png")
    plt.close(fig)


def main():
    """Generate all report figures."""
    setup()
    figure_pipeline()
    figure_model_comparison()
    figure_training_curves()
    figure_per_class()
    figure_per_event()
    figure_confusion()
    figure_leave_one_event_out()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
