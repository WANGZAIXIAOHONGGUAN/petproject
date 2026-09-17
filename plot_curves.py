"""把各组实验的 logs/<exp_name>/history.csv 画成训练曲线（报告用图）。

用法：python plot_curves.py --runs baseline label_smoothing --out results/curves.png
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(exp_name: str):
    path = os.path.join("logs", exp_name, "history.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [int(r["epoch"]) for r in rows], {
        key: [float(r[key]) for r in rows]
        for key in ("train_loss", "val_loss", "train_acc", "val_acc")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["baseline", "label_smoothing"])
    parser.add_argument("--out", type=str, default="results/curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    for exp_name in args.runs:
        epochs, history = load_history(exp_name)
        label = exp_name.replace("_", " ")
        axes[0].plot(epochs, history["train_loss"], label=label, lw=1.8)
        axes[1].plot(epochs, history["val_loss"], label=label, lw=1.8)
        axes[2].plot(epochs, [v * 100 for v in history["val_acc"]], label=label, lw=1.8)
        best_epoch = epochs[max(range(len(epochs)), key=lambda i: history["val_acc"][i])]
        print(
            f"{exp_name}: best val acc {max(history['val_acc']):.4f} @epoch {best_epoch}, "
            f"final val acc {history['val_acc'][-1]:.4f}"
        )

    titles = ["(a) Train loss", "(b) Validation loss", "(c) Validation Top-1 accuracy (%)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[2].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
