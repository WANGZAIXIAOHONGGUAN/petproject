"""准确率、F1 与混淆矩阵绘制。

对外提供三类功能：
1. compute_metrics：由 logits 直接算 Top-1 / Top-5 / Macro-F1；
2. per_class_report / top_confused_pairs：逐类召回率、F1 与最易混淆的类别对；
3. plot_confusion_matrix / plot_confusion_zoom：整体混淆矩阵与局部放大图。
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score

from models.model import NUM_CLASSES


def compute_metrics(logits, labels, num_classes: int = NUM_CLASSES) -> dict:
    """由模型输出的 logits 计算 Top-1、Top-5 与 Macro-F1。"""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = logits.argmax(axis=1)
    return {
        "top1_acc": float((preds == labels).mean()),
        "top5_acc": float(
            top_k_accuracy_score(labels, logits, k=5, labels=list(range(num_classes)))
        ),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def confusion_matrix_of(labels, preds, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """返回 num_classes × num_classes 的原始计数混淆矩阵。"""
    return confusion_matrix(labels, preds, labels=list(range(num_classes)))


def per_class_report(labels, preds, class_names) -> dict:
    """逐类召回率 / F1，并挑出召回率最低的几类（报告的错误分析要用）。"""
    num_classes = len(class_names)
    cm = confusion_matrix_of(labels, preds, num_classes)
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = f1_score(labels, preds, average=None, labels=list(range(num_classes)),
                  zero_division=0)
    info = [
        {"class": class_names[i], "recall": float(recalls[i]), "f1": float(f1[i])}
        for i in range(num_classes)
    ]
    return {
        "confusion_matrix": cm,
        "median_class_recall": float(np.median(recalls)),
        "per_class_f1": {class_names[i]: float(f1[i]) for i in range(num_classes)},
        "per_class_recall": {class_names[i]: float(recalls[i]) for i in range(num_classes)},
        "lowest_recall_classes": sorted(info, key=lambda row: row["recall"])[:5],
        "best_classes": sorted(info, key=lambda row: row["f1"], reverse=True)[:5],
    }


def top_confused_pairs(cm, class_names, k: int = 5) -> list:
    """找出混淆矩阵里最容易被搞混的类别对（双向计数合并后排序）。"""
    cm = np.array(cm, dtype=np.int64).copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            count = int(cm[i, j] + cm[j, i])
            if count > 0:
                pairs.append({
                    "pair": [class_names[i], class_names[j]],
                    "count": count,
                    f"{class_names[i]}->{class_names[j]}": int(cm[i, j]),
                    f"{class_names[j]}->{class_names[i]}": int(cm[j, i]),
                })
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:k]


def plot_confusion_matrix(cm, class_names, path: str, show_labels: bool = True) -> None:
    """绘制按行归一化（召回率）的混淆矩阵。

    show_labels=False 用于报告里的小尺寸整体图（37×37 的标签在这个尺寸下不可读）。
    """
    cm = np.asarray(cm)
    normalized = np.divide(cm, np.maximum(cm.sum(axis=1, keepdims=True), 1), dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 7.8) if not show_labels else (12.5, 11))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    if show_labels:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6.5)
        ax.set_yticklabels(class_names, fontsize=6.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix (row-normalized, recall) - overall {np.trace(cm) / cm.sum():.4f}",
        fontsize=12,
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label("Recall (row-normalized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_zoom(cm, class_names, pairs, path: str, top_n: int = 4) -> None:
    """把最易混淆的若干类别单独放大，单元格标注具体样本数。"""
    classes = []
    for item in pairs[:top_n]:
        for name in item["pair"]:
            if name not in classes:
                classes.append(name)
    idx = [class_names.index(name) for name in classes]
    sub = np.asarray(cm)[np.ix_(idx, idx)]
    normalized = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)

    labels = ["\n".join(textwrap.wrap(name, 15)) for name in classes]
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title("Local confusion matrix (cell = number of test images, color = row-normalized)",
                 fontsize=9.5)
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = int(sub[i, j])
            if value == 0:
                continue
            ax.text(j, i, str(value), ha="center", va="center", fontsize=9,
                    color="white" if normalized[i, j] > 0.55 else "#10233A")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label("Recall", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
