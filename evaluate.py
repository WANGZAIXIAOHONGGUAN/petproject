"""评估与生成可视化结果的独立脚本。

在留出测试集上计算 Top-1 / Top-5 / Macro-F1，并输出混淆矩阵（整体 + 局部放大）
与 Grad-CAM 热力图。模型、指标、Grad-CAM 的实现分别来自 models/ 与 utils/。

用法：
    python evaluate.py --ckpt best_model_baseline.pth --name baseline
    python evaluate.py --ckpt best_model_label_smoothing.pth --name label_smoothing
对照实验（复现"评估集误用随机增强"这一 Bug 的影响）：
    python evaluate.py --ckpt best_model_baseline.pth --name random_aug \
        --random-aug-test --repeats 5
"""
import argparse
import json
import os

import torch
from torchvision.datasets import OxfordIIITPet

from data.dataset import build_train_transform, get_dataloaders
from models.model import NUM_CLASSES, load_checkpoint
from utils.gradcam import GradCAMVisualizer
from utils.metrics import (
    compute_metrics,
    per_class_report,
    plot_confusion_matrix,
    plot_confusion_zoom,
    top_confused_pairs,
)


@torch.no_grad()
def collect_logits(model, loader, device):
    """跑一遍 DataLoader，收集全部 logits 与标签。"""
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def pick_examples(model, loader, device):
    """挑一张最有把握的正确样本、一张最有把握的错误样本（Grad-CAM 用）。"""
    best_correct = {"conf": -1.0, "image": None, "true": -1, "pred": -1}
    best_wrong = {"conf": -1.0, "image": None, "true": -1, "pred": -1}
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        conf, preds = probs.max(dim=1)
        for i in range(len(labels)):
            record = {
                "image": images[i:i + 1],
                "true": int(labels[i]),
                "pred": int(preds[i]),
                "conf": float(conf[i]),
            }
            target = best_correct if record["true"] == record["pred"] else best_wrong
            if record["conf"] > target["conf"]:
                target.update(record)
    return best_correct, best_wrong


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the held-out test split")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径")
    parser.add_argument("--name", type=str, required=True, help="实验名，用于命名输出文件")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results", help="图片与指标的输出目录")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-aug-test", action="store_true",
                        help="对照实验：让测试集误用训练集的随机增强")
    parser.add_argument("--repeats", type=int, default=1,
                        help="重复评估次数（配合 --random-aug-test 观察指标波动）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    class_names = OxfordIIITPet(root=args.data_dir, download=False).classes

    model = load_checkpoint(args.ckpt, device=device)
    test_transform = build_train_transform() if args.random_aug_test else None
    _, _, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, seed=args.seed, test_transform=test_transform
    )

    repeats = max(1, args.repeats)
    runs, logits, labels = [], None, None
    for _ in range(repeats):
        logits, labels = collect_logits(model, test_loader, device)
        runs.append(compute_metrics(logits, labels))

    mean_top1 = float(sum(r["top1_acc"] for r in runs) / repeats)
    metrics = {
        "checkpoint": os.path.basename(args.ckpt),
        "test_size": int(len(labels)),
        "random_aug_test": bool(args.random_aug_test),
        "repeats": repeats,
        "per_run": runs,
        "top1_acc": mean_top1,
        "top5_acc": float(sum(r["top5_acc"] for r in runs) / repeats),
        "macro_f1": float(sum(r["macro_f1"] for r in runs) / repeats),
        "top1_std": float((sum((r["top1_acc"] - mean_top1) ** 2 for r in runs) / repeats) ** 0.5),
        "top1_min": float(min(r["top1_acc"] for r in runs)),
        "top1_max": float(max(r["top1_acc"] for r in runs)),
    }

    preds = logits.argmax(axis=1)
    report = per_class_report(labels, preds, class_names)
    cm = report["confusion_matrix"]
    metrics["median_class_recall"] = report["median_class_recall"]
    metrics["per_class_f1"] = report["per_class_f1"]
    metrics["per_class_recall"] = report["per_class_recall"]
    metrics["lowest_recall_classes"] = report["lowest_recall_classes"]
    metrics["best_classes"] = report["best_classes"]
    metrics["top_confused_pairs"] = top_confused_pairs(cm, class_names, k=5)

    plot_confusion_matrix(cm, class_names, os.path.join(args.out, f"confusion_matrix_{args.name}.png"))
    plot_confusion_matrix(
        cm, class_names, os.path.join(args.out, f"confusion_matrix_{args.name}_print.png"),
        show_labels=False,
    )
    plot_confusion_zoom(
        cm, class_names, metrics["top_confused_pairs"],
        os.path.join(args.out, f"confusion_zoom_{args.name}.png"),
    )

    visualizer = GradCAMVisualizer(model)
    correct, wrong = pick_examples(model, test_loader, device)
    if correct["image"] is not None:
        visualizer.save_figure(correct, class_names,
                               os.path.join(args.out, f"gradcam_{args.name}_correct.png"))
    if wrong["image"] is not None:
        visualizer.save_figure(wrong, class_names,
                               os.path.join(args.out, f"gradcam_{args.name}_wrong.png"))
    metrics["gradcam"] = {
        "correct": {"true": class_names[correct["true"]], "pred": class_names[correct["pred"]],
                    "confidence": round(correct["conf"], 4)},
        "wrong": {"true": class_names[wrong["true"]], "pred": class_names[wrong["pred"]],
                  "confidence": round(wrong["conf"], 4)},
    }

    with open(os.path.join(args.out, f"{args.name}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[{args.name}] Top-1 {metrics['top1_acc']:.4f} | Top-5 {metrics['top5_acc']:.4f} "
          f"| Macro-F1 {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
