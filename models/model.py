"""模型定义与 Head 替换。

任务书要求模型定义独立成模块：train.py 与 evaluate.py 都从这里取模型，
避免训练和评估各写一份、改一处漏一处。
"""
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 37  # Oxford-IIIT Pet：37 个猫、狗品种


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """构建 ResNet-18 分类模型。

    Args:
        num_classes: 分类头输出维度，Oxford-IIIT Pet 为 37。
        pretrained: 是否加载 ImageNet 预训练权重。
        freeze_backbone: 冻结除 fc 之外的全部参数（只微调分类头）。

    Returns:
        替换好分类头的 ResNet-18。
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # ResNet-18 的全局平均池化后是 512 维特征，替换最后的全连接层即可
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model


def load_checkpoint(
    ckpt_path: str,
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """加载训练好的 best_model_*.pth，返回 eval 模式的模型。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


if __name__ == "__main__":
    net = build_model(pretrained=False).eval()
    total = sum(p.numel() for p in net.parameters())
    print(f"输出类别数: {net.fc.out_features} | 参数量: {total / 1e6:.2f} M")

    # 答辩抽查第 1 题：输入 [32,3,224,224] 经过最后一个卷积层输出 [32,512,7,7]
    shapes = {}

    def record_shape(module, inputs, output):
        # 注意：hook 不能有返回值，否则会替换掉该层的输出
        shapes["layer4"] = tuple(output.shape)

    net.layer4.register_forward_hook(record_shape)
    with torch.no_grad():
        logits = net(torch.randn(2, 3, 224, 224))
    print(f"最后一个卷积层输出形状（AvgPool 之前）: {shapes['layer4']}")
    print(f"分类头输出形状: {tuple(logits.shape)}")
