"""Oxford-IIIT Pet：数据加载、分层划分与 Transform。

要点：
1. 合并官方 trainval(3680) + test(3669) = 7349 张，按 70:15:15 做分层抽样；
2. 训练集与验证/测试集使用互相独立的 transform 实例，验证集只做
   Resize + CenterCrop，不含任何随机增强（否则评估结果不稳定、且属于数据泄漏）；
3. 标签由 torchvision 统一转成 0~36，直接对得上 nn.CrossEntropyLoss。
"""
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# 类别数（37）定义在 models/model.py 的 NUM_CLASSES，避免两处各写一份


def build_train_transform() -> transforms.Compose:
    """训练集：Resize -> RandomCrop -> RandomHorizontalFlip -> ToTensor -> Normalize."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    """验证/测试集：只有确定性的 Resize + CenterCrop，绝不加随机增强。"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TransformSubset(Dataset):
    """带独立 transform 的 Subset。

    直接用 Subset 再改 ``subset.dataset.transform`` 会改到共享的底层数据集上，
    导致验证/测试集也套用训练集的随机增强；这里让每个 split 各自持有
    transform，从根上避免该问题。
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int], transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[int(self.indices[index])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class _PILPassthrough:
    """占位 transform：保持 PIL 图像原样，真正的预处理在 TransformSubset 里做。"""

    def __call__(self, image):
        return image


def _load_parts(data_dir: str, download: bool = False):
    placeholder = _PILPassthrough()
    trainval = datasets.OxfordIIITPet(
        root=data_dir, split="trainval", download=download, transform=placeholder
    )
    test = datasets.OxfordIIITPet(
        root=data_dir, split="test", download=download, transform=placeholder
    )
    return trainval, test


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    test_transform=None,
):
    """返回 (train_loader, val_loader, test_loader)。

    全部 7349 张按 70 : 15 : 15 分层抽样；随机种子固定为 seed。
    ``test_transform`` 仅用于复现"评估集误用随机增强"这一 Bug 的对照实验，
    正式评估时保持默认 None（CenterCrop 确定性变换）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("正在加载数据集（7349 张，首次使用需先下载/解压到 data/oxford-iiit-pet）...")
    trainval, test_part = _load_parts(data_dir)
    full_dataset = ConcatDataset([trainval, test_part])

    # torchvision 已做 label-1，这里是 0~36；索引顺序与 _images 一致。
    targets = np.array(list(trainval._labels) + list(test_part._labels))
    indices = np.arange(len(targets))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=seed
    )

    train_dataset = TransformSubset(full_dataset, train_idx, build_train_transform())
    val_dataset = TransformSubset(full_dataset, val_idx, build_eval_transform())
    test_dataset = TransformSubset(
        full_dataset, test_idx, test_transform or build_eval_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(
        f"数据划分完成：训练集 {len(train_dataset)} 张，"
        f"验证集 {len(val_dataset)} 张，测试集 {len(test_dataset)} 张"
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"检查点 1：Batch 形状 {tuple(images.shape)}（期望 [32, 3, 224, 224]）")
    print(f"检查点 2：Label 形状 {tuple(labels.shape)}，取值范围 "
          f"[{int(labels.min())}, {int(labels.max())}]（期望 [0, 36]）")
    print("检查点 3：验证集 transform =", val_loader.dataset.transform)
