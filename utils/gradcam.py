"""Grad-CAM 可视化工具。

基于 pytorch-grad-cam 封装：拿到目标层（ResNet-18 的 layer4[-1]）、
生成热力图叠加图，并按"输入图 + Grad-CAM"两联图的形式保存。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data.dataset import IMAGENET_MEAN, IMAGENET_STD


def target_layer(model):
    """ResNet-18 最后一个卷积层（AvgPool 之前，输出 512×7×7）。"""
    return model.layer4[-1]


def denormalize(image_tensor) -> np.ndarray:
    """把 Normalize 之后的 [1,3,H,W] 张量还原为 [0,1] 的 RGB 图像。"""
    image = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    image = np.array(IMAGENET_STD) * image + np.array(IMAGENET_MEAN)
    return np.clip(image, 0, 1)


class GradCAMVisualizer:
    """封装 Grad-CAM 的调用与出图。"""

    def __init__(self, model):
        self.model = model
        self.cam = GradCAM(model=model, target_layers=[target_layer(model)])

    def overlay(self, image_tensor, target_class: int) -> np.ndarray:
        """返回叠加了热力图的 RGB 图像（numpy uint8）。"""
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]
        return show_cam_on_image(denormalize(image_tensor), grayscale_cam, use_rgb=True)

    def save_figure(self, record: dict, class_names, path: str) -> None:
        """保存"输入图 | Grad-CAM"两联图。

        record 需要包含 image（[1,3,224,224] 张量）、true、pred、conf。
        """
        image = record["image"]
        overlay = self.overlay(image, record["pred"])

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        axes[0].imshow(denormalize(image))
        axes[0].set_title("Input image", fontsize=11)
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM overlay", fontsize=11)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"True: {class_names[record['true']]}    "
            f"Pred: {class_names[record['pred']]}    "
            f"confidence: {record['conf'] * 100:.1f}%",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
