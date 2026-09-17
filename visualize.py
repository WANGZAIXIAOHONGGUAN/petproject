import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data.dataset import get_dataloaders

# 设置字体，防止中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 用黑体
plt.rcParams['axes.unicode_minus'] = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    _, _, test_loader = get_dataloaders("./data", batch_size=32)
    
    # 2. 加载 Baseline 模型
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    model.load_state_dict(torch.load('best_model_baseline.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # 3. 获取类别名称（为了画混淆矩阵用的标签）
    # 直接用 torchvision 内部机制获取类别名
    from torchvision.datasets import OxfordIIITPet
    class_names = OxfordIIITPet(root="./data", download=False).classes
    
    all_preds = []
    all_labels = []
    
    # 4. 在测试集上推理
    print("正在测试集上计算预测结果...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 5. 绘制并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', colorbar=False)
    plt.title("Confusion Matrix - Baseline")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("✅ 混淆矩阵已保存为 confusion_matrix.png")
    
    # 6. Grad-CAM 可视化
    print("正在生成 Grad-CAM 热力图...")
    target_layers = [model.layer4[-1]] # ResNet18 最后一层卷积层
    
    # 从测试集中找一张预测正确的和一张预测错误的图片
    correct_img, correct_label, correct_pred = None, None, None
    wrong_img, wrong_label, wrong_pred = None, None, None
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, preds = outputs.max(1)
            for i in range(len(labels)):
                if preds[i] == labels[i] and correct_img is None:
                    correct_img = images[i:i+1]
                    correct_label = labels[i].item()
                    correct_pred = preds[i].item()
                if preds[i] != labels[i] and wrong_img is None:
                    wrong_img = images[i:i+1]
                    wrong_label = labels[i].item()
                    wrong_pred = preds[i].item()
            if correct_img is not None and wrong_img is not None:
                break

    cam = GradCAM(model=model, target_layers=target_layers)
    
    def generate_cam_image(img_tensor, target_class, filename, title):
        # 反归一化，让图片恢复原样
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target_class)])
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(visualization)
        plt.title(title)
        plt.axis('off')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存 {filename}")

    # 生成正确预测的 Grad-CAM
    generate_cam_image(correct_img, correct_pred, "gradcam_correct.png", 
                       f"Correct: True={class_names[correct_label]} | Pred={class_names[correct_pred]}")
    
    # 生成错误预测的 Grad-CAM
    generate_cam_image(wrong_img, wrong_pred, "gradcam_wrong.png", 
                       f"Wrong: True={class_names[wrong_label]} | Pred={class_names[wrong_pred]}")

if __name__ == "__main__":
    main()