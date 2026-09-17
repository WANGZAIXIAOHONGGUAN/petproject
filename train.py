import argparse
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # ⚠️ 梯度清零
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
        
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval() # ⚠️ 评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # ⚠️ 关闭梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    # 1. 新增：命令行参数解析
    parser = argparse.ArgumentParser(description="Pet Classification Training")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'ls'], help='ce: 标准CrossEntropy, ls: LabelSmoothing')
    parser.add_argument('--exp_name', type=str, default='baseline', help='实验名称，用于区分日志和模型')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，所有对比实验保持一致')
    parser.add_argument('--out', type=str, default='logs', help='TensorBoard 日志输出目录')
    parser.add_argument('--download', action=argparse.BooleanOptionalAction, default=True,
                        help='数据集缺失时是否自动下载（默认开启；本地已有数据则直接使用）')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | 实验名称: {args.exp_name} | 损失函数: {args.loss}")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.batch_size, seed=args.seed, download=args.download
    )
    
    # 模型定义统一放在 models/model.py，训练和评估共用同一份实现
    model = build_model(pretrained=True).to(device)
    
    # 2. 根据参数选择损失函数
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # ⚠️ Label Smoothing
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 3. 用实验名字区分日志文件夹
    log_dir = f'{args.out}/{args.exp_name}'
    writer = SummaryWriter(log_dir)
    best_acc = 0.0
    history = []
    
    print(f"开始训练 {args.exp_name}...")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        history.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'epoch_time_s': round(epoch_time, 1),
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | {epoch_time:.1f}s")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存不同实验的模型，防止覆盖
            torch.save(model.state_dict(), f'best_model_{args.exp_name}.pth') 
            print(f"[best] 保存最佳模型！当前准确率: {best_acc:.4f}")
            
    writer.close()

    # 每轮指标落盘，报告画曲线直接用 history.csv
    with open(f'{log_dir}/history.csv', 'w', newline='', encoding='utf-8') as fh:
        csv_writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(history)

    print(f"训练完成！{args.exp_name} 最佳验证集准确率: {best_acc:.4f}")

if __name__ == "__main__":
    main()
