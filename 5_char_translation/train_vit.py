import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =======================
# 1. 环境与性能优化配置
# =======================
# 启用 cuDNN 自动寻找最优算法
torch.backends.cudnn.benchmark = True

def main():
    # 配置参数
    DATA_DIR = './char_dataset'
    BATCH_SIZE = 32  # RTX 4060 开启 AMP 后 32 是比较理想的
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    
    # 强制指定 CUDA
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 已检测到显卡: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # =======================
    # 2. 数据增强
    # =======================
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.RandomInvert(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =======================
    # 3. 数据集与采样器
    # =======================
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    NUM_CLASSES = len(full_dataset.classes)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # 类别权重采样 (解决不平衡)
    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = torch.bincount(torch.tensor(train_targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # RTX 4060 建议开启 pin_memory
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # =======================
    # 4. 模型与 AMP 初始化
    # =======================
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # 初始化梯度缩放器 (AMP 核心)
    scaler = torch.amp.GradScaler('cuda') 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # =======================
    # 5. 训练循环
    # =======================
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 使用混合精度进行前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 缩放损失值并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step()
        
        print(f"✨ Epoch {epoch+1} Summary: Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_oracle_bone_vit_rtx4060.pth')
            print("💾 模型已保存")

if __name__ == '__main__':
    main()