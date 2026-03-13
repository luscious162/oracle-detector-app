import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import copy
import time
from sklearn.model_selection import train_test_split
import numpy as np

# ================= 配置参数 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30 # 增加到30轮，因为引入增强后收敛会变慢，但更稳健
NUM_CLASSES = 2
DATA_DIR = 'all_dataset'

# ================= 核心：防捷径逻辑 =================

class BackgroundSanitizer(object):
    """
    强制背景洗白：将所有接近白色的像素统一为 255。
    彻底抹平 bone (254.6) 和 shell (254.7) 之间的微小亮度偏移。
    """
    def __init__(self, threshold=250):
        self.threshold = threshold / 255.0

    def __call__(self, tensor):
        # 假设 tensor 已转为 [C, H, W] 且在 [0, 1] 范围
        mask = tensor > self.threshold
        tensor[mask] = 1.0 # 强制设为绝对纯白
        return tensor

class AddGaussianNoise(object):
    """
    背景噪声对齐：给图片注入随机高斯噪声。
    使得模型无法依赖背景的标准差（1.00 vs 0.74）进行分类。
    """
    def __init__(self, mean=0.0, std=0.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# ================= 辅助函数: TTA =================

def tta_inference(model, inputs):
    out_0 = torch.softmax(model(inputs), dim=1)
    inputs_90 = torch.rot90(inputs, k=1, dims=[2, 3])
    out_90 = torch.softmax(model(inputs_90), dim=1)
    inputs_180 = torch.rot90(inputs, k=2, dims=[2, 3])
    out_180 = torch.softmax(model(inputs_180), dim=1)
    inputs_270 = torch.rot90(inputs, k=3, dims=[2, 3])
    out_270 = torch.softmax(model(inputs_270), dim=1)
    return (out_0 + out_90 + out_180 + out_270) / 4.0

# ================= 主程序 =================

def main():
    print(f"🚀 初始化防捷径训练方案 (GPU: {torch.cuda.get_device_name(0)})")
    
    # 1. 定义增强策略
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            # 增加缩放范围，防止模型依赖物体在画面中的固定大小
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180), # 拓片完全无方向，给 180 度旋转
            transforms.ToTensor(),
            BackgroundSanitizer(threshold=248), # 关键步骤 1：抹平背景差异
            AddGaussianNoise(std=0.02),         # 关键步骤 2：用强噪声覆盖微弱背景差异
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # 关键步骤 3：遮挡局部，强迫学习全局纹理
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            BackgroundSanitizer(threshold=248), # 验证集也要统一背景
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. 加载与划分
    full_dataset = datasets.ImageFolder(DATA_DIR)
    targets = full_dataset.targets
    
    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        shuffle=True,
        stratify=targets,
        random_state=42
    )

    train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    val_dataset_full = datasets.ImageFolder(DATA_DIR, transform=data_transforms['val'])

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    dataset_sizes = {'train': len(train_idx), 'val': len(val_idx)}

    # 3. 构建模型
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # 4. 优化器与调度
    # 增加 weight_decay 到 0.1，进一步惩罚对噪声的过度拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. 训练循环
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        outputs = tta_inference(model, inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {running_loss/dataset_sizes[phase]:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # 保存
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'swin_robust_fixed_background.pth')
    print(f"\n✅ 稳健性模型已保存。Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()