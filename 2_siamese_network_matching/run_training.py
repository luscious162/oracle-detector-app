# Siamese 孪生网络 - 甲骨缀合模型
# 功能：基于碴口曲线相似度的甲骨缀合匹配

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import random
from scipy.interpolate import interp1d


# ============== 配置参数 ==============
DATA_PATH = "/Users/Zhuanz/Desktop/oracle/3_siamese_network_matching/dataset"  # 数据集路径，留空
IMAGE_SIZE = 105  # 论文中的标准输入尺寸
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
MARGIN = 1.0  # 对比损失边界
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== Siamese 网络模型 ==============
class ConvBlock(nn.Module):
    """卷积块：Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VGG16Backbone(nn.Module):
    """VGG16 骨干网络（简化版）
    用于提取碴口曲线的深度特征
    """

    def __init__(self, in_channels=1):
        super(VGG16Backbone, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            # Block 1
            ConvBlock(in_channels, 64, kernel_size=3, padding=1),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 4
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SiameseNetwork(nn.Module):
    """Siamese 孪生网络
    
    架构：共享权重的两个VGG16骨干网络
    度量：曼哈顿距离 + 全连接层 + Sigmoid
    输出：可缀合概率 (0-1)
    """

    def __init__(self, in_channels=1):
        super(SiameseNetwork, self).__init__()

        # 共享权重的骨干网络
        self.backbone = VGG16Backbone(in_channels)

        # 特征度量层
        self.metric = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        """单分支特征提取"""
        features = self.backbone(x)
        return features

    def forward(self, x1, x2):
        """双分支前向传播"""
        # 提取两个输入的特征（共享权重）
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)

        # 计算曼哈顿距离（特征差异）
        diff = torch.abs(f1 - f2)

        # 输出可缀合概率
        output = self.metric(diff)
        return output


# ============== 碴口曲线提取 ==============
def extract_edge_curve(image_path, threshold=128):
    """从图像中提取碴口曲线轨迹序列
    
    步骤：
    1. 阈值分割
    2. 边缘检测
    3. 轨迹规范化
    """
    # 加载图像并灰度化
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # 阈值分割
    binary = (img_array > threshold).astype(np.uint8)
    
    # 寻找边缘
    edges = []
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] > 0:
                # 检查是否为边缘点
                if i > 0 and binary[i-1, j] == 0:
                    edges.append((j, i))
                elif i < binary.shape[0]-1 and binary[i+1, j] == 0:
                    edges.append((j, i))
                elif j > 0 and binary[i, j-1] == 0:
                    edges.append((j, i))
                elif j < binary.shape[1]-1 and binary[i, j+1] == 0:
                    edges.append((j, i))
    
    if len(edges) == 0:
        return None
    
    # 转换为numpy数组
    edges = np.array(edges)
    
    # 规范化曲线
    curve = normalize_curve(edges)
    
    return curve


def normalize_curve(curve, target_length=200):
    """规范化曲线为固定长度
    
    使用插值将曲线规范化为统一长度
    """
    if curve is None or len(curve) < 2:
        return None
    
    # 按x坐标排序
    curve = curve[curve[:, 0].argsort()]
    
    # 创建参数化曲线
    t = np.linspace(0, 1, len(curve))
    t_new = np.linspace(0, 1, target_length)
    
    try:
        f_x = interp1d(t, curve[:, 0], kind='linear')
        f_y = interp1d(t, curve[:, 1], kind='linear')
        
        x_new = f_x(t_new)
        y_new = f_y(t_new)
        
        return np.column_stack([x_new, y_new])
    except:
        return None


def curve_to_image(curve, size=105):
    """将曲线转换为图像
    
    将规范化后的曲线绘制成灰度图像
    """
    if curve is None:
        return None
    
    # 创建空白图像
    img = np.zeros((size, size), dtype=np.uint8)
    
    # 归一化曲线坐标到图像尺寸
    x = ((curve[:, 0] - curve[:, 0].min()) / (curve[:, 0].max() - curve[:, 0].min() + 1e-8) * (size - 1)).astype(int)
    y = ((curve[:, 1] - curve[:, 1].min()) / (curve[:, 1].max() - curve[:, 1].min() + 1e-8) * (size - 1)).astype(int)
    
    # 绘制曲线
    for i in range(len(x) - 1):
        # 简单的线条绘制
        cv2_line(img, (x[i], y[i]), (x[i+1], y[i+1]), 255, 2)
    
    return img


def cv2_line(img, pt1, pt2, color, thickness):
    """简单的线条绘制函数"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    while True:
        if 0 <= y1 < img.shape[0] and 0 <= x1 < img.shape[1]:
            img[y1, x1] = color
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


# ============== 数据增强：多曲线融合 ==============
def multi_curve_fusion(curves, num_fusions=5):
    """多曲线融合算法
    
    针对可缀合样本匮乏的问题，
    通过随机截取真实曲线片段并进行重组，
    生成大量有效的数据样本
    """
    if len(curves) < 2:
        return []
    
    fused_curves = []
    
    for _ in range(num_fusions):
        # 随机选择两条曲线
        curve1 = curves[random.randint(0, len(curves) - 1)]
        curve2 = curves[random.randint(0, len(curves) - 1)]
        
        # 随机截取片段
        start1 = random.randint(0, len(curve1) // 2)
        end1 = random.randint(len(curve1) // 2, len(curve1))
        
        start2 = random.randint(0, len(curve2) // 2)
        end2 = random.randint(len(curve2) // 2, len(curve2))
        
        segment1 = curve1[start1:end1]
        segment2 = curve2[start2:end2]
        
        # 重组片段
        fused = np.concatenate([segment1, segment2], axis=0)
        
        # 规范化
        fused = normalize_curve(fused)
        
        if fused is not None:
            fused_curves.append(fused)
    
    return fused_curves


# ============== 数据集类 ==============
class OracleMatchingDataset(Dataset):
    """甲骨缀合匹配数据集"""

    def __init__(self, data_path, split='train', use_fusion=True):
        self.data_path = data_path
        self.split = split
        self.use_fusion = use_fusion
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        # 加载配对数据
        self.positive_pairs = []  # 正样本：可缀合
        self.negative_pairs = []  # 负样本：不可缀合
        
        self.load_pairs()

    def load_pairs(self):
        """加载正负样本对"""
        positive_dir = os.path.join(self.data_path, self.split, 'positive')
        negative_dir = os.path.join(self.data_path, self.split, 'negative')
        
        # 加载正样本
        if os.path.exists(positive_dir):
            for file in os.listdir(positive_dir):
                if file.endswith('.txt'):
                    pair_file = os.path.join(positive_dir, file)
                    with open(pair_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split(',')
                            if len(parts) == 2:
                                self.positive_pairs.append((parts[0], parts[1], 1))
        
        # 加载负样本
        if os.path.exists(negative_dir):
            for file in os.listdir(negative_dir):
                if file.endswith('.txt'):
                    pair_file = os.path.join(negative_dir, file)
                    with open(pair_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split(',')
                            if len(parts) == 2:
                                self.negative_pairs.append((parts[0], parts[1], 0))
        
        # 数据增强：多曲线融合
        if self.use_fusion and self.split == 'train':
            self.apply_curve_fusion()

    def apply_curve_fusion(self):
        """应用多曲线融合生成更多样本"""
        # 从现有曲线中提取
        all_curves = []
        
        for pair in self.positive_pairs + self.negative_pairs:
            img1_path = pair[0]
            if os.path.exists(img1_path):
                curve = extract_edge_curve(img1_path)
                if curve is not None:
                    all_curves.append(curve)
        
        # 生成融合样本
        if len(all_curves) >= 2:
            fused_curves = multi_curve_fusion(all_curves, num_fusions=len(all_curves))
            
            # 将融合曲线添加到正样本
            for fused_curve in fused_curves[:len(all_curves)]:
                # 保存融合曲线为临时图像
                fused_img = curve_to_image(fused_curve, IMAGE_SIZE)
                if fused_img is not None:
                    temp_path = f"temp_fused_{random.randint(0, 10000)}.png"
                    Image.fromarray(fused_img).save(temp_path)
                    # 添加自配对作为正样本
                    self.positive_pairs.append((temp_path, temp_path, 1))

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            img1_path, img2_path, label = self.positive_pairs[idx]
        else:
            img1_path, img2_path, label = self.negative_pairs[idx - len(self.positive_pairs)]
        
        # 加载图像
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        
        # 转换为RGB（单通道复制3次）
        img1 = Image.merge('RGB', [img1, img1, img1])
        img2 = Image.merge('RGB', [img2, img2, img2])
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ============== 对比损失 ==============
class ContrastiveLoss(nn.Module):
    """对比损失 (Contrastive Loss)
    
    正样本：距离越小越好
    负样本：距离大于边界值MARGIN
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label):
        # output: 预测的可缀合概率 (0-1)
        # label: 真实标签 (1=正样本, 0=负样本)
        
        # 转换为距离
        distance = 1 - output
        
        # 正样本：使距离最小化
        positive_loss = label * torch.pow(distance, 2)
        
        # 负样本：使距离大于边界
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


# ============== 训练函数 ==============
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for img1, img2, labels in loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # 计算准确率
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


# ============== 主训练流程 ==============
def main():
    print(f"使用设备: {DEVICE}")

    # 创建模型
    model = SiameseNetwork(in_channels=3).to(DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载数据
    train_dataset = OracleMatchingDataset(DATA_PATH, split='train', use_fusion=True)
    val_dataset = OracleMatchingDataset(DATA_PATH, split='val', use_fusion=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 损失函数和优化器
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    # 训练循环
    best_acc = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'best_siamese.pth')
            print(f"  -> 保存最佳模型 (Accuracy: {best_acc:.2f}%)")

    print(f"\n训练完成! 最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
