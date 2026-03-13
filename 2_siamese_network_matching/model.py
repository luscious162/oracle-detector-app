# Siamese 网络模型定义
# 功能：定义Siamese孪生网络架构用于甲骨缀合

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    采用VGG16架构的卷积层进行特征提取
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

        # 自适应池化，将特征池化到1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class SiameseNetwork(nn.Module):
    """Siamese 孪生网络
    
    架构说明：
    - 两个共享权重的VGG16骨干网络
    - 曼哈顿距离度量特征相似性
    - 全连接层 + Sigmoid输出可缀合概率
    
    输入：105x105的碴口曲线图像对
    输出：可缀合概率 (0-1之间)
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
        """双分支前向传播
        
        Args:
            x1: 第一张碴口曲线图像
            x2: 第二张碴口曲线图像
        
        Returns:
            output: 可缀合概率 (0-1)
        """
        # 提取两个输入的特征（共享权重）
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)

        # 计算曼哈顿距离（L1距离）
        diff = torch.abs(f1 - f2)

        # 输出可缀合概率
        output = self.metric(diff)
        return output


class SiameseNetworkWithFeatures(nn.Module):
    """带特征输出的Siamese网络（用于特征可视化）"""

    def __init__(self, in_channels=1):
        super(SiameseNetworkWithFeatures, self).__init__()
        self.backbone = VGG16Backbone(in_channels)

    def extract_features(self, x):
        """提取单张图像的特征"""
        return self.backbone(x)

    def compute_similarity(self, x1, x2):
        """计算两张图像的相似度"""
        f1 = self.extract_features(x1)
        f2 = self.extract_features(x2)
        
        # 曼哈顿距离
        manhattan_dist = torch.abs(f1 - f2).mean()
        
        # 余弦相似度
        cosine_sim = F.cosine_similarity(f1, f2).mean()
        
        # 欧氏距离
        euclidean_dist = F.pairwise_distance(f1, f2).mean()
        
        return {
            'manhattan_distance': manhattan_dist.item(),
            'cosine_similarity': cosine_sim.item(),
            'euclidean_distance': euclidean_dist.item()
        }


def test_model():
    """测试模型输出形状"""
    model = SiameseNetwork(in_channels=3)
    x1 = torch.randn(2, 3, 105, 105)
    x2 = torch.randn(2, 3, 105, 105)
    y = model(x1, x2)
    print(f"输入1形状: {x1.shape}")
    print(f"输入2形状: {x2.shape}")
    print(f"输出形状: {y.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_model()
