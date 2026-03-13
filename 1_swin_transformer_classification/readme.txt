# Swin Transformer 分类模型训练文件夹

## 文件说明

### run_training.py
甲骨文拓片分类训练脚本，使用 Swin Transformer (swin_t) 模型对甲骨文拓片进行骨片(Bone)与龟甲(Shell)分类。

**主要功能：**
- 使用 torchvision 提供的预训练 Swin-Tiny 模型作为骨干网络
- 自适应调整分类头为二分类（Bone/Shell）
- 采用防捷径训练策略，防止模型依赖背景差异进行分类：
  - `BackgroundSanitizer`: 强制背景洗白，消除 bone 和 shell 之间的微小亮度差异
  - `AddGaussianNoise`: 注入高斯噪声，破坏背景标准差特征
  - `RandomErasing`: 随机遮挡局部区域，强迫模型学习全局纹理特征
- 数据增强：随机裁剪、翻转、旋转（180度）
- TTA (Test Time Augmentation): 推理时使用多角度旋转融合提升准确率
- 使用 AdamW 优化器和余弦学习率调度器

**训练配置：**
- 批量大小: 32
- 学习率: 1e-4
- 训练轮数: 30 epochs
- 权重衰减: 0.1
- 标签平滑: 0.1

**输出：**
- 训练完成后保存模型为 `swin_robust_fixed_background.pth`

### dataset/
训练和验证数据集目录，结构如下：

```
dataset/
├── train/
│   ├── bone/
│   │   └── original_bone.png    # 骨片训练样本
│   └── shell/
│       └── original_shell.png    # 龟甲训练样本
└── val/
    ├── bone/
    │   └── original_bone.png     # 骨片验证样本
    └── shell/
        └── original_shell.png    # 龟甲验证样本
```

**注意：** 实际训练时使用 `all_dataset` 目录（需从外部指定），该目录应包含按类别组织的图像子文件夹。

## 使用方法

```bash
# 进入目录
cd 1_swin_transformer_classification

# 确保数据集目录结构如下：
# all_dataset/
# ├── bone/
# │   └── *.png (或其他图像格式)
# └── shell/
#     └── *.png (或其他图像格式)

# 运行训练
python run_training.py
```

## 依赖库

- torch
- torchvision
- scikit-learn
- numpy
