# GAN生成器模块

## 简介

本模块使用Pix2Pix（条件生成对抗网络）实现拓片到摹本的图像转换任务。模型接收拓片图像作为输入，生成对应的摹本图像。

## 数据集

训练数据存放在 `dataset/` 目录下：
- `dataset/tapian/` - 拓片图像（原始甲骨文拓片）
- `dataset/muben/` - 摹本图像（人工描绘的甲骨文）

两个目录中的图像文件名一一对应，形成训练图像对。

## 文件说明

```
3_gan_genarator/
├── run_training.py    # GAN模型训练脚本（本模块核心文件）
└── readme.txt         # 本文件
```

## 核心组件

### 1. 配置参数 (Config类)

定义训练相关的所有超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| tapian_dir | 'dataset/tapian' | 拓片图像目录 |
| muben_dir | 'dataset/muben' | 摹本图像目录 |
| batch_size | 16 | 批次大小 |
| num_epochs | 200 | 训练轮数 |
| learning_rate | 0.0002 | 学习率 |
| image_size | 512 | 输入图像尺寸 |
| lambda_l1 | 100 | L1损失权重 |
| ngf | 64 | 生成器特征数 |
| ndf | 64 | 判别器特征数 |

### 2. 数据集类 (OracleBoneDataset)

自定义PyTorch数据集，用于加载拓片-摹本图像对：
- 遍历拓片目录，寻找有对应摹本的图像
- 返回 (拓片, 摹本) 图像对
- 支持图像变换（Resize、归一化）

### 3. 生成器 (UNetGenerator)

基于U-Net结构的生成器网络：
- 编码器-解码器架构，带跳跃连接（skip connections）
- 8层编码器 + 8层解码器
- 输入：拓片图像 (1通道灰度)
- 输出：生成的摹本图像 (1通道灰度)

### 4. 判别器 (PatchGANDiscriminator)

PatchGAN判别器：
- 输入：拓片与目标图像的拼接 (2通道)
- 输出：_patch_级别的概率图（判断每个patch是否真实）
- 4层卷积结构，更关注局部纹理细节

### 5. 训练器 (Trainer)

完整的训练流程管理：
- 模型初始化与优化器配置
- 混合精度训练（AMP）支持
- 训练循环与损失计算
- 模型检查点保存
- 样本图像可视化

## 损失函数

模型使用多损失联合优化：
1. **GAN损失 (BCEWithLogitsLoss)** - 对抗损失，让生成器生成更真实的图像
2. **L1损失** - 像素级重建损失，确保生成图像与目标图像的像素相似度

总损失公式：
```
loss_G = loss_GAN + lambda_l1 * loss_L1
```

## 训练输出

训练过程中会自动创建输出目录 `oracle_bone_gan_output_YYYYMMDD_HHMMSS/`，包含：

```
oracle_bone_gan_output_YYYYMMDD_HHMMSS/
├── checkpoints/     # 模型权重
│   ├── G_epoch_X.pth    # 生成器
│   └── D_epoch_X.pth    # 判别器
├── samples/         # 可视化样本
│   └── epoch_X.png     # 每隔几个epoch生成的结果对比图
└── logs/            # 训练日志
```

## 使用方法

### 1. 准备数据

确保数据集目录结构正确：
```
dataset/
├── tapian/
│   ├── xxx.jpg
│   └── ...
└── muben/
    ├── xxx.jpg    # 与tapian中文件名对应
    └── ...
```

### 2. 运行训练

```bash
cd 3_gan_genarator
python run_training.py
```

### 3. 使用训练好的模型推理

训练完成后，参考以下代码进行推理：
```python
import torch
from run_training import UNetGenerator, Config

# 加载模型
config = Config()
generator = UNetGenerator()
generator.load_state_dict(torch.load('checkpoints/G_epoch_XXX.pth'))
generator.eval()

# 推理
with torch.no_grad():
    generated = generator(input_image)
```

## 技术特点

1. **U-Net结构** - 跳跃连接保留了图像细节，适合图像到图像的转换任务
2. **PatchGAN判别器** - 更关注局部纹理，生成更精细的边缘
3. **L1损失** - 像素级监督，确保内容准确性
4. **AMP混合精度** - 减少显存占用，加速训练
5. **动态输出目录** - 每次训练创建带时间戳的独立输出目录

## 注意事项

1. 拓片和摹本图像需要一一对应（相同文件名）
2. 建议使用GPU进行训练，512x512图像需要较大显存
3. 可根据硬件调整 batch_size 和 image_size
4. 训练200轮可能需要较长时间，建议使用早停或手动中断
