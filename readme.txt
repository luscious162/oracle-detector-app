# 素材与源码

本项目是一个甲骨文拓片识别与处理系统，包含四个核心模块和一个集成应用，用于完成从图像检测、分类、匹配到生成的完整流程。

## 文件结构

```
素材与源码/
├── 1_swin_transformer_classification/  # 骨片/龟甲分类
├── 2_siamese_network_matching/        # 文字匹配
├── 3_gan_genarator/                   # 图像生成
├── 4_yolo_detector/                  # 文字检测
├── application.py                     # Web 集成应用（推荐使用）
└── readme.txt                         # 本文档
```

## 核心模块功能

### 1. 1_swin_transformer_classification（骨片/龟甲分类）

**功能：** 使用 Swin Transformer 模型对甲骨文拓片进行分类，区分**骨片（Bone）**和**龟甲（Shell）**。

**技术特点：**
- 预训练 Swin-Tiny 模型 + 防捷径训练策略（背景洗白、噪声注入、随机遮挡）
- TTA（测试时增强）提升准确率
- 二分类任务，输出 Bone/Shell 类别

### 2. 2_siamese_network_matching（文字匹配/缀合）

**功能：** 使用孪生网络（Siamese Network）判断两个甲骨文字符是否属于**同一字**，支持碎片缀合。

**技术特点：**
- 基于 VGG16 的孪生网络结构
- 对比损失（Contrastive Loss）训练
- 支持 One-shot 识别

### 3. 3_gan_genarator（摹本生成）

**功能：** 使用 Pix2Pix GAN 将**拓片图像**转换为**摹本图像**。

**技术特点：**
- U-Net 生成器 + PatchGAN 判别器
- L1 + GAN 联合损失
- 图像到图像的转换

### 4. 4_yolo_detector（文字检测）

**功能：** 使用 YOLOv8 检测拓片图像中的**文字区域**。

**技术特点：**
- YOLOv8 Nano 模型
- 输出文字区域的边界框坐标

---

## 集成应用：application.py（推荐使用）

`application.py` 是一个** Flask Web 集成应用**，将上述四个模块的功能整合为一个统一的可视化 Web 系统。

### 核心功能

| 功能模块 | 描述 |
|---------|------|
| **分类识别** | 上传拓片图片，自动识别为骨片(Bone)或龟甲(Shell)，并生成 Grad-CAM 注意力热图 |
| **碎片缀合** | 上传两个甲骨文碎片图片，使用几何配准+孪生网络联合进行智能缀合 |
| **摹本生成** | 上传拓片图片，使用 GAN 生成对应的摹本图像 |
| **甲骨文检测** | 上传图片，使用 YOLOv8 检测并框出其中的甲骨文字符 |

### 技术实现

- **Web 框架**：Flask + Flask-CORS
- **前端**：原生 HTML/CSS/JS，响应式设计，支持移动端触摸滑动
- **模型加载**：自动从 GitHub 下载预训练权重
- **可视化**：Grad-CAM 热图、YOLO 检测框、图像轮播展示

### 启动方式

```bash
cd 素材与源码
pip install flask flask-cors torch torchvision opencv-python pillow numpy requests tqdm pytorch-grad-cam ultralytics
python application.py
```

启动后自动打开浏览器访问 `http://127.0.0.1:5000`。

### 预训练模型

四个模块的预训练模型权重（自动下载）：

| 模块 | 权重文件 | 下载链接 |
|-----|---------|---------|
| 模块一 | `best_swin.pth` | [GitHub](https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_swin.pth) |
| 模块二 | `best_siamese.pth` | [GitHub](https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_siamese.pth) |
| 模块三 | `best_gan.pth` | [GitHub](https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_gan.pth) |
| 模块四 | `inscription_detect.pt` | [GitHub](https://github.com/luscious162/oracle_CCCC/releases/download/weight/inscription_detect.pt) |

### 依赖库

```
torch, torchvision, flask, flask-cors, opencv-python, pillow, numpy, requests, tqdm, pytorch-grad-cam, ultralytics
```

---

## 使用建议

1. **入门推荐**：直接运行 `python application.py` 体验完整功能
2. **模块独立训练**：如需重新训练各模块模型，进入对应子文件夹操作
3. **GPU 加速**：建议使用 NVIDIA GPU 加速推理（自动检测 CUDA）
