# YOLO 目标检测模型

## 简介

本模块使用YOLOv8模型对拓片图像中的文字区域进行目标检测。

## 数据集

- **来源**: 完整数据集/dataset_yolo
- **图片数量**: 2000张
- **标注数量**: 1985个
- **标注格式**: JSON格式，边界框坐标 [x1, y1, x2, y2, class_id]

## 文件说明

```
4_yolo_detector/
├── prepare_dataset.py   # 数据集准备脚本（格式转换、划分训练/验证/测试集）
├── train_yolo.py       # YOLO模型训练脚本
├── inference_yolo.py   # YOLO模型推理脚本
├── dataset/            # 转换后的YOLO格式数据集
│   ├── data.yaml       # 数据集配置文件
│   ├── images/         # 图像文件
│   │   ├── train/     # 训练集
│   │   ├── val/       # 验证集
│   │   └── test/      # 测试集
│   └── labels/        # 标注文件
│       ├── train/     # 训练集标注
│       └── val/       # 验证集标注
└── readme.txt          # 本文件
```

## 使用方法

### 1. 安装依赖

```bash
pip install ultralytics
```

### 2. 准备数据集

运行数据集准备脚本，将JSON标注转换为YOLO格式，并划分训练集/验证集/测试集：

```bash
cd 4_yolo_detector
python prepare_dataset.py
```

此脚本会：
- 读取 `完整数据集/dataset_yolo` 中的图片和标注
- 将JSON标注转换为YOLO格式（class_id x_center y_center width height）
- 按8:1:1比例划分训练集、验证集、测试集
- 生成 `dataset/data.yaml` 配置文件

### 3. 训练模型

```bash
cd 4_yolo_detector
python train_yolo.py
```

训练配置：
- 模型: YOLOv8n (nano版本，速度最快)
- 输入尺寸: 640x640
- 批次大小: 16
- 训练轮数: 100
- 早停耐心: 10轮

训练完成后：
- 最佳模型: `runs/detect/text_detector/weights/best.pt`
- 最后模型: `runs/detect/text_detector/weights/last.pt`

### 4. 推理预测

#### 单张图片检测

```bash
python inference_yolo.py detect <model_path> <image_path>
```

示例：
```bash
python inference_yolo.py detect runs/detect/text_detector/weights/best.pt test_image.jpg
```

#### 批量检测

```bash
python inference_yolo.py batch <model_path> <image_dir>
```

示例：
```bash
python inference_yolo.py batch runs/detect/text_detector/weights/best.pt /path/to/images
```

### 5. API调用

```python
from inference_yolo import detect_text_regions, convert_yolo_to_xyxy
from PIL import Image

# 检测文字区域
results = detect_text_regions('best.pt', 'image.jpg', conf=0.25, iou=0.45)

# 转换为像素坐标
with Image.open('image.jpg') as img:
    width, height = img.size
    boxes = convert_yolo_to_xyxy([x_center, y_center, w, h], width, height)
```

## 模型性能

训练完成后可在控制台查看：
- mAP50: 50% IoU阈值下的平均精度
- mAP50-95: 50%-95% IoU阈值范围内的平均精度

## 注意事项

1. 数据集中有15张图片没有对应的标注文件，这些图片会被跳过
2. 默认使用YOLOv8n模型，如需更高精度可改为yolov8s/yolov8m/yolov8l
3. 训练需要GPU加速，如无GPU可减少epochs或使用更小的模型
