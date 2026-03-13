"""
YOLO目标检测模型训练脚本
用于检测拓片中的文字区域

数据集: 完整数据集/dataset_yolo
- images/: 2000张图片
- annotations/: 1985个标注文件(JSON格式)
"""

import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image


# ================= 配置参数 =================
# 数据集路径
SOURCE_DATA_DIR = Path('完整数据集/dataset_yolo')
OUTPUT_DATA_DIR = Path('4_yolo_detector/dataset')

# YOLO相关配置
NUM_CLASSES = 1  # 只有文字一类
CLASS_NAMES = ['text']  # 类别名称

# 训练超参数
IMG_SIZE = 640
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ================= 数据格式转换 =================

def convert_json_to_yolo(json_path, img_width, img_height):
    """
    将JSON标注转换为YOLO格式的txt标注
    JSON格式: [[x1, y1, x2, y2, class_id], ...]
    YOLO格式: class_id x_center y_center width height (归一化)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    yolo_annotations = []
    
    if 'ann' in data and data['ann']:
        for bbox in data['ann']:
            x1, y1, x2, y2, class_id = bbox
            
            # 计算YOLO格式的坐标（归一化到0-1）
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # 限制范围在0-1之间
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # YOLO格式: class_id x_center y_center width height
            yolo_annotations.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def prepare_yolo_dataset():
    """
    准备YOLO格式的数据集
    将JSON标注转换为YOLO格式，并划分训练集/验证集/测试集
    """
    print("=" * 50)
    print("开始准备YOLO格式数据集")
    print("=" * 50)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            output_path = OUTPUT_DATA_DIR / subdir
            if split != 'test':  # test集只有images
                (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    images_dir = SOURCE_DATA_DIR / 'images'
    annotations_dir = SOURCE_DATA_DIR / 'annotations'
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"总共找到 {len(image_files)} 张图片")
    
    # 划分数据集 (80% train, 10% val, 10% test)
    random.shuffle(image_files)
    train_size = int(0.8 * len(image_files))
    val_size = int(0.1 * len(image_files))
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    print(f"测试集: {len(test_files)} 张")
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    total_converted = 0
    
    for split_name, file_list in splits.items():
        for img_file in file_list:
            # 图片路径
            src_img_path = images_dir / img_file
            
            # 对应的JSON标注文件（不含后缀）
            img_name_without_ext = os.path.splitext(img_file)[0]
            json_file = img_name_without_ext + '.json'
            json_path = annotations_dir / json_file
            
            # 检查标注文件是否存在
            if not json_path.exists():
                print(f"警告: 找不到标注文件 {json_path}")
                continue
            
            # 读取图片获取尺寸
            try:
                with Image.open(src_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"警告: 无法读取图片 {src_img_path}: {e}")
                continue
            
            # 转换标注
            yolo_annotations = convert_json_to_yolo(json_path, img_width, img_height)
            
            if not yolo_annotations:
                print(f"警告: {img_name_without_ext} 没有有效标注")
                continue
            
            # 复制图片
            if split_name == 'test':
                dst_img_dir = OUTPUT_DATA_DIR / 'images'
            else:
                dst_img_dir = OUTPUT_DATA_DIR / 'images' / split_name
            
            dst_img_path = dst_img_dir / img_file
            shutil.copy2(src_img_path, dst_img_path)
            
            # 写入YOLO格式标注
            if split_name != 'test':
                dst_label_dir = OUTPUT_DATA_DIR / 'labels' / split_name
                dst_label_path = dst_label_dir / (img_name_without_ext + '.txt')
                
                with open(dst_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            total_converted += 1
    
    print(f"\n成功转换 {total_converted} 个样本")
    print("数据集准备完成！")
    print("=" * 50)
    
    return len(train_files), len(val_files), len(test_files)


def create_yaml_config(train_count, val_count, test_count):
    """
    创建YOLO数据集配置文件
    """
    yaml_content = f"""# YOLO目标检测数据集配置
# 用于训练检测拓片中文字区域的YOLO模型

# 数据集根目录
path: {OUTPUT_DATA_DIR.absolute()}
train: images/train
val: images/val

# 测试集（用于最终评估）
test: images/test

# 类别数量
nc: {NUM_CLASSES}

# 类别名称
names:
  0: text

# 数据集统计
# 训练集: {train_count} 张
# 验证集: {val_count} 张
# 测试集: {test_count} 张
"""
    
    yaml_path = OUTPUT_DATA_DIR / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_path}")


# ================= 主函数 =================

def main():
    """
    主函数：准备数据集
    """
    print("\n" + "=" * 60)
    print("YOLO目标检测模型训练准备")
    print("=" * 60 + "\n")
    
    # 1. 准备YOLO格式数据集
    train_count, val_count, test_count = prepare_yolo_dataset()
    
    # 2. 创建数据集配置文件
    create_yaml_config(train_count, val_count, test_count)
    
    print("\n" + "=" * 60)
    print("准备完成！")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 安装依赖: pip install ultralytics")
    print("2. 运行训练: python train_yolo.py")
    print("3. 运行推理: python inference_yolo.py detect <model> <image>")
    print("=" * 60)


if __name__ == '__main__':
    main()
