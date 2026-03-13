"""
YOLO模型推理脚本
用于在新图像上检测文字区域
"""
import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np


def detect_text_regions(model_path, image_path, conf=0.25, iou=0.45):
    """
    检测图像中的文字区域
    
    Args:
        model_path: 模型权重路径
        image_path: 待检测图像路径
        conf: 置信度阈值
        iou: IOU阈值
    
    Returns:
        results: 检测结果
    """
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=True,
        save_conf=True,
        show=True
    )
    
    return results


def batch_detect(model_path, image_dir, output_dir='results'):
    """
    批量检测目录下所有图像
    
    Args:
        model_path: 模型权重路径
        image_dir: 图像目录
        output_dir: 结果保存目录
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_dir,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='detections'
    )
    
    print(f"检测完成，结果保存在: {output_dir}/detections")
    return results


def extract_detected_boxes(txt_path):
    """
    从YOLO生成的txt文件中提取检测框
    
    Args:
        txt_path: YOLO格式的标注文件路径
    
    Returns:
        boxes: 检测框列表 [class_id, x_center, y_center, width, height, confidence]
    """
    boxes = []
    
    if not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                
                boxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })
    
    return boxes


def convert_yolo_to_xyxy(yolo_box, img_width, img_height):
    """
    将YOLO格式的边界框转换为XYXY格式
    
    Args:
        yolo_box: [x_center, y_center, width, height] (归一化)
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        xyxy: [x1, y1, x2, y2] (像素坐标)
    """
    x_center, y_center, width, height = yolo_box
    
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("用法:")
        print("  python inference_yolo.py detect <model_path> <image_path>")
        print("  python inference_yolo.py batch <model_path> <image_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'detect':
        model_path = sys.argv[2]
        image_path = sys.argv[3]
        detect_text_regions(model_path, image_path)
    
    elif mode == 'batch':
        model_path = sys.argv[2]
        image_dir = sys.argv[3]
        batch_detect(model_path, image_dir)
    
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)
