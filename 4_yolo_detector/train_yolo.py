"""
YOLO训练脚本（使用ultralytics库）
需要安装: pip install ultralytics
"""
import os
from pathlib import Path
from ultralytics import YOLO


def main():
    # 使用YOLOv8 nano模型（最快）
    model = YOLO('yolov8n.pt')
    
    # 训练配置
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsize=640,
        batch=16,
        patience=10,
        save=True,
        project='runs/detect',
        name='text_detector',
        exist_ok=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        conf=None,
        iou=0.7,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )
    
    print("训练完成!")
    print(f"最佳模型路径: {results.save_dir}/weights/best.pt")
    
    # 验证模型
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    main()
