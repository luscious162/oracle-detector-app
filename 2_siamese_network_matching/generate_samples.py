# -*- coding: utf-8 -*-
"""
生成甲骨碴口曲线样例数据
功能：生成模拟的甲骨碴口曲线图像及配对文件
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random


def generate_curve_image(width=105, height=105, num_points=20, noise=2):
    """生成一条碴口曲线图像
    
    Args:
        width, height: 图像尺寸
        num_points: 曲线控制点数量
        noise: 曲线噪声程度
    
    Returns:
        PIL Image对象
    """
    # 生成平滑曲线控制点
    x_points = np.linspace(10, width - 10, num_points)
    y_points = []
    
    # 随机生成有起伏的y坐标
    base_y = height // 2
    for i in range(num_points):
        # 中间部分弯曲较大，两端较平
        factor = np.sin(np.pi * i / (num_points - 1))
        y = base_y + random.randint(-20, 20) * factor + random.randint(-noise, noise)
        y = max(15, min(height - 15, y))
        y_points.append(y)
    
    # 创建图像
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # 绘制曲线（使用线条连接各点）
    points = list(zip(x_points.astype(int), y_points))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=255, width=2)
    
    return img


def generate_matching_curve(base_curve, variation=5):
    """生成与给定曲线相似的曲线（用于正样本）
    
    Args:
        base_curve: 基础曲线点列表
        variation: 变化程度
    
    Returns:
        新的曲线点列表
    """
    new_curve = []
    for x, y in base_curve:
        new_x = x + random.randint(-variation, variation)
        new_y = y + random.randint(-variation, variation)
        new_curve.append((max(0, min(104, new_x)), max(0, min(104, new_y))))
    return new_curve


def create_curve_from_points(points, width=105, height=105):
    """根据点列表创建曲线图像"""
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=255, width=2)
    
    return img


def generate_dataset(base_path, num_positive=10, num_negative=10):
    """生成完整的数据集样例
    
    Args:
        base_path: 数据集根目录
        num_positive: 正样本对数量
        num_negative: 负样本对数量
    """
    train_curves = os.path.join(base_path, 'train', 'curves')
    val_curves = os.path.join(base_path, 'val', 'curves')
    
    # 生成训练集
    print("生成训练集曲线图像...")
    train_curves_list = []
    
    for i in range(num_positive + num_negative):
        # 生成基础曲线
        img = generate_curve_image()
        curve_path = os.path.join(train_curves, f'curve_{i:03d}.png')
        img.save(curve_path)
        train_curves_list.append(curve_path)
    
    # 生成验证集
    print("生成验证集曲线图像...")
    val_curves_list = []
    
    for i in range(num_positive // 2 + num_negative // 2):
        img = generate_curve_image()
        curve_path = os.path.join(val_curves, f'curve_{i:03d}.png')
        img.save(curve_path)
        val_curves_list.append(curve_path)
    
    # 生成正样本对文件（可缀合）
    print("生成正样本对...")
    positive_file = os.path.join(base_path, 'train', 'positive', 'pairs.txt')
    with open(positive_file, 'w') as f:
        for i in range(num_positive):
            # 选择两条相似曲线
            if i < len(train_curves_list) // 2:
                curve1 = train_curves_list[i]
                curve2 = train_curves_list[i + 1]
            else:
                curve1 = train_curves_list[i % len(train_curves_list)]
                curve2 = train_curves_list[(i + 1) % len(train_curves_list)]
            f.write(f"{curve1},{curve2}\n")
    
    # 生成负样本对文件（不可缀合）
    print("生成负样本对...")
    negative_file = os.path.join(base_path, 'train', 'negative', 'pairs.txt')
    with open(negative_file, 'w') as f:
        for i in range(num_negative):
            # 选择两条不相似曲线（距离较远）
            idx1 = i
            idx2 = (i + len(train_curves_list) // 2) % len(train_curves_list)
            curve1 = train_curves_list[idx1]
            curve2 = train_curves_list[idx2]
            f.write(f"{curve1},{curve2}\n")
    
    # 生成验证集配对文件
    val_positive_file = os.path.join(base_path, 'val', 'positive', 'pairs.txt')
    with open(val_positive_file, 'w') as f:
        for i in range(num_positive // 2):
            if i < len(val_curves_list) // 2:
                curve1 = val_curves_list[i]
                curve2 = val_curves_list[i + 1]
            else:
                curve1 = val_curves_list[i % len(val_curves_list)]
                curve2 = val_curves_list[(i + 1) % len(val_curves_list)]
            f.write(f"{curve1},{curve2}\n")
    
    val_negative_file = os.path.join(base_path, 'val', 'negative', 'pairs.txt')
    with open(val_negative_file, 'w') as f:
        for i in range(num_negative // 2):
            idx1 = i
            idx2 = (i + len(val_curves_list) // 2) % len(val_curves_list)
            curve1 = val_curves_list[idx1]
            curve2 = val_curves_list[idx2]
            f.write(f"{curve1},{curve2}\n")
    
    print(f"\n数据集生成完成!")
    print(f"训练集: {len(train_curves_list)} 条曲线")
    print(f"验证集: {len(val_curves_list)} 条曲线")
    print(f"正样本对: {num_positive} 对")
    print(f"负样本对: {num_negative} 对")


def generate_more_realistic_curves():
    """生成更逼真的碴口曲线"""
    curves = []
    
    for variant in range(5):
        # 不同的曲线形态
        img = Image.new('L', (105, 105), 0)
        draw = ImageDraw.Draw(img)
        
        # 模拟不同形状的碴口曲线
        if variant == 0:  # 平滑上升
            for x in range(10, 95):
                y = 52 + int(20 * np.sin(np.pi * (x - 10) / 85)) + random.randint(-2, 2)
                draw.point((x, y), fill=255)
        elif variant == 1:  # 平滑下降
            for x in range(10, 95):
                y = 52 - int(20 * np.sin(np.pi * (x - 10) / 85)) + random.randint(-2, 2)
                draw.point((x, y), fill=255)
        elif variant == 2:  # S形曲线
            for x in range(10, 95):
                y = 52 + int(25 * np.sin(2 * np.pi * (x - 10) / 85)) + random.randint(-2, 2)
                draw.point((x, y), fill=255)
        elif variant == 3:  # 先平后升
            for x in range(10, 95):
                if x < 50:
                    y = 40 + random.randint(-2, 2)
                else:
                    y = 40 + int(20 * (x - 50) / 45) + random.randint(-2, 2)
                draw.point((x, y), fill=255)
        else:  # 波浪形
            for x in range(10, 95):
                y = 52 + int(15 * np.sin(3 * np.pi * (x - 10) / 85)) + random.randint(-2, 2)
                draw.point((x, y), fill=255)
        
        curves.append(img)
    
    return curves


if __name__ == "__main__":
    # 数据集路径
    base_path = "/Users/Zhuanz/Desktop/oracle/2_siamese_network_matching/dataset"
    
    # 生成数据集样例
    generate_dataset(base_path, num_positive=10, num_negative=10)
    
    # 同时生成一些更逼真的曲线
    print("\n生成逼真曲线样例...")
    realistic_curves = generate_more_realistic_curves()
    for i, curve_img in enumerate(realistic_curves):
        curve_img.save(os.path.join(base_path, 'train', 'curves', f'realistic_{i}.png'))
    
    print("逼真曲线生成完成!")
