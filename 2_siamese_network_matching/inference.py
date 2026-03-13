# Siamese 网络推理脚本
# 功能：对甲骨碴口曲线图像进行缀合匹配预测

import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SiameseNetwork, SiameseNetworkWithFeatures
import os
import numpy as np

# ============== 配置参数 ==============
MODEL_PATH = "best_siamese.pth"
IMAGE_SIZE = 105  # 论文中的标准输入尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # 缀合阈值


def load_model(model_path, device):
    """加载训练好的Siamese网络模型"""
    model = SiameseNetwork(in_channels=3).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: {model_path}")
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化模型")
    
    model.eval()
    return model


def preprocess_image(image_path, size=105):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('L')
    # 转换为RGB（单通道复制3次）
    image = Image.merge('RGB', [image, image, image])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    
    return image_tensor, image


def predict_pair(model, image_path1, image_path2, device, threshold=0.5):
    """预测一对图像的可缀合概率
    
    Args:
        model: Siamese网络模型
        image_path1: 第一张图像路径
        image_path2: 第二张图像路径
        device: 计算设备
        threshold: 缀合阈值
    
    Returns:
        预测结果字典
    """
    img1, _ = preprocess_image(image_path1, IMAGE_SIZE)
    img2, _ = preprocess_image(image_path2, IMAGE_SIZE)
    
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    with torch.no_grad():
        output = model(img1, img2).squeeze()
        probability = output.item()
        is_match = probability > threshold
    
    return {
        'probability': probability,
        'is_match': is_match,
        'can_combine': '是' if is_match else '否',
        'confidence': probability if is_match else 1 - probability
    }


def find_best_match(model, target_image_path, candidate_dir, device, top_k=5):
    """在候选图像中找到最佳匹配
    
    Args:
        model: Siamese网络模型
        target_image_path: 目标图像路径
        candidate_dir: 候选图像目录
        device: 计算设备
        top_k: 返回前k个最佳匹配
    
    Returns:
        匹配结果列表
    """
    candidate_files = [f for f in os.listdir(candidate_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    results = []
    
    for candidate_file in candidate_files:
        candidate_path = os.path.join(candidate_dir, candidate_file)
        result = predict_pair(model, target_image_path, candidate_path, device, THRESHOLD)
        result['candidate'] = candidate_file
        results.append(result)
    
    # 按概率排序
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    return results[:top_k]


def batch_matching(model, pairs_file, device):
    """批量匹配预测
    
    Args:
        model: Siamese网络模型
        pairs_file: 图像对列表文件（每行：img1,img2）
        device: 计算设备
    
    Returns:
        匹配结果列表
    """
    results = []
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 2:
                img1_path = parts[0].strip()
                img2_path = parts[1].strip()
                
                result = predict_pair(model, img1_path, img2_path, device, THRESHOLD)
                result['image1'] = os.path.basename(img1_path)
                result['image2'] = os.path.basename(img2_path)
                results.append(result)
    
    return results


def extract_edge_curves(image_dir, output_dir, threshold=128):
    """批量提取碴口曲线
    
    Args:
        image_dir: 输入图像目录
        output_dir: 输出曲线图像目录
        threshold: 阈值分割阈值
    """
    os.makedirs(output_dir, exist_ok=True)
    
    from run_training import extract_edge_curve, curve_to_image
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        # 提取曲线
        curve = extract_edge_curve(img_path, threshold)
        
        if curve is not None:
            # 转换为图像
            curve_img = curve_to_image(curve, IMAGE_SIZE)
            
            if curve_img is not None:
                output_path = os.path.join(output_dir, img_file)
                Image.fromarray(curve_img).save(output_path)
                print(f"已处理: {img_file}")
    
    print(f"\n曲线提取完成! 结果保存在: {output_dir}")


def main():
    print(f"使用设备: {DEVICE}")
    
    # 加载模型
    model = load_model(MODEL_PATH, DEVICE)
    
    # 示例1: 单对预测
    # result = predict_pair(model, "curve1.png", "curve2.png", DEVICE)
    # print(f"图像1: curve1.png, 图像2: curve2.png")
    # print(f"可缀合: {result['can_combine']}, 概率: {result['probability']:.4f}")
    
    # 示例2: 查找最佳匹配
    # results = find_best_match(model, "target.png", "candidates/", DEVICE)
    # for r in results:
    #     print(f"候选: {r['candidate']}, 概率: {r['probability']:.4f}")
    
    # 示例3: 批量匹配
    # results = batch_matching(model, "pairs.txt", DEVICE)
    # for r in results:
    #     print(f"{r['image1']} + {r['image2']}: 可缀合={r['can_combine']}, 概率={r['probability']:.4f}")
    
    print("\n=== 使用说明 ===")
    print("1. 单对预测: predict_pair(model, 'img1.png', 'img2.png', DEVICE)")
    print("2. 查找最佳匹配: find_best_match(model, 'target.png', 'candidates/', DEVICE)")
    print("3. 批量匹配: batch_matching(model, 'pairs.txt', DEVICE)")
    print("4. 提取曲线: extract_edge_curves('input/', 'output/')")


if __name__ == "__main__":
    main()
