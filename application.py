import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import requests
import webbrowser
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 尝试导入 YOLOv8 (用于甲骨文检测)
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ 未安装 ultralytics，请运行: pip install ultralytics 以启用 YOLO 检测功能")
    sys.exit(1)

# ================= 默认配置与权重自动下载 =================
SWIN_WEIGHTS_PATH = "best_swin.pth"
SIAMESE_WEIGHTS_PATH = "best_siamese.pth"
GAN_WEIGHTS_PATH = "best_gan.pth"
YOLO_WEIGHTS_PATH = "inscription_detect.pt"

URLS = {
    SWIN_WEIGHTS_PATH: "https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_swin.pth",
    SIAMESE_WEIGHTS_PATH: "https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_siamese.pth",
    GAN_WEIGHTS_PATH: "https://github.com/luscious162/oracle_CCCC/releases/download/weight/best_gan.pth",
    YOLO_WEIGHTS_PATH: "https://github.com/luscious162/oracle_CCCC/releases/download/weight/inscription_detect.pt"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_weight(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1024 * 1024:
        return
    print(f"正在下载权重文件: {save_path} ...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=save_path, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        if os.path.exists(save_path): os.remove(save_path)
        print(f"下载失败: {e}，请检查网络或稍后重试。")
        sys.exit(1)

# 执行下载全部必要权重
for path, url in URLS.items():
    download_weight(url, path)


# ==================== 分类与缀合 模型定义 ====================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class VGG16Backbone(nn.Module):
    def __init__(self, in_channels=1):
        super(VGG16Backbone, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64), ConvBlock(64, 64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

class SiameseNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(SiameseNetwork, self).__init__()
        self.backbone = VGG16Backbone(in_channels)
        self.metric = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward_once(self, x): return self.backbone(x)
    def forward(self, x1, x2):
        f1, f2 = self.forward_once(x1), self.forward_once(x2)
        return self.metric(torch.abs(f1 - f2))

class SwinModel(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        base_model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        base_model.head = nn.Linear(base_model.head.in_features, 2)
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        self.model = base_model
        self.model.load_state_dict(state_dict, strict=False)
        self.norm = base_model.norm
    def forward(self, x): return self.model(x)

# ==================== GAN 摹本生成模型定义 ====================
# 基于你提供的 run_training.py 提取的结构
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64):
        super(UNetGenerator, self).__init__()
        # 编码器
        self.e1 = self.encoder_block(in_channels, ngf, normalize=False)
        self.e2 = self.encoder_block(ngf, ngf * 2)
        self.e3 = self.encoder_block(ngf * 2, ngf * 4)
        self.e4 = self.encoder_block(ngf * 4, ngf * 8)
        self.e5 = self.encoder_block(ngf * 8, ngf * 8)
        self.e6 = self.encoder_block(ngf * 8, ngf * 8)
        self.e7 = self.encoder_block(ngf * 8, ngf * 8)
        self.e8 = self.encoder_block(ngf * 8, ngf * 8, normalize=False)

        # 解码器
        self.d1 = self.decoder_block(ngf * 8, ngf * 8, dropout=True)
        self.d2 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.d3 = self.decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.d4 = self.decoder_block(ngf * 16, ngf * 8)
        self.d5 = self.decoder_block(ngf * 16, ngf * 4)
        self.d6 = self.decoder_block(ngf * 8, ngf * 2)
        self.d7 = self.decoder_block(ngf * 4, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def encoder_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)

    def decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout: layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], 1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], 1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], 1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], 1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], 1)

        return self.final(d7)


# ==================== 加载所有模型 ====================
print("正在加载模型到内存...")

# 1. Swin (分类)
swin_model = SwinModel(SWIN_WEIGHTS_PATH).to(DEVICE).eval()

# 2. Siamese (缀合)
siamese_model = SiameseNetwork(in_channels=3).to(DEVICE)
s_checkpoint = torch.load(SIAMESE_WEIGHTS_PATH, map_location=DEVICE)
siamese_model.load_state_dict(s_checkpoint['model_state_dict'])
siamese_model.eval()

# 3. GAN (摹本生成)
gan_model = UNetGenerator(in_channels=1, out_channels=1, ngf=64).to(DEVICE)
gan_checkpoint = torch.load(GAN_WEIGHTS_PATH, map_location=DEVICE)
# 处理由于保存格式不同导致的 state_dict 提取问题
if 'generator_state_dict' in gan_checkpoint:
    gan_model.load_state_dict(gan_checkpoint['generator_state_dict'])
else:
    gan_model.load_state_dict(gan_checkpoint)
gan_model.eval()

# 4. YOLOv8 (甲骨文检测)
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

print("模型加载完毕！")

# ==================== 工具函数与核心算法 ====================

def reshape_transform(tensor):
    if len(tensor.shape) == 4: return tensor.permute(0, 3, 1, 2)
    B, L, C = tensor.shape
    H = W = int(np.sqrt(L))
    return tensor.transpose(1, 2).reshape(B, C, H, W)

def cv2_to_base64(cv_img):
    _, buffer = cv2.imencode('.png', cv_img)
    return base64.b64encode(buffer).decode('utf-8')

# [ 此处保留原代码中所有的缀合算法: segment_to_image, siamese_predict, OracleSplicer ]
# 为控制篇幅且不丢失逻辑，将它们复用在此
def segment_to_image(segment, target_size=105):
    if segment is None or len(segment) < 2: return None
    segment = np.array(segment)
    min_x, min_y = segment.min(axis=0)
    max_x, max_y = segment.max(axis=0)
    width, height = max_x - min_x, max_y - min_y
    if width < 1 or height < 1: return None
    img = np.zeros((int(height) + 20, int(width) + 20), dtype=np.uint8)
    segment_shifted = segment - np.array([min_x, min_y]) + 10
    points = segment_shifted.astype(np.int32)
    cv2.polylines(img, [points], False, 255, 2)
    pil_img = Image.fromarray(img).resize((target_size, target_size), Image.LANCZOS).convert('L')
    return Image.merge('RGB', [pil_img, pil_img, pil_img])

def siamese_predict(model, seg1, seg2):
    if seg1 is None or seg2 is None: return float('inf')
    img1, img2 = segment_to_image(seg1), segment_to_image(seg2)
    if img1 is None or img2 is None: return float('inf')
    transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    tensor1 = transform(img1).unsqueeze(0).to(DEVICE)
    tensor2 = transform(img2).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probability = model(tensor1, tensor2).item()
    if probability < 0.5:
        return float('inf')
    return probability * 10000

class OracleSplicer:
    def __init__(self, img1, img2, siamese_net=None):
        self.img1 = img1
        self.img2 = img2
        self.siamese_net = siamese_net
        self.contour1 = None
        self.contour2 = None

    def get_largest_contour(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def segment_contour(self, contour):
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        points = contour.reshape(-1, 2)
        indices = [np.argmin(np.sqrt(np.sum((points - p) ** 2, axis=1))) for p in approx]
        unique_indices = sorted(list(set(indices)))
        if len(unique_indices) < 2: return [points]
        
        segments = []
        n = len(points)
        for i in range(len(unique_indices)):
            start = unique_indices[i]
            end = unique_indices[(i + 1) % len(unique_indices)]
            s, e = min(start, end), max(start, end)
            if (e - s) <= (n - e + s): segments.append(points[s:e + 1])
            else: segments.append(np.vstack((points[e:], points[:s + 1])))
        return segments

    def resample_segment(self, segment, num_points):
        if len(segment) < 2: return []
        lengths = np.sqrt(np.sum(np.diff(segment, axis=0) ** 2, axis=1))
        cumulative_lengths = np.insert(np.cumsum(lengths), 0, 0)
        if cumulative_lengths[-1] < 1e-6: return np.array([segment[0]] * num_points)
        new_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
        return np.vstack((np.interp(new_lengths, cumulative_lengths, segment[:, 0]),
                          np.interp(new_lengths, cumulative_lengths, segment[:, 1]))).T

    def compare_segments(self, seg1, seg2):
        resampled1 = self.resample_segment(seg1, 50)
        resampled2 = self.resample_segment(np.flip(seg2, axis=0), 50)
        if len(resampled1) == 0 or len(resampled2) == 0: return float('inf'), None
        c1, c2 = np.mean(resampled1, axis=0), np.mean(resampled2, axis=0)
        s1, s2 = resampled1 - c1, resampled2 - c2
        H = s1.T @ s2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c1 - R @ c2
        transform = np.hstack((R, t.reshape(2, 1)))
        transformed_s2 = (R @ s2.T).T
        error = np.sum((s1 - transformed_s2) ** 2)
        return error, transform

    def stitch(self, transform, anchor_segment, gap=5.0):
        def get_masked(img, cnt):
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            b, g, r = cv2.split(img)
            return cv2.merge([b, g, r, mask])
        masked1, masked2 = get_masked(self.img1, self.contour1), get_masked(self.img2, self.contour2)
        h1, w1 = masked1.shape[:2]
        h2, w2 = masked2.shape[:2]

        c1 = np.mean(anchor_segment, axis=0)
        p1 = cv2.perspectiveTransform(np.array([[c1]]), np.vstack((transform, [0, 0, 1])))[0][0]
        offset_vec = c1 - p1
        norm_vec = offset_vec / (np.linalg.norm(offset_vec) + 1e-6)

        M = np.copy(transform)
        M[0, 2] += norm_vec[0] * gap
        M[1, 2] += norm_vec[1] * gap

        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners2 = cv2.transform(np.array([corners2]), M)[0]
        all_corners = np.vstack((np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]), transformed_corners2))

        x_min, y_min = np.min(all_corners, axis=0)
        x_max, y_max = np.max(all_corners, axis=0)

        offset_transform = np.float32([[1, 0, -x_min], [0, 1, -y_min]])
        canvas_width, canvas_height = int(x_max - x_min), int(y_max - y_min)

        final_transform2 = offset_transform @ np.vstack((M, [0, 0, 1]))
        warped2 = cv2.warpAffine(masked2, final_transform2[:2, :], (canvas_width, canvas_height))
        canvas = cv2.warpAffine(masked1, offset_transform, (canvas_width, canvas_height))

        alpha2 = warped2[:, :, 3] / 255.0
        for c in range(0, 3): canvas[:, :, c] = canvas[:, :, c] * (1 - alpha2) + warped2[:, :, c] * alpha2
        canvas[:, :, 3] = np.maximum(canvas[:, :, 3], warped2[:, :, 3])
        return canvas

    def run(self):
        self.contour1, self.contour2 = self.get_largest_contour(self.img1), self.get_largest_contour(self.img2)
        if self.contour1 is None or self.contour2 is None: raise ValueError("无法提取有效边缘轮廓！")
        segments1, segments2 = self.segment_contour(self.contour1), self.segment_contour(self.contour2)
        best_match = {'score': float('inf'), 'seg1': None, 'transform': None}

        for seg1 in segments1:
            for seg2 in segments2:
                if len(seg1) < 10 or len(seg2) < 10: continue
                score, transform = self.compare_segments(seg1, seg2)
                if score >= 5000 and self.siamese_net is not None:
                    siamese_score = siamese_predict(self.siamese_net, seg1, seg2)
                    if siamese_score < score: score = siamese_score
                if score < best_match['score']: best_match = {'score': score, 'seg1': seg1, 'transform': transform}

        if best_match['seg1'] is None: raise ValueError("未找到可用的匹配边缘。")
        stitched_canvas = self.stitch(best_match['transform'], best_match['seg1'])
        return stitched_canvas, best_match['score']

# ==================== Flask Web 后端逻辑 ====================

app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>甲骨文智能分析全能系统</title>
    <style>
        :root {
            --gradient-1: #E0D68A;
            --gradient-2: #CB9173;
            --gradient-3: #8E443D;
            --gradient-4: #511730;
            --text-primary: #2D2D2D;
            --text-secondary: #5A5A5A;
            --surface: rgba(255, 255, 255, 0.92);
            --shadow-sm: 0 2px 8px rgba(81, 23, 48, 0.08);
            --shadow-md: 0 8px 32px rgba(81, 23, 48, 0.15);
            --shadow-lg: 0 16px 48px rgba(81, 23, 48, 0.2);
            --radius-sm: 8px;
            --radius-md: 16px;
            --radius-lg: 24px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
            background: 
                radial-gradient(ellipse at 20% 20%, var(--gradient-1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 30%, var(--gradient-2) 0%, transparent 50%),
                radial-gradient(ellipse at 60% 70%, var(--gradient-3) 0%, transparent 50%),
                radial-gradient(ellipse at 30% 90%, var(--gradient-4) 0%, transparent 50%),
                linear-gradient(135deg, #FDF6E3 0%, #F5E6D3 50%, #E8D5C4 100%);
            background-attachment: fixed;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--surface);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            padding: 40px;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
        }

        .header::after {
            content: '';
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            width: 80px;
            height: 4px;
            background: linear-gradient(90deg, var(--gradient-3), var(--gradient-4));
            border-radius: 2px;
        }

        .header h2 {
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--gradient-4) 0%, var(--gradient-3) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 4px;
            margin-bottom: 8px;
        }

        .header .subtitle {
            color: var(--text-secondary);
            font-size: 14px;
            letter-spacing: 2px;
        }

        .tabs {
            display: flex;
            border-bottom: 2px solid rgba(81, 23, 48, 0.1);
            margin-bottom: 32px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .tab {
            padding: 14px 28px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: var(--text-secondary);
            position: relative;
            border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        }

        .tab::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--gradient-3), var(--gradient-4));
            border-radius: 2px;
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .tab:hover {
            color: var(--gradient-3);
            background: rgba(81, 23, 48, 0.04);
        }

        .tab.active {
            color: var(--gradient-4);
            font-weight: 700;
        }

        .tab.active::after {
            transform: scaleX(1);
        }

        .panel {
            display: none;
            animation: fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .panel.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .carousel {
            position: relative;
            width: 100%;
            overflow: hidden;
            border-radius: var(--radius-md);
            background: linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.6));
            border: 1px solid rgba(81, 23, 48, 0.1);
            margin-bottom: 28px;
            box-shadow: inset 0 0 30px rgba(81, 23, 48, 0.03);
            touch-action: pan-y;
        }

        .carousel-inner {
            display: flex;
            transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .carousel-item {
            min-width: 100%;
            box-sizing: border-box;
            padding: 32px;
            text-align: center;
        }

        .carousel-item img {
            max-height: 220px;
            object-fit: contain;
            border-radius: var(--radius-sm);
            box-shadow: var(--shadow-sm);
            background: white;
            padding: 12px;
            margin: 0 12px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .carousel-item img:hover {
            transform: scale(1.02);
            box-shadow: var(--shadow-md);
        }

        .carousel-control {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(81, 23, 48, 0.15);
            color: var(--gradient-4);
            width: 44px;
            height: 44px;
            font-size: 20px;
            cursor: pointer;
            border-radius: 50%;
            z-index: 10;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-sm);
        }

        .carousel-control:hover {
            background: var(--gradient-4);
            color: white;
            box-shadow: var(--shadow-md);
            transform: translateY(-50%) scale(1.1);
        }

        .carousel-control.prev { left: 16px; }
        .carousel-control.next { right: 16px; }

        .carousel-title {
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--gradient-4);
            font-size: 16px;
            letter-spacing: 1px;
        }

        .upload-area {
            border: 2px dashed rgba(81, 23, 48, 0.25);
            padding: 32px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: var(--radius-md);
            background: linear-gradient(145deg, rgba(255,255,255,0.6), rgba(245,230,211,0.4));
        }

        .upload-area:hover {
            background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(245,230,211,0.7));
            border-color: var(--gradient-3);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .upload-area .icon {
            font-size: 36px;
            margin-bottom: 8px;
            display: block;
        }

        .btn-example {
            background: linear-gradient(135deg, var(--gradient-3) 0%, var(--gradient-4) 100%);
            color: white;
            border: none;
            padding: 16px 28px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            width: 100%;
            font-weight: 600;
            font-size: 15px;
            letter-spacing: 1px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 16px rgba(81, 23, 48, 0.25);
            margin-bottom: 24px;
        }

        .btn-example:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(81, 23, 48, 0.35);
            background: linear-gradient(135deg, var(--gradient-4) 0%, #3D1225 100%);
        }

        .action-btn {
            background: linear-gradient(135deg, var(--gradient-2) 0%, var(--gradient-3) 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 4px 16px rgba(142, 68, 61, 0.3);
            letter-spacing: 1px;
        }

        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(142, 68, 61, 0.4);
            background: linear-gradient(135deg, var(--gradient-3) 0%, var(--gradient-4) 100%);
        }

        button:disabled {
            background: #ccc !important;
            cursor: not-allowed !important;
            transform: none !important;
            box-shadow: none !important;
        }

        .results {
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            margin-top: 28px;
        }

        .res-card {
            flex: 1;
            min-width: 280px;
            border: 1px solid rgba(81, 23, 48, 0.1);
            padding: 24px;
            text-align: center;
            border-radius: var(--radius-md);
            background: linear-gradient(145deg, rgba(255,255,255,0.8), rgba(245,230,211,0.5));
            transition: all 0.3s ease;
        }

        .res-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-4px);
        }

        .res-card h4 {
            color: var(--gradient-4);
            font-size: 16px;
            margin-bottom: 16px;
            font-weight: 600;
            letter-spacing: 1px;
        }

        .res-card img {
            max-width: 100%;
            max-height: 450px;
            height: auto;
            border-radius: var(--radius-sm);
            box-shadow: var(--shadow-sm);
            background: white;
            padding: 8px;
        }

        .status {
            margin-bottom: 20px;
            font-weight: 600;
            min-height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid rgba(81, 23, 48, 0.1);
            border-top-color: var(--gradient-3);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .upload-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .upload-row .upload-area {
            flex: 1;
        }

        .label-tag {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }

        .label-bone {
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            color: #1565C0;
        }

        .label-shell {
            background: linear-gradient(135deg, #FCE4EC, #F8BBD9);
            color: #C2185B;
        }

        .score-display {
            background: linear-gradient(135deg, var(--gradient-1) 0%, var(--gradient-2) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 20px;
            font-weight: 700;
            margin-top: 16px;
        }

        @media (max-width: 768px) {
            .container { padding: 24px 16px; }
            .header h2 { font-size: 24px; }
            .tab { padding: 12px 16px; font-size: 14px; }
            .upload-row { flex-direction: column; }
            .results { flex-direction: column; }
            .res-card { min-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>🏛️ 甲骨文智能分析全能系统</h2>
            <p class="subtitle">ORACLE BONE INSCRIPTION INTELLIGENT ANALYSIS</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('classify', event)">📊 分类识别</button>
            <button class="tab" onclick="showTab('stitch', event)">🔗 碎片缀合</button>
            <button class="tab" onclick="showTab('gan', event)">🖋️ 摹本生成</button>
            <button class="tab" onclick="showTab('yolo', event)">🔎 甲骨文检测</button>
        </div>

        <div id="classify" class="panel active">
            <div class="carousel" id="cls-carousel" ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, 'cls')">
                <button class="carousel-control prev" onclick="moveCarousel('cls', -1)">&#10094;</button>
                <div class="carousel-inner" id="cls-track"></div>
                <button class="carousel-control next" onclick="moveCarousel('cls', 1)">&#10095;</button>
            </div>
            
            <button id="cls-btn-example" class="btn-example" onclick="runActiveExample('cls')">🧪 识别上方展示的在线示例图片</button>
            <div class="upload-area" onclick="document.getElementById('file-cls').click()">
                <span class="icon">📂</span>
                或点击此处上传本地拓片图片
                <input type="file" id="file-cls" hidden onchange="handleLocalFile(this, 'cls')">
            </div>
            <div id="cls-status" class="status"></div>
            
            <div class="results" id="cls-res" style="display:none">
                <div class="res-card">
                    <h4>📜 原始图像</h4>
                    <img id="img-orig">
                </div>
                <div class="res-card">
                    <h4>🔥 Grad-CAM 注意力热图</h4>
                    <img id="img-heat">
                    <p id="cls-text" style="font-weight:bold; margin-top:20px; font-size:18px;"></p>
                </div>
            </div>
        </div>

        <div id="stitch" class="panel">
            <div class="carousel" id="stitch-carousel" ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, 'stitch')">
                <button class="carousel-control prev" onclick="moveCarousel('stitch', -1)">&#10094;</button>
                <div class="carousel-inner" id="stitch-track"></div>
                <button class="carousel-control next" onclick="moveCarousel('stitch', 1)">&#10095;</button>
            </div>

            <button id="stitch-btn-example" class="btn-example" onclick="runActiveExample('stitch')">🧪 缀合上方展示的在线示例碎片对</button>
            <div class="upload-row">
                <div class="upload-area" onclick="document.getElementById('file-a').click()">
                    <span class="icon">📄</span>
                    上传本地碎片 A
                    <input type="file" id="file-a" hidden onchange="previewImg(this, 'pre-a')">
                    <br><img id="pre-a" style="margin-top:12px; display:none; max-height:140px; margin-left:auto; margin-right:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                </div>
                <div class="upload-area" onclick="document.getElementById('file-b').click()">
                    <span class="icon">📄</span>
                    上传本地碎片 B
                    <input type="file" id="file-b" hidden onchange="previewImg(this, 'pre-b')">
                    <br><img id="pre-b" style="margin-top:12px; display:none; max-height:140px; margin-left:auto; margin-right:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                </div>
            </div>
            <button id="stitch-btn" class="action-btn" onclick="handleStitch()">🚀 开始智能缀合 (本地图片)</button>
            <div id="stitch-status" class="status" style="margin-top:20px;"></div>
            
            <div class="res-card" id="stitch-res" style="display:none; margin-top:24px;">
                <h4>✨ 最终缀合结果</h4>
                <img id="img-stitched">
                <p id="stitch-score" class="score-display"></p>
            </div>
        </div>

        <div id="gan" class="panel">
            <div class="carousel" id="gan-carousel" ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, 'gan')">
                <button class="carousel-control prev" onclick="moveCarousel('gan', -1)">&#10094;</button>
                <div class="carousel-inner" id="gan-track"></div>
                <button class="carousel-control next" onclick="moveCarousel('gan', 1)">&#10095;</button>
            </div>
            
            <button id="gan-btn-example" class="btn-example" onclick="runActiveExample('gan')">🧪 生成上方展示拓片的摹本</button>
            <div class="upload-area" onclick="document.getElementById('file-gan').click()">
                <span class="icon">📂</span>
                或点击此处上传本地拓片图片
                <input type="file" id="file-gan" hidden onchange="handleLocalFile(this, 'gan')">
            </div>
            <div id="gan-status" class="status"></div>
            
            <div class="results" id="gan-res" style="display:none">
                <div class="res-card">
                    <h4>📜 原图</h4>
                    <img id="gan-orig">
                </div>
                <div class="res-card">
                    <h4>🎨 生成的摹本 (GAN)</h4>
                    <img id="gan-output">
                </div>
            </div>
        </div>

        <div id="yolo" class="panel">
            <div class="carousel" id="yolo-carousel" ontouchstart="handleTouchStart(event)" ontouchend="handleTouchEnd(event, 'yolo')">
                <button class="carousel-control prev" onclick="moveCarousel('yolo', -1)">&#10094;</button>
                <div class="carousel-inner" id="yolo-track"></div>
                <button class="carousel-control next" onclick="moveCarousel('yolo', 1)">&#10095;</button>
            </div>
            
            <button id="yolo-btn-example" class="btn-example" onclick="runActiveExample('yolo')">🧪 检测上方展示图片中的甲骨文</button>
            <div class="upload-area" onclick="document.getElementById('file-yolo').click()">
                <span class="icon">📂</span>
                或点击此处上传本地图片
                <input type="file" id="file-yolo" hidden onchange="handleLocalFile(this, 'yolo')">
            </div>
            <div id="yolo-status" class="status"></div>
            
            <div class="results" id="yolo-res" style="display:none; justify-content:center;">
                <div class="res-card" style="flex: 0 0 80%;">
                    <h4>🎯 YOLOv8 目标检测结果</h4>
                    <img id="yolo-output">
                </div>
            </div>
        </div>

    </div>

    <script>
        // ================= 在线素材数据与轮播逻辑 =================
        const baseUrl = "https://github.com/luscious162/oracle_CCCC/releases/download/weight/";
        const db = {
            cls: ['bone1.jpg', 'bone2.jpg', 'bone3.jpg', 'bone4.jpg', 'bone5.jpg', 'shell1.jpg', 'shell2.jpg', 'shell3.jpg', 'shell4.jpg', 'shell5.jpg'],
            stitch: [{a: 'rejion1_1.png', b: 'rejion1_2.png'}, {a: 'rejion2_1.png', b: 'rejion2_2.png'}, {a: 'rejion3_1.png', b: 'rejion3_2.png'}],
            gan: ['gan1.jpg', 'gan2.jpg', 'gan3.jpg'],
            yolo: ['yolo1.jpg', 'yolo2.jpg', 'yolo3.jpg']
        };

        let indices = { cls: 0, stitch: 0, gan: 0, yolo: 0 };

        window.onload = function() {
            // 初始化所有轮播图 HTML
            ['cls', 'gan', 'yolo'].forEach(type => {
                const track = document.getElementById(`${type}-track`);
                db[type].forEach(img => {
                    track.innerHTML += `<div class="carousel-item"><div class="carousel-title">素材名: ${img}</div><img src="${baseUrl}${img}"></div>`;
                });
            });

            const stitchTrack = document.getElementById('stitch-track');
            db.stitch.forEach((pair, idx) => {
                stitchTrack.innerHTML += `<div class="carousel-item"><div class="carousel-title">示例碎片对 ${idx + 1}</div><div style="display:flex; justify-content:center; gap:30px;"><div><img src="${baseUrl}${pair.a}"><br><small>${pair.a}</small></div><div><img src="${baseUrl}${pair.b}"><br><small>${pair.b}</small></div></div></div>`;
            });
        };

        function moveCarousel(type, dir) {
            indices[type] = (indices[type] + dir + db[type].length) % db[type].length;
            document.getElementById(`${type}-track`).style.transform = `translateX(-${indices[type] * 100}%)`;
        }

        // 滑动手势支持
        let startX = 0;
        function handleTouchStart(e) { startX = e.changedTouches[0].screenX; }
        function handleTouchEnd(e, type) {
            let endX = e.changedTouches[0].screenX;
            if (startX - endX > 40) moveCarousel(type, 1);
            if (endX - startX > 40) moveCarousel(type, -1);
        }

        function showTab(id, event) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(id).classList.add('active');
        }

        function previewImg(input, imgId) {
            const reader = new FileReader();
            reader.onload = e => { document.getElementById(imgId).src = e.target.result; document.getElementById(imgId).style.display = 'block'; };
            if(input.files[0]) reader.readAsDataURL(input.files[0]);
        }

        // ================= 统一请求发送器 =================
        async function runActiveExample(type) {
            const formData = new FormData();
            let urlEndpoint = '', statusMsg = '';
            
            if (type === 'stitch') {
                formData.append('url_a', baseUrl + db.stitch[indices.stitch].a);
                formData.append('url_b', baseUrl + db.stitch[indices.stitch].b);
                urlEndpoint = '/stitch';
                statusMsg = '正在下载碎片图并进行几何+网络联合配准，请稍候...';
            } else {
                formData.append('url', baseUrl + db[type][indices[type]]);
                if (type === 'cls') { urlEndpoint = '/predict'; statusMsg = '正在进行分类识别与热图生成...'; }
                if (type === 'gan') { urlEndpoint = '/generate_gan'; statusMsg = '正在通过 GAN 网络生成摹本...'; }
                if (type === 'yolo') { urlEndpoint = '/detect_yolo'; statusMsg = '正在通过 YOLOv8 检测甲骨文...'; }
            }
            await submitRequest(type, formData, urlEndpoint, statusMsg);
        }

        async function handleLocalFile(input, type) {
            if (!input.files[0]) return;
            const formData = new FormData();
            formData.append('file', input.files[0]);
            
            let urlEndpoint = '', statusMsg = '正在处理本地图像...';
            if (type === 'cls') urlEndpoint = '/predict';
            if (type === 'gan') urlEndpoint = '/generate_gan';
            if (type === 'yolo') urlEndpoint = '/detect_yolo';
            
            await submitRequest(type, formData, urlEndpoint, statusMsg);
        }

        async function handleStitch() {
            const fileA = document.getElementById('file-a').files[0], fileB = document.getElementById('file-b').files[0];
            if (!fileA || !fileB) return alert('请先上传本地的碎片 A 和碎片 B！');
            const formData = new FormData();
            formData.append('file_a', fileA); formData.append('file_b', fileB);
            await submitRequest('stitch', formData, '/stitch', '正在提取边缘并进行联合配准运算，请稍候...');
        }

        async function submitRequest(type, formData, endpoint, statusMsg) {
            const btnEx = document.getElementById(`${type}-btn-example`);
            const statusNode = document.getElementById(`${type}-status`);
            const resNode = document.getElementById(`${type}-res`);
            
            if(btnEx) btnEx.disabled = true;
            statusNode.innerHTML = `<span class="spinner"></span> ${statusMsg}`;
            statusNode.style.color = 'var(--text-secondary)';
            resNode.style.display = 'none'; 
            resNode.classList.remove('fade-in');

            try {
                const res = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                statusNode.innerHTML = '✨ 处理完成';
                statusNode.style.color = 'var(--gradient-3)';
                resNode.style.display = 'flex'; 
                void resNode.offsetWidth; 
                resNode.classList.add('fade-in');
                
                // 渲染不同类型的返回结果
                if (type === 'cls') {
                    document.getElementById('img-orig').src = 'data:image/png;base64,' + data.original;
                    document.getElementById('img-heat').src = 'data:image/png;base64,' + data.heatmap;
                    const labelClass = data.label === 'Bone' ? 'label-bone' : 'label-shell';
                    document.getElementById('cls-text').innerHTML = `预测: <span class="label-tag ${labelClass}">${data.label}</span> <br><span style="font-size:14px; color:#888; margin-top:8px; display:block;">Bone: ${(data.prob_bone*100).toFixed(1)}% | Shell: ${(data.prob_shell*100).toFixed(1)}%</span>`;
                } else if (type === 'stitch') {
                    document.getElementById('img-stitched').src = 'data:image/png;base64,' + data.result;
                    document.getElementById('stitch-score').innerText = `🏆 最终匹配质量得分: ${data.score.toFixed(2)}`;
                } else if (type === 'gan') {
                    document.getElementById('gan-orig').src = 'data:image/png;base64,' + data.original;
                    document.getElementById('gan-output').src = 'data:image/png;base64,' + data.muben;
                } else if (type === 'yolo') {
                    document.getElementById('yolo-output').src = 'data:image/png;base64,' + data.result;
                }
            } catch (e) {
                statusNode.innerHTML = '❌ 失败: ' + e; 
                statusNode.style.color = '#C62828';
            } finally {
                if(btnEx) btnEx.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ---------- 路由 1: 分类识别 ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
        elif 'url' in request.form:
            img_bytes = requests.get(request.form['url']).content
        else: return jsonify({'error': '未提供图片'})

        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = swin_model(input_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            pred = output.argmax(dim=1).item()
        
        target_layers = [swin_model.norm]
        targets = [ClassifierOutputTarget(pred)]
        rgb_img = np.float32(pil_img.resize((224, 224))) / 255
        with GradCAMPlusPlus(model=swin_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        return jsonify({'label': 'Bone' if pred == 0 else 'Shell', 'prob_bone': float(probs[0]), 'prob_shell': float(probs[1]), 'original': cv2_to_base64(img_bgr), 'heatmap': cv2_to_base64(cam_image)})
    except Exception as e: return jsonify({'error': str(e)}), 500

# ---------- 路由 2: 碎片缀合 ----------
@app.route('/stitch', methods=['POST'])
def stitch():
    try:
        if 'file_a' in request.files and 'file_b' in request.files and request.files['file_a'].filename != '':
            bytes_a, bytes_b = request.files['file_a'].read(), request.files['file_b'].read()
        elif 'url_a' in request.form and 'url_b' in request.form:
            bytes_a, bytes_b = requests.get(request.form['url_a']).content, requests.get(request.form['url_b']).content
        else: return jsonify({'error': '缺少碎片'})

        img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

        splicer = OracleSplicer(img_a, img_b, siamese_net=siamese_model)
        stitched_canvas, final_score = splicer.run()
        return jsonify({'result': cv2_to_base64(stitched_canvas), 'score': float(final_score), 'success': True})
    except Exception as e: return jsonify({'error': str(e)})

# ---------- 路由 3: GAN 摹本生成 ----------
@app.route('/generate_gan', methods=['POST'])
def generate_gan():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
        elif 'url' in request.form:
            img_bytes = requests.get(request.form['url']).content
        else: return jsonify({'error': '未提供图片'})

        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 转换为单通道灰度图传入 GAN
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        
        # 数据转换
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] 归一化
        ])
        
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output_tensor = gan_model(input_tensor)
            # 反归一化并转回 numpy 图像格式
            output_tensor = output_tensor * 0.5 + 0.5
            output_numpy = output_tensor.squeeze().cpu().numpy()
            output_numpy = (output_numpy * 255).astype(np.uint8)
        
        return jsonify({
            'original': cv2_to_base64(img_bgr),
            'muben': cv2_to_base64(output_numpy)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

# ---------- 路由 4: YOLOv8 甲骨文检测 ----------
@app.route('/detect_yolo', methods=['POST'])
def detect_yolo():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
        elif 'url' in request.form:
            img_bytes = requests.get(request.form['url']).content
        else: return jsonify({'error': '未提供图片'})

        # 解码为 BGR numpy 数组用于 cv2 和 YOLO 输入
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 使用 YOLO 进行预测
        results = yolo_model(img_bgr)
        
        # plot() 方法会将检测框画到原图上并返回一个带有框的 numpy array (BGR)
        res_plotted = results[0].plot()

        return jsonify({
            'result': cv2_to_base64(res_plotted)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"🚀 服务已启动，自动打开浏览器: {url}")
    webbrowser.open(url)
    app.run(host='127.0.0.1', port=port, debug=False)