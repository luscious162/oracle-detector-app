import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import requests
import base64
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify
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

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64):
        super(UNetGenerator, self).__init__()
        self.e1 = self.encoder_block(in_channels, ngf, normalize=False)
        self.e2 = self.encoder_block(ngf, ngf * 2)
        self.e3 = self.encoder_block(ngf * 2, ngf * 4)
        self.e4 = self.encoder_block(ngf * 4, ngf * 8)
        self.e5 = self.encoder_block(ngf * 8, ngf * 8)
        self.e6 = self.encoder_block(ngf * 8, ngf * 8)
        self.e7 = self.encoder_block(ngf * 8, ngf * 8)
        self.e8 = self.encoder_block(ngf * 8, ngf * 8, normalize=False)

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

swin_model = SwinModel(SWIN_WEIGHTS_PATH).to(DEVICE).eval()

siamese_model = SiameseNetwork(in_channels=3).to(DEVICE)
s_checkpoint = torch.load(SIAMESE_WEIGHTS_PATH, map_location=DEVICE)
siamese_model.load_state_dict(s_checkpoint['model_state_dict'])
siamese_model.eval()

gan_model = UNetGenerator(in_channels=1, out_channels=1, ngf=64).to(DEVICE)
gan_checkpoint = torch.load(GAN_WEIGHTS_PATH, map_location=DEVICE)
if 'generator_state_dict' in gan_checkpoint:
    gan_model.load_state_dict(gan_checkpoint['generator_state_dict'])
else:
    gan_model.load_state_dict(gan_checkpoint)
gan_model.eval()

yolo_model = YOLO(YOLO_WEIGHTS_PATH)

print("模型加载完毕！")

# ==================== 工具函数 ====================

def reshape_transform(tensor):
    if len(tensor.shape) == 4: return tensor.permute(0, 3, 1, 2)
    B, L, C = tensor.shape
    H = W = int(np.sqrt(L))
    return tensor.transpose(1, 2).reshape(B, C, H, W)

def cv2_to_base64(cv_img):
    _, buffer = cv2.imencode('.png', cv_img)
    return base64.b64encode(buffer).decode('utf-8')

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


# ==================== Flask API 服务器 ====================

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        'message': 'Oracle Bone API Server',
        'version': '2.0',
        'endpoints': {
            '/predict': 'POST - 分类识别（骨片/龟甲）',
            '/stitch': 'POST - 碎片缀合',
            '/generate_gan': 'POST - GAN 摹本生成',
            '/detect_yolo': 'POST - YOLOv8 甲骨文检测'
        }
    })

# ---------- 路由 1: 分类识别 ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
        elif 'url' in request.form:
            img_bytes = requests.get(request.form['url']).content
        else:
            return jsonify({'error': '未提供图片'})

        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

        return jsonify({
            'label': 'Bone' if pred == 0 else 'Shell',
            'prob_bone': float(probs[0]),
            'prob_shell': float(probs[1]),
            'original': cv2_to_base64(img_bgr),
            'heatmap': cv2_to_base64(cam_image)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- 路由 2: 碎片缀合 ----------
@app.route('/stitch', methods=['POST'])
def stitch():
    try:
        if 'file_a' in request.files and 'file_b' in request.files and request.files['file_a'].filename != '':
            bytes_a, bytes_b = request.files['file_a'].read(), request.files['file_b'].read()
        elif 'url_a' in request.form and 'url_b' in request.form:
            bytes_a = requests.get(request.form['url_a']).content
            bytes_b = requests.get(request.form['url_b']).content
        else:
            return jsonify({'error': '缺少碎片'})

        img_a = cv2.imdecode(np.frombuffer(bytes_a, np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(bytes_b, np.uint8), cv2.IMREAD_COLOR)

        splicer = OracleSplicer(img_a, img_b, siamese_net=siamese_model)
        stitched_canvas, final_score = splicer.run()
        return jsonify({
            'result': cv2_to_base64(stitched_canvas),
            'score': float(final_score),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------- 路由 3: GAN 摹本生成 ----------
@app.route('/generate_gan', methods=['POST'])
def generate_gan():
    try:
        if 'file' in request.files and request.files['file'].filename != '':
            img_bytes = request.files['file'].read()
        elif 'url' in request.form:
            img_bytes = requests.get(request.form['url']).content
        else:
            return jsonify({'error': '未提供图片'})

        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_tensor = gan_model(input_tensor)
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
        else:
            return jsonify({'error': '未提供图片'})

        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        results = yolo_model(img_bgr)
        res_plotted = results[0].plot()

        return jsonify({
            'result': cv2_to_base64(res_plotted)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Flask server port')
    args = parser.parse_args()
    
    print("🚀 API 服务器启动成功！")
    print(f"📡 后端运行在: http://127.0.0.1:{args.port}")
    app.run(host='127.0.0.1', port=args.port, debug=False)
