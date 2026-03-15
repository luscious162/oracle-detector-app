# 甲骨文智能分析全能系统

## 快速启动

### 一键启动（推荐）

```bash
cd /Users/Zhuanz/Desktop/oracle/application_web
python run.py
```

这将自动：
1. 检查并安装前端依赖（如需要）
2. 启动 Flask 后端 API 服务器
3. 启动 React 前端开发服务器

启动成功后访问：**http://localhost:3000**

### 手动启动

如果需要分别启动：

```bash
cd /Users/Zhuanz/Desktop/oracle/application_web

# 终端 1: 启动后端
python app.py

# 终端 2: 启动前端
npm run dev
```

## 项目结构

```
application_web/
├── run.py                 # 一键启动脚本
├── app.py                 # Flask API 后端服务器
├── package.json           # React 项目配置
├── vite.config.js        # Vite 配置
├── index.html            # HTML 入口
└── src/
    ├── main.jsx          # React 入口
    ├── App.jsx           # 路由配置
    ├── index.css         # 全局样式
    ├── pages/
    │   ├── Landing.jsx   # 落地页（带 MeshGradient 动画）
    │   └── Main.jsx      # 主界面（带 MeshGradient 背景）
    └── components/
        ├── Classify.jsx  # 分类识别
        ├── Stitch.jsx    # 碎片缀合
        ├── Gan.jsx      # GAN 摹本生成
        └── Yolo.jsx     # YOLOv8 甲骨文检测
```

## 功能说明

- **分类识别**: 使用 Swin Transformer + Grad-CAM 进行骨片/龟甲分类
- **碎片缀合**: 使用 Siamese Network + 几何配准进行甲骨碎片缀合
- **摹本生成**: 使用 GAN 生成甲骨文摹本
- **甲骨文检测**: 使用 YOLOv8 检测图片中的甲骨文

## 注意事项

- 确保 Python 环境中已安装所需依赖:
  ```bash
  pip install flask flask-cors torch torchvision opencv-python pillow tqdm requests ultralytics pytorch-grad-cam
  ```
- 确保已安装 Node.js（https://nodejs.org/）
- 首次运行时会自动下载模型权重文件
