# 目标检测系统

这是一个基于 Flask 和 Vue 3 的目标检测系统，用于上传图像并检测其中的对象。

## 项目结构

```
├── app.py                   # Flask 后端应用
├── requirements.txt         # Python 依赖列表
├── src/                     # Vue 前端源代码
│   ├── App.vue              # 主组件
│   └── main.js              # Vue 应用入口
├── public/                  # 静态文件
└── uploads/                 # 临时上传文件存储目录
```

## 后端设置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行后端服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 上启动。

## 前端设置

### 安装依赖

```bash
npm install
```

### 运行开发服务器

```bash
npm run serve
```

前端应用将在 `http://localhost:8080` 上启动。

### 构建生产版本

```bash
npm run build
```

## 使用说明

1. 启动后端和前端服务
2. 打开浏览器访问 `http://localhost:8080`
3. 点击或拖拽图片到上传区域
4. 点击"开始检测"按钮
5. 系统将显示检测结果，在图片上标记出边界框和类别标签

## API 接口

### 目标检测接口

- **URL**: `/api/detect`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **参数**:
  - `image`: 要分析的图像文件

#### 成功响应 (200 OK)

```json
[
  {
    "box": [x_min, y_min, x_max, y_max],
    "label": "object_label",
    "score": 0.95
  },
  ...
]
```

#### 错误响应

- 400 Bad Request: 如果文件无效或缺失
- 500 Internal Server Error: 如果处理过程中发生错误

## 注意事项

- 确保 `masked_patch_classifier_final.keras` 模型文件位于项目根目录
- 支持的图像格式: JPG, JPEG, PNG 