<template>
  <div class="app-container">
    <header class="header">
      <h1>目标检测系统</h1>
    </header>
    
    <main class="main-content">
      <div class="upload-section">
        <div 
          class="drop-zone" 
          :class="{ active: isDragging }" 
          @dragover.prevent="handleDragOver" 
          @dragleave.prevent="handleDragLeave" 
          @drop.prevent="handleDrop"
          @click="triggerFileInput"
        >
          <input 
            ref="fileInput" 
            type="file" 
            accept="image/jpeg,image/png,image/jpg" 
            class="file-input" 
            @change="handleFileSelect" 
          />
          <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
          </div>
          <p>点击或拖拽图片到这里上传</p>
          <p class="file-types">支持 JPG, JPEG, PNG 格式</p>
        </div>
        
        <button 
          class="detect-button" 
          :disabled="!imageSelected || isLoading" 
          @click="detectObjects"
        >
          <i class="fas fa-search"></i> 开始检测
        </button>
      </div>
      
      <div class="preview-section" v-if="imageSelected || isLoading || detectionResults.length > 0">
        <div class="preview-container" ref="previewContainer">
          <div class="loading-overlay" v-if="isLoading">
            <div class="spinner"></div>
            <p>正在处理中...</p>
          </div>
          
          <img 
            v-if="previewUrl" 
            :src="previewUrl" 
            alt="预览图片" 
            class="preview-image" 
            ref="previewImage"
            @load="imageLoaded"
          />
          
          <canvas 
            v-if="detectionResults.length > 0 && !isLoading" 
            ref="detectionCanvas" 
            class="detection-canvas"
          ></canvas>
        </div>
        
        <div class="results-summary" v-if="detectionResults.length > 0 && !isLoading">
          <h3>检测结果 ({{ detectionResults.length }})</h3>
          <ul class="results-list">
            <li v-for="(result, index) in detectionResults" :key="index" class="result-item">
              <span class="result-label">{{ result.label }}</span>
              <span class="result-score">置信度: {{ (result.score * 100).toFixed(2) }}%</span>
            </li>
          </ul>
        </div>
      </div>
    </main>

    <div class="error-message" v-if="errorMessage">
      <div class="error-content">
        <i class="fas fa-exclamation-circle"></i>
        <p>{{ errorMessage }}</p>
        <button @click="errorMessage = ''">关闭</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      imageFile: null,
      previewUrl: null,
      isDragging: false,
      isLoading: false,
      imageSelected: false,
      imageWidth: 0,
      imageHeight: 0,
      detectionResults: [],
      errorMessage: ''
    }
  },
  methods: {
    // 文件上传处理方法
    triggerFileInput() {
      this.$refs.fileInput.click();
    },
    handleDragOver(e) {
      this.isDragging = true;
    },
    handleDragLeave() {
      this.isDragging = false;
    },
    handleDrop(e) {
      this.isDragging = false;
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.processFile(files[0]);
      }
    },
    handleFileSelect(e) {
      const files = e.target.files;
      if (files.length > 0) {
        this.processFile(files[0]);
      }
    },
    processFile(file) {
      // 检查文件类型
      if (!file.type.match('image/jpeg') && !file.type.match('image/png') && !file.type.match('image/jpg')) {
        this.showError('请上传 JPG、JPEG 或 PNG 格式的图片');
        return;
      }

      // 清除之前的结果
      this.detectionResults = [];
      
      // 设置新文件和预览
      this.imageFile = file;
      this.previewUrl = URL.createObjectURL(file);
      this.imageSelected = true;
    },
    
    // 图像加载完成时调用
    imageLoaded() {
      if (this.$refs.previewImage) {
        this.imageWidth = this.$refs.previewImage.naturalWidth;
        this.imageHeight = this.$refs.previewImage.naturalHeight;
      }
    },
    
    // 目标检测
    async detectObjects() {
      if (!this.imageFile) return;
      
      this.isLoading = true;
      this.detectionResults = [];
      this.errorMessage = '';
      
      // 创建表单数据
      const formData = new FormData();
      formData.append('image', this.imageFile);
      
      try {
        // 发送请求到后端 API
        const response = await fetch('http://localhost:5000/api/detect', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || '检测请求失败');
        }
        
        // 处理响应
        const results = await response.json();
        this.detectionResults = results;
        
        // 在图像上绘制检测结果
        this.$nextTick(() => {
          this.drawDetectionResults();
        });
      } catch (error) {
        this.showError(`检测失败: ${error.message}`);
        console.error('检测错误:', error);
      } finally {
        this.isLoading = false;
      }
    },
    
    // 在画布上绘制检测结果
    drawDetectionResults() {
      const canvas = this.$refs.detectionCanvas;
      const img = this.$refs.previewImage;
      
      if (!canvas || !img || this.detectionResults.length === 0) return;
      
      // 设置画布大小与图像相同
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // 计算缩放比例
      const scaleX = img.clientWidth / this.imageWidth;
      const scaleY = img.clientHeight / this.imageHeight;
      
      // 绘制每个检测框
      this.detectionResults.forEach((result, index) => {
        const [x, y, x2, y2] = result.box;
        const label = result.label;
        const score = result.score;
        
        // 根据类别设置不同颜色
        let color;
        if (label === 'animal_bone') {
          color = 'rgba(255, 0, 0, 0.8)'; // 红色
        } else if (label === 'tortoise_shell') {
          color = 'rgba(0, 0, 255, 0.8)'; // 蓝色
        } else {
          color = 'rgba(0, 255, 0, 0.8)'; // 绿色（默认）
        }
        
        // 缩放坐标
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = (x2 - x) * scaleX;
        const scaledHeight = (y2 - y) * scaleY;
        
        // 绘制边界框
        ctx.lineWidth = 3;
        ctx.strokeStyle = color;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // 绘制标签背景
        ctx.fillStyle = color;
        const text = `${index+1}: ${label.charAt(0)}(${(score * 100).toFixed(0)}%)`;
        const textWidth = ctx.measureText(text).width + 10;
        ctx.fillRect(scaledX, scaledY - 30, textWidth, 30);
        
        // 绘制标签文本
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(text, scaledX + 5, scaledY - 10);
      });
    },
    
    // 显示错误信息
    showError(message) {
      this.errorMessage = message;
      // 5秒后自动关闭
      setTimeout(() => {
        if (this.errorMessage === message) {
          this.errorMessage = '';
        }
      }, 5000);
    }
  }
}
</script>

<style>
/* 导入字体图标 */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');

/* 基础样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Arial', sans-serif;
  background-color: #f5f8fa;
  color: #333;
  line-height: 1.6;
}

.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* 页头 */
.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 2.5rem;
  color: #2c3e50;
}

/* 主要内容区域 */
.main-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

@media (min-width: 992px) {
  .main-content {
    flex-direction: row;
  }
  
  .upload-section {
    width: 40%;
  }
  
  .preview-section {
    width: 60%;
  }
}

/* 上传区域 */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.drop-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.drop-zone:hover, .drop-zone.active {
  border-color: #3498db;
  background-color: rgba(52, 152, 219, 0.05);
}

.upload-icon {
  font-size: 3rem;
  color: #3498db;
  margin-bottom: 15px;
}

.file-input {
  display: none;
}

.file-types {
  font-size: 0.8rem;
  color: #7f8c8d;
  margin-top: 10px;
}

.detect-button {
  padding: 12px 24px;
  background-color: #2ecc71;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.detect-button:hover {
  background-color: #27ae60;
}

.detect-button:disabled {
  background-color: #95a5a6;
  cursor: not-allowed;
}

/* 预览区域 */
.preview-section {
  background-color: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}

.preview-container {
  position: relative;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background-color: #f0f0f0;
}

.preview-image {
  max-width: 100%;
  max-height: 600px;
  display: block;
}

.detection-canvas {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}

/* 加载状态 */
.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #3498db;
  animation: spin 1s linear infinite;
  margin-bottom: 10px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 结果摘要区域 */
.results-summary {
  padding: 20px;
  border-top: 1px solid #eee;
}

.results-summary h3 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.results-list {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.result-item {
  background-color: #f8f9fa;
  border-radius: 4px;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.result-label {
  font-weight: bold;
}

.result-score {
  color: #7f8c8d;
  font-size: 0.9rem;
}

/* 错误消息 */
.error-message {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background-color: #e74c3c;
  color: white;
  padding: 15px 20px;
  border-radius: 8px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
  z-index: 100;
  max-width: 90%;
  animation: slideUp 0.3s ease-out;
}

.error-content {
  display: flex;
  align-items: center;
  gap: 15px;
}

.error-content i {
  font-size: 1.5rem;
}

.error-content button {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.5);
  color: white;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: auto;
}

.error-content button:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translate(-50%, 20px);
  }
  to {
    opacity: 1;
    transform: translate(-50%, 0);
  }
}
</style> 