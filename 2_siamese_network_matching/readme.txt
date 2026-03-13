Siamese 孪生网络缀合模型 - 文件说明
==================================

本模块实现甲骨缀合匹配，基于"碴口曲线（边缘轮廓）"的相似度匹配。

【核心算法】
- Siamese孪生网络：共享权重的双分支网络
- VGG16骨干网络：提取碴口曲线深度特征
- 曼哈顿距离：度量特征相似性
- 多曲线融合：针对样本匮乏的数据增强策略

【文件结构】

1. model.py
   - Siamese孪生网络模型定义
   - 包含：VGG16Backbone(VGG16骨干)、SiameseNetwork(孪生网络)
   - 用途：定义缀合匹配网络结构

2. run_training.py
   - 模型训练脚本
   - 包含：
     * 碴口曲线提取函数(extract_edge_curve, normalize_curve, curve_to_image)
     * 多曲线融合数据增强(multi_curve_fusion)
     * 对比损失函数(ContrastiveLoss)
     * 训练/验证流程
   - 使用方法：修改DATA_PATH为数据集路径后运行
   - 数据集目录结构：
     train/positive/ - 可缀合的图像对列表(.txt文件)
     train/negative/ - 不可缀合的图像对列表(.txt文件)
     val/positive/   - 验证正样本
     val/negative/   - 验证负样本

3. inference.py
   - 模型推理脚本
   - 功能：
     * 单对图像缀合预测
     * 在候选图像中查找最佳匹配
     * 批量匹配预测
     * 碴口曲线批量提取
   - 辅助函数：extract_edge_curves（批量曲线提取）

4. readme.txt
   - 本文件，简要说明各文件功能

【数据集要求】
- 输入图像：105x105的碴口曲线灰度图像
- 图像格式：PNG/JPG
- 训练数据：正样本（可缀合对）、负样本（不可缀合对）

【使用方法】
1. 准备数据集：
   - 提取碴口曲线（参考论文方法：阈值分割、边缘检测、规范化）
   - 生成105x105的曲线图像
   - 准备正负样本对列表文件

2. 修改run_training.py中的DATA_PATH

3. 运行训练：python run_training.py

4. 使用inference.py进行推理：
   - 单对预测：predict_pair()
   - 最佳匹配：find_best_match()
   - 批量匹配：batch_matching()

【核心流程】
1. 数据预处理：阈值分割、边缘寻找、规范化算法提取碴口曲线
2. 曲线转图像：生成105x105的曲线图像
3. 网络前向：双分支提取特征 -> 曼哈顿距离度量 -> Sigmoid输出概率
4. 缀合判定：概率>0.5判定为可缀合
