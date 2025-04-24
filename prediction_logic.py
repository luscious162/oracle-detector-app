import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_preprocess
import numpy as np
import cv2
import os
import re

# --- 配置 (可以从调用方传入或在此处定义默认值) ---
DEFAULT_MODEL_PATH = "masked_patch_classifier_final.keras"
DEFAULT_CLASSES = ['animal_bone', 'tortoise_shell']

# --- 分割和过滤参数 (默认值) ---
DEFAULT_THRESHOLD_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
DEFAULT_CLOSING_KERNEL_SIZE = 5
DEFAULT_CLOSING_ITERATIONS = 2
DEFAULT_OPENING_KERNEL_SIZE = 3
DEFAULT_OPENING_ITERATIONS = 1
DEFAULT_MIN_AREA = 4000
DEFAULT_MAX_AREA_THRESHOLD = 10000000
DEFAULT_MAX_LETTER_HEIGHT = 50
DEFAULT_MAX_LETTER_WIDTH = 50
DEFAULT_BORDER_MARGIN = 15
DEFAULT_MAX_RELATIVE_SIZE = 0.9
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.5

# --- 分类模型输入尺寸 ---
DEFAULT_PATCH_HEIGHT = 160
DEFAULT_PATCH_WIDTH = 160

# --- 分割函数 ---
def find_rubbings_and_calculate_area_for_prediction(image_path,
                                     threshold_type=DEFAULT_THRESHOLD_TYPE,
                                     closing_kernel_size=DEFAULT_CLOSING_KERNEL_SIZE,
                                     closing_iterations=DEFAULT_CLOSING_ITERATIONS,
                                     opening_kernel_size=DEFAULT_OPENING_KERNEL_SIZE,
                                     opening_iterations=DEFAULT_OPENING_ITERATIONS,
                                     min_area_threshold=DEFAULT_MIN_AREA,
                                     max_area_threshold=DEFAULT_MAX_AREA_THRESHOLD,
                                     max_letter_height=DEFAULT_MAX_LETTER_HEIGHT,
                                     max_letter_width=DEFAULT_MAX_LETTER_WIDTH,
                                     border_margin=DEFAULT_BORDER_MARGIN,
                                     max_relative_size=DEFAULT_MAX_RELATIVE_SIZE):
    """
    查找拓片轮廓，用于预测。
    """
    try:
        if not os.path.exists(image_path):
             print(f"错误：测试文件路径不存在 {image_path}")
             return [], None, None # 返回空列表、None图像、错误消息
        n = np.fromfile(image_path, np.uint8)
        img_gray = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"错误：无法解码灰度图像 {image_path}")
            return [], None, f"无法解码灰度图像 {os.path.basename(image_path)}"
        img_color = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img_color is None:
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img_gray.shape[:2]

    except Exception as e:
        print(f"读取或解码图像时发生错误 {image_path}: {e}")
        return [], None, f"读取或解码图像时发生错误: {e}"

    # --- 1. 全局阈值处理 (Otsu) ---
    _, binary_mask = cv2.threshold(img_gray, 0, 255, threshold_type)

    # --- 2. 形态学闭运算 ---
    if closing_kernel_size > 0 and closing_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # --- 3. 形态学开运算 ---
    if opening_kernel_size > 0 and opening_iterations > 0:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=opening_iterations)

    # --- 4. 查找轮廓 ---
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rubbing_info_list = []
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold or area > max_area_threshold:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < max_letter_height or w < max_letter_width:
                continue
            if x < border_margin or y < border_margin or \
               (x + w) > (img_width - border_margin) or \
               (y + h) > (img_height - border_margin):
                continue
            if w > img_width * max_relative_size or h > img_height * max_relative_size:
                continue

            rubbing_info_list.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})

    # 返回轮廓信息、原始彩色图像和 None (表示无错误)
    return rubbing_info_list, img_color, None

# --- Keras 模型预处理函数 ---
def tf_keras_preprocess_input(img):
    img_float = img.astype(np.float32)
    return model_preprocess(img_float)

# --- 加载 Keras 模型 ---
def load_keras_model(model_path=DEFAULT_MODEL_PATH):
    try:
        if not os.path.exists(model_path):
            print(f"错误：模型文件未找到: {model_path}")
            return None, f"模型文件未找到: {model_path}"
        print(f"开始加载 Keras 模型: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Keras 模型加载成功。")
        return model, None
    except Exception as e:
        print(f"加载 Keras 模型时出错: {e}")
        return None, f"加载 Keras 模型时出错: {e}"

# --- 核心预测函数 ---
def predict_rubbings(image_path, model,
                     classes=DEFAULT_CLASSES,
                     patch_height=DEFAULT_PATCH_HEIGHT,
                     patch_width=DEFAULT_PATCH_WIDTH,
                     min_confidence_threshold=DEFAULT_MIN_CONFIDENCE_THRESHOLD,
                     # 传递分割参数
                     threshold_type=DEFAULT_THRESHOLD_TYPE,
                     closing_kernel_size=DEFAULT_CLOSING_KERNEL_SIZE,
                     closing_iterations=DEFAULT_CLOSING_ITERATIONS,
                     opening_kernel_size=DEFAULT_OPENING_KERNEL_SIZE,
                     opening_iterations=DEFAULT_OPENING_ITERATIONS,
                     min_area_threshold=DEFAULT_MIN_AREA,
                     max_area_threshold=DEFAULT_MAX_AREA_THRESHOLD,
                     max_letter_height=DEFAULT_MAX_LETTER_HEIGHT,
                     max_letter_width=DEFAULT_MAX_LETTER_WIDTH,
                     border_margin=DEFAULT_BORDER_MARGIN,
                     max_relative_size=DEFAULT_MAX_RELATIVE_SIZE):
    """
    对给定图像中的拓片进行检测和分类。
    返回预测结果列表和带有绘制结果的图像。
    """
    if model is None:
        return None, None, "模型未加载"

    # --- 1. 查找轮廓 ---
    rubbing_infos, original_image_color, error_msg = find_rubbings_and_calculate_area_for_prediction(
        image_path,
        threshold_type=threshold_type,
        closing_kernel_size=closing_kernel_size,
        closing_iterations=closing_iterations,
        opening_kernel_size=opening_kernel_size,
        opening_iterations=opening_iterations,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        max_letter_height=max_letter_height,
        max_letter_width=max_letter_width,
        border_margin=border_margin,
        max_relative_size=max_relative_size
    )
    if error_msg:
        return None, None, error_msg
    if not rubbing_infos:
        print("未能检测到符合条件的拓片轮廓")
        # 即使没有轮廓，也返回原始图像以便显示
        return [], original_image_color, None

    # --- 2. 准备预测 ---
    all_predictions_api_format = [] # 用于API返回
    kept_rubbing_details = [] # 用于绘制

    # --- 3. 遍历、掩码、预处理和预测 ---
    for i, info in enumerate(rubbing_infos):
        contour = info['contour']
        x, y, w, h = info['bbox']

        # 应用掩码
        mask = np.zeros(original_image_color.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        masked_rubbing = cv2.bitwise_and(original_image_color, original_image_color, mask=mask)
        rubbing_crop_masked = masked_rubbing[y:y+h, x:x+w]

        if rubbing_crop_masked.size == 0:
            continue

        # 预处理
        patch_resized = cv2.resize(rubbing_crop_masked, (patch_width, patch_height), interpolation=cv2.INTER_AREA)
        patch_processed = tf_keras_preprocess_input(patch_resized)
        patch_batch = np.expand_dims(patch_processed, axis=0)

        # 预测
        try:
            predictions = model.predict(patch_batch)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = classes[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])

            # 置信度过滤
            if confidence >= min_confidence_threshold:
                # API 格式: [x_min, y_min, x_max, y_max]
                box_api = [int(x), int(y), int(x+w), int(y+h)]
                all_predictions_api_format.append({
                    "box": box_api,
                    "label": predicted_class_name,
                    "score": confidence
                })
                # 存储用于绘制的信息
                kept_rubbing_details.append({
                    'box': box_api, # 使用API格式的box
                    'label': predicted_class_name,
                    'score': confidence,
                    'id': i + 1 # 保留原始ID用于绘制标签
                })

        except Exception as e:
            print(f"处理拓片 {i+1} 时预测出错: {e}")
            # 可以选择记录错误或跳过此拓片
            continue

    # --- 4. 绘制结果 --- (移到单独的函数)
    result_image_with_boxes = draw_results_on_image(original_image_color, kept_rubbing_details)

    return all_predictions_api_format, result_image_with_boxes, None

# --- 绘制结果函数 ---
def draw_results_on_image(image, prediction_details):
    """在图像上绘制检测结果 (使用 prediction_details 列表)"""
    result_image = image.copy()
    contour_thickness = 2 # 可以设为参数

    for pred in prediction_details:
        box = pred['box'] # [x_min, y_min, x_max, y_max]
        label = pred['label']
        score = pred['score']
        pred_id = pred['id']

        # 设置颜色 - 动物骨（红色）/ 龟甲（蓝色）
        color = (0, 0, 255) if label == 'animal_bone' else (255, 0, 0)

        # 绘制边界框
        cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), color, contour_thickness)

        # 绘制标签 (使用原始ID)
        label_text = f"{pred_id}:{label[0]}({score:.2f})"
        text_x = box[0]
        text_y = box[1] - 10 if box[1] > 20 else box[3] + 15 # 调整标签位置避免出界
        cv2.putText(result_image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return result_image

# --- (可选) 主程序块，用于独立测试 prediction_logic.py ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # --- 测试配置 ---
    TEST_IMAGE_PATH = r"E:/oracle_talent/oracle_data/微信图片_20250421181946.png" # 使用你的测试图片路径
    TEST_OUTPUT_DIR = r"E:\oracle_agent\result_logic_test"
    TEST_MODEL_PATH = DEFAULT_MODEL_PATH

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    print("--- 开始独立测试 prediction_logic.py ---")

    # 1. 加载模型
    start_time = time.time()
    model, error_msg = load_keras_model(TEST_MODEL_PATH)
    if error_msg:
        exit(f"模型加载失败: {error_msg}")
    print(f"模型加载耗时: {time.time() - start_time:.2f} 秒")

    # 2. 进行预测
    if not os.path.exists(TEST_IMAGE_PATH):
        exit(f"错误：测试图片不存在 '{TEST_IMAGE_PATH}'")

    print(f"\n开始预测图片: {TEST_IMAGE_PATH}")
    start_time = time.time()
    predictions, result_image, error_msg = predict_rubbings(
        TEST_IMAGE_PATH,
        model,
        # 可以覆盖默认参数进行测试
        # min_area_threshold=5000,
        # min_confidence_threshold=0.8
    )
    print(f"预测耗时: {time.time() - start_time:.2f} 秒")

    if error_msg:
        exit(f"预测失败: {error_msg}")

    # 3. 显示和保存结果
    print(f"\n预测完成，找到 {len(predictions)} 个符合条件的拓片。")
    if predictions:
        print("预测结果 (API 格式):")
        for p in predictions:
            print(f"  Label: {p['label']}, Score: {p['score']:.4f}, Box: {p['box']}")
    else:
        print("没有找到符合置信度和面积阈值的拓片。")

    if result_image is not None:
        # 显示图像
        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'预测结果 (找到 {len(predictions)} 个)')
        plt.axis('off')
        plt.ion()  # 启用交互模式
        plt.draw()
        plt.pause(0.1)  # 暂停一小段时间以确保图像显示
        plt.show(block=True)  # 阻塞显示图像

        # 保存图像
        try:
            base_filename = os.path.basename(TEST_IMAGE_PATH)
            safe_basename = re.sub(r'[\\/*?:"]<>|]', '_', base_filename)
            output_filename = os.path.join(TEST_OUTPUT_DIR, "logic_test_result_" + safe_basename)
            # 确保使用支持中文路径的保存方式
            is_success, im_buf_arr = cv2.imencode(os.path.splitext(output_filename)[1], result_image)
            if is_success:
                im_buf_arr.tofile(output_filename)
                print(f"结果图像已保存到: {output_filename}")
            else:
                print(f"错误：无法编码结果图像以保存到 {output_filename}")
        except Exception as e:
            print(f"保存结果图像时出错: {e}")
    else:
        print("未能生成结果图像。")

    print("--- prediction_logic.py 独立测试结束 ---")