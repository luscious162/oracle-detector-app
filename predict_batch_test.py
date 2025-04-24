import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_preprocess
import numpy as np
import cv2
import os
import re
import time

# --- 配置常量 (移到顶部，并添加默认值) ---
DEFAULT_MODEL_PATH = "masked_patch_classifier_final.keras" # 默认模型路径
DEFAULT_CLASSES = ['animal_bone', 'tortoise_shell']        # 默认类别
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
DEFAULT_PATCH_HEIGHT = 160
DEFAULT_PATCH_WIDTH = 160
DEFAULT_CONTOUR_THICKNESS = 3

# --- 模型加载函数 (供 app.py 调用) ---
def load_keras_model(model_path=DEFAULT_MODEL_PATH):
    """
    加载 Keras 模型。
    返回 (model, error_message)
    """
    print(f"--- [模型加载] 尝试加载模型: {model_path} ---")
    if not os.path.exists(model_path):
        err = f"错误：找不到模型文件 {model_path}"
        print(err)
        return None, err
    try:
        # 可选: 设置 TF 日志级别 (如果需要在 Flask 环境下控制)
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # tf.get_logger().setLevel('ERROR')
        model = keras.models.load_model(model_path)
        print("  [模型加载] 模型加载成功。")
        return model, None
    except Exception as e:
        err = f"加载 Keras 模型时出错: {e}"
        print(err)
        return None, err


# --- 分割函数 (保持不变，但确认它能被导入) ---
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
    查找拓片轮廓，用于预测。使用与训练时相同的参数。
    返回 (rubbing_info_list, original_color_image or None)
    """
    print(f"--- [分割函数] 尝试处理: {image_path} ---")
    try:
        if not os.path.exists(image_path):
             print(f"错误：[分割函数] 测试文件路径不存在 {image_path}")
             return [], None
        n = np.fromfile(image_path, np.uint8)
        img_gray = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"错误：[分割函数] 无法解码灰度图像 {image_path}")
            return [], None
        print(f"  [分割函数] 灰度图像加载成功, shape: {img_gray.shape}")
        img_color = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img_color is None:
            print("  [分割函数] 无法解码彩色图像，将从灰度图转换。")
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img_gray.shape[:2]
        print(f"  [分割函数] 彩色图像加载/转换成功, shape: {img_color.shape}")
    except Exception as e:
        print(f"错误：[分割函数] 读取或解码图像时发生错误 {image_path}: {e}")
        return [], None

    # --- 阈值处理 ---
    print("  [分割函数] 应用阈值处理...")
    thresh_val, binary_mask = cv2.threshold(img_gray, 0, 255, threshold_type)
    print(f"  [分割函数] 阈值处理完成, Otsu阈值: {thresh_val}")

    # --- 形态学闭运算 ---
    if closing_kernel_size > 0 and closing_iterations > 0:
        print(f"  [分割函数] 应用闭运算 (kernel={closing_kernel_size}, iter={closing_iterations})...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # --- 形态学开运算 ---
    if opening_kernel_size > 0 and opening_iterations > 0:
        print(f"  [分割函数] 应用开运算 (kernel={opening_kernel_size}, iter={opening_iterations})...")
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=opening_iterations)

    # --- 查找轮廓 ---
    print("  [分割函数] 查找轮廓...")
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  [分割函数] 找到 {len(contours)} 个初始轮廓。")

    rubbing_info_list = []
    if contours:
        print(f"  [分割函数] 开始过滤轮廓 (min_area={min_area_threshold}, max_letter_wh={max_letter_width}/{max_letter_height}, border={border_margin})...")
        filtered_out_count = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # --- 应用所有过滤器 ---
            if not (min_area_threshold <= area <= max_area_threshold): filtered_out_count += 1; continue
            if h < max_letter_height or w < max_letter_width: filtered_out_count += 1; continue
            if x < border_margin or y < border_margin or \
               (x + w) > (img_width - border_margin) or \
               (y + h) > (img_height - border_margin): filtered_out_count += 1; continue
            if w > img_width * max_relative_size or h > img_height * max_relative_size: filtered_out_count += 1; continue
            # Passed filters
            rubbing_info_list.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
        print(f"  [分割函数] 轮廓过滤完成. 保留: {len(rubbing_info_list)}, 过滤掉: {filtered_out_count}")

    print(f"--- [分割函数] 返回 {len(rubbing_info_list)} 个有效轮廓 ---")
    return rubbing_info_list, img_color # 返回彩色图，即使没有轮廓


# --- 核心预测与绘制函数 (供 app.py 调用) ---
def predict_rubbings(image_path,
                     model, # 传入已加载的模型
                     classes=DEFAULT_CLASSES,
                     patch_height=DEFAULT_PATCH_HEIGHT,
                     patch_width=DEFAULT_PATCH_WIDTH,
                     min_confidence_threshold=DEFAULT_MIN_CONFIDENCE_THRESHOLD,
                     # Pass segmentation params through
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
                     max_relative_size=DEFAULT_MAX_RELATIVE_SIZE,
                     contour_thickness=DEFAULT_CONTOUR_THICKNESS):
    """
    对单个图像进行拓片检测、分类并返回结果。
    返回 (predictions_api_list, result_image_with_boxes or None, error_message or None)
    """
    print(f"\n--- [核心预测] 开始处理图像: {image_path} ---")
    if model is None:
        return [], None, "模型未加载"

    # --- 1. 查找拓片轮廓 ---
    try:
        rubbing_infos, original_image_color = find_rubbings_and_calculate_area_for_prediction(
            image_path, threshold_type, closing_kernel_size, closing_iterations,
            opening_kernel_size, opening_iterations, min_area_threshold,
            max_area_threshold, max_letter_height, max_letter_width,
            border_margin, max_relative_size
        )
        if original_image_color is None:
             # find_rubbings.. should have printed error, but handle just in case
             return [], None, "加载或处理图像时出错"
        if not rubbing_infos:
            print("  [核心预测] 未找到符合条件的轮廓。")
            # 返回空结果和原始图像（无框）
            return [], original_image_color, None # Not an error, just no detections

    except Exception as e:
        err = f"查找轮廓时发生意外错误: {e}"
        print(f"  错误: {err}")
        return [], None, err

    print(f"  [核心预测] 找到 {len(rubbing_infos)} 个候选拓片进行分类。")
    result_image_final = original_image_color.copy() # 创建副本用于绘制
    predictions_api_list = []
    kept_predictions_count = 0

    # --- 2. 遍历、处理、预测 ---
    for i, info in enumerate(rubbing_infos):
        print(f"  --- 处理候选 {i+1}/{len(rubbing_infos)} ---")
        contour = info['contour']
        x, y, w, h = info['bbox']

        try:
            # --- 应用掩码和裁剪 ---
            mask = np.zeros(original_image_color.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            masked_full_image = cv2.bitwise_and(original_image_color, original_image_color, mask=mask)
            crop_y_end = min(y + h, original_image_color.shape[0])
            crop_x_end = min(x + w, original_image_color.shape[1])
            crop_y_start = max(0, y)
            crop_x_start = max(0, x)
            if crop_y_end <= crop_y_start or crop_x_end <= crop_x_start: continue # Skip if bbox invalid
            rubbing_crop_masked = masked_full_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            if rubbing_crop_masked.size == 0: continue

            # --- 预处理 ---
            patch_resized = cv2.resize(rubbing_crop_masked, (patch_width, patch_height), interpolation=cv2.INTER_AREA)
            patch_processed = patch_resized.astype('float32')
            patch_processed = model_preprocess(patch_processed) # Specific to model
            patch_batch = np.expand_dims(patch_processed, axis=0)

            # --- 预测 ---
            print(f"    预测中...")
            predictions = model.predict(patch_batch, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = classes[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index]) # Convert numpy float to python float
            print(f"    结果: {predicted_class_name} (置信度: {confidence:.4f})")

            # --- 过滤与记录 ---
            if confidence >= min_confidence_threshold:
                print(f"    保留 (置信度 >= {min_confidence_threshold})")
                kept_predictions_count += 1
                # 确保 bbox 坐标是 Python int 类型
                bbox_int = [int(coord) for coord in (x, y, x+w, y+h)]
                predictions_api_list.append({
                    "box": bbox_int, # [x_min, y_min, x_max, y_max]
                    "label": predicted_class_name,
                    "score": confidence
                })

                # --- 绘制 ---
                color = (255, 0, 0) if predicted_class_name == 'animal_bone' else (0, 0, 255)
                cv2.drawContours(result_image_final, [contour], -1, color, contour_thickness)
                label_text = f"{predicted_class_name[0]}:{confidence:.2f}" # Label: ClassInitial:Score
                text_x = x
                text_y = y - 10 if y > 20 else y + h + 15
                cv2.putText(result_image_final, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                print(f"    已过滤 (置信度 < {min_confidence_threshold})")

        except Exception as e:
            print(f"  错误: 处理/预测拓片 {i+1} 时出错: {e}")
            continue # Skip to next contour

    print(f"--- [核心预测] 处理完成. 保留了 {kept_predictions_count} 个预测。 ---")
    return predictions_api_list, result_image_final, None # 返回结果列表, 带框的图像, 无错误信息


# --- 主预测流程 (现在仅用于独立测试脚本) ---
if __name__ == "__main__":
    print("--- 独立脚本测试流程开始 ---")
    # 使用文件顶部的默认常量进行测试
    test_image = r"E:/oracle_talent/oracle_data/tokyo_data/27-41.jpg" # 或者选择一个实际存在的测试图片
    test_output_dir = r"./test_output" # 保存到当前目录下的 test_output

    # 检查测试文件是否存在
    if not os.path.exists(test_image):
        exit(f"错误：用于独立测试的图像文件不存在: {test_image}")

    os.makedirs(test_output_dir, exist_ok=True)

    # 1. 加载模型
    model_instance, err = load_keras_model(DEFAULT_MODEL_PATH)
    if err:
        exit(f"独立测试失败: {err}")

    # 2. 执行预测
    predictions, result_img, err_msg = predict_rubbings(
        test_image,
        model_instance,
        classes=DEFAULT_CLASSES,
        patch_height=DEFAULT_PATCH_HEIGHT,
        patch_width=DEFAULT_PATCH_WIDTH,
        min_confidence_threshold=DEFAULT_MIN_CONFIDENCE_THRESHOLD,
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
        max_relative_size=DEFAULT_MAX_RELATIVE_SIZE,
        contour_thickness=DEFAULT_CONTOUR_THICKNESS
    )

    if err_msg:
        print(f"独立测试中预测出错: {err_msg}")
    else:
        print("\n--- 独立测试预测结果 ---")
        if predictions:
            for i, p in enumerate(predictions):
                print(f"  {i+1}: Label={p['label']}, Score={p['score']:.2f}, Box={p['box']}")
        else:
            print("  未检测到目标。")

        if result_img is not None:
            try:
                output_path = os.path.join(test_output_dir, "test_result_" + os.path.basename(test_image))
                is_success, im_buf_arr = cv2.imencode('.jpg', result_img)
                if is_success:
                    with open(output_path, 'wb') as f: im_buf_arr.tofile(f)
                    print(f"\n测试结果图像已保存到: {output_path}")
                else: print("错误：无法编码测试结果图像。")
                # 尝试显示图像 (可选，可能在某些服务器环境无效)
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(12,12))
                # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                # plt.title("独立测试结果")
                # plt.axis('off')
                # plt.show()
            except Exception as e:
                print(f"保存或显示测试结果时出错: {e}")
        else:
            print("预测函数未返回结果图像。")

    print("\n--- 独立脚本测试流程结束 ---")