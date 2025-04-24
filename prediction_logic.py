import tensorflow as tf
# from tensorflow import keras # 不再直接需要 keras
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_preprocess # 预处理逻辑可能需要调整或确认
import numpy as np
import cv2
import os
import re
import time

# --- 配置常量 (保持不变) ---
DEFAULT_TFLITE_MODEL_PATH = "oracle_detector_model.tflite" # 使用 TFLite 文件名
DEFAULT_CLASSES = ['animal_bone', 'tortoise_shell']
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
DEFAULT_PATCH_HEIGHT = 160 # 需要与 TFLite 模型输入尺寸一致
DEFAULT_PATCH_WIDTH = 160  # 需要与 TFLite 模型输入尺寸一致
DEFAULT_CONTOUR_THICKNESS = 3

# --- TFLite 模型加载函数 ---
def load_tflite_model(model_path=DEFAULT_TFLITE_MODEL_PATH):
    """
    加载 TFLite 模型并分配张量。
    返回 (interpreter, error_message)
    """
    print(f"--- [TFLite 模型加载] 尝试加载模型: {model_path} ---")
    if not os.path.exists(model_path):
        err = f"错误：找不到 TFLite 模型文件 {model_path}"
        print(err)
        return None, err
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors() # 非常重要：分配内存
        print("  [TFLite 模型加载] 模型加载并分配张量成功。")
        # (可选) 打印输入输出细节
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"  [TFLite 模型加载] 输入细节: {input_details}")
        print(f"  [TFLite 模型加载] 输出细节: {output_details}")
        return interpreter, None
    except Exception as e:
        err = f"加载 TFLite 模型时出错: {e}"
        print(err)
        return None, err

# --- 分割函数 (保持不变) ---
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
    # ... (分割函数的代码保持不变) ...
    # --- 查找轮廓 ---
    print("  [分割函数] 查找轮廓...")
    try:
        if not os.path.exists(image_path):
            print(f"错误：[分割函数] 测试文件路径不存在 {image_path}")
            return [], None, f"图像文件不存在 {image_path}" # 返回错误信息
        n = np.fromfile(image_path, np.uint8)
        img_gray = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"错误：[分割函数] 无法解码灰度图像 {image_path}")
            return [], None, f"无法解码灰度图像 {os.path.basename(image_path)}"
        print(f"  [分割函数] 灰度图像加载成功, shape: {img_gray.shape}")
        img_color = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img_color is None:
            print("  [分割函数] 无法解码彩色图像，将从灰度图转换。")
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img_gray.shape[:2]
        print(f"  [分割函数] 彩色图像加载/转换成功, shape: {img_color.shape}")
    except Exception as e:
        err = f"读取或解码图像时发生错误 {image_path}: {e}"
        print(f"错误：[分割函数] {err}")
        return [], None, err

    # --- 阈值处理 ---
    print("  [分割函数] 应用阈值处理...")
    thresh_val, binary_mask = cv2.threshold(img_gray, 0, 255, threshold_type)
    print(f"  [分割函数] 阈值处理完成, Otsu阈值: {thresh_val}")

    # --- 形态学闭运算 ---
    if closing_kernel_size > 0 and closing_iterations > 0:
        print(f"  [分割函数] 应用闭运算 (kernel={closing_kernel_size}, iter={closing_iterations})...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # --- 形态学开运算 ---
    if opening_kernel_size > 0 and opening_iterations > 0:
        print(f"  [分割函数] 应用开运算 (kernel={opening_kernel_size}, iter={opening_iterations})...")
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=opening_iterations)

    # --- 查找轮廓 ---
    print("  [分割函数] 查找轮廓...")
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  [分割函数] 找到 {len(contours)} 个初始轮廓。")

    rubbing_info_list = []
    if contours:
        print(f"  [分割函数] 开始过滤轮廓 (min_area={min_area_threshold}, max_letter_wh={max_letter_width}/{max_letter_height}, border={border_margin})...")
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
            rubbing_info_list.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h), 'id': i + 1}) # 添加原始ID
        print(f"  [分割函数] 轮廓过滤完成. 保留: {len(rubbing_info_list)}, 过滤掉: {filtered_out_count}")

    print(f"--- [分割函数] 返回 {len(rubbing_info_list)} 个有效轮廓 ---")
    return rubbing_info_list, img_color, None # 返回彩色图，即使没有轮廓, 无错误


# --- 核心预测与绘制函数 (使用 TFLite Interpreter) ---
def predict_rubbings(image_path,
                     interpreter, # 传入已加载的 TFLite Interpreter
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
    对单个图像进行拓片检测、分类(使用TFLite)并返回结果。
    返回 (predictions_api_list, result_image_with_boxes or None, error_message or None)
    """
    print(f"\n--- [核心预测 TFLite] 开始处理图像: {image_path} ---")
    if interpreter is None:
        return [], None, "TFLite Interpreter 未加载"

    # --- 1. 查找拓片轮廓 ---
    try:
        rubbing_infos, original_image_color, error_msg = find_rubbings_and_calculate_area_for_prediction(
            image_path, threshold_type, closing_kernel_size, closing_iterations,
            opening_kernel_size, opening_iterations, min_area_threshold,
            max_area_threshold, max_letter_height, max_letter_width,
            border_margin, max_relative_size
        )
        if error_msg:
             return [], None, f"分割时出错: {error_msg}"
        if original_image_color is None:
            return [], None, "分割函数未能返回图像"
        if not rubbing_infos:
            print("  [核心预测 TFLite] 未找到符合条件的轮廓。")
            return [], original_image_color, None # Not an error, just no detections

    except Exception as e:
        err = f"查找轮廓时发生意外错误: {e}"
        print(f"  错误: {err}")
        return [], None, err

    print(f"  [核心预测 TFLite] 找到 {len(rubbing_infos)} 个候选拓片进行分类。")
    result_image_final = original_image_color.copy() # 创建副本用于绘制
    predictions_api_list = []
    kept_predictions_count = 0
    kept_rubbing_details = [] # 用于绘制

    # --- 获取 TFLite 输入/输出细节 ---
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # 假设模型只有一个输入和一个输出
        input_shape = input_details[0]['shape'] # e.g., [1, 160, 160, 3]
        input_dtype = input_details[0]['dtype'] # e.g., float32
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        # 验证输入尺寸是否匹配
        if input_shape[1] != patch_height or input_shape[2] != patch_width:
             print(f"警告: 配置的尺寸 ({patch_height}x{patch_width}) 与 TFLite 模型输入尺寸 ({input_shape[1]}x{input_shape[2]}) 不匹配！")
             # 可以选择报错退出，或者强制使用模型尺寸
             # patch_height = input_shape[1]
             # patch_width = input_shape[2]
    except Exception as e:
        err = f"获取 TFLite 输入/输出细节时出错: {e}"
        print(f"  错误: {err}")
        return [], original_image_color, err


    # --- 2. 遍历、处理、预测 ---
    for i, info in enumerate(rubbing_infos):
        print(f"  --- 处理候选 {info.get('id', i+1)}/{len(rubbing_infos)} ---") # 使用ID
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
            if crop_y_end <= crop_y_start or crop_x_end <= crop_x_start: continue
            rubbing_crop_masked = masked_full_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            if rubbing_crop_masked.size == 0: continue

            # --- 预处理 (需要与模型训练和转换时一致!) ---
            patch_resized = cv2.resize(rubbing_crop_masked, (patch_width, patch_height), interpolation=cv2.INTER_AREA)

            # !! 关键：这里的预处理必须匹配模型 !!
            # 假设 TFLite 模型需要 float32 输入，且归一化到 [-1, 1] (类似 MobileNetV2)
            # 如果您的模型需要不同的预处理（例如 0 到 1），需要修改这里！
            patch_processed = patch_resized.astype(input_dtype) # 使用模型期望的 dtype
            patch_processed = (patch_processed / 127.5) - 1.0 # 归一化到 [-1, 1]
            # patch_processed = patch_processed / 255.0 # 或者归一化到 [0, 1]

            patch_batch = np.expand_dims(patch_processed, axis=0) # 添加 batch 维度

            # --- TFLite 预测 ---
            print(f"    预测中 (TFLite)...")
            interpreter.set_tensor(input_index, patch_batch) # 设置输入张量
            interpreter.invoke() # 运行推理
            predictions = interpreter.get_tensor(output_index)[0] # 获取输出张量 (移除 batch 维度)

            predicted_class_index = np.argmax(predictions)
            predicted_class_name = classes[predicted_class_index]
            confidence = float(predictions[predicted_class_index]) # 转为 Python float
            print(f"    结果: {predicted_class_name} (置信度: {confidence:.4f})")

            # --- 过滤与记录 ---
            if confidence >= min_confidence_threshold:
                print(f"    保留 (置信度 >= {min_confidence_threshold})")
                kept_predictions_count += 1
                bbox_int = [int(coord) for coord in (x, y, x+w, y+h)]
                predictions_api_list.append({
                    "box": bbox_int,
                    "label": predicted_class_name,
                    "score": confidence
                })
                # 存储用于绘制的信息
                kept_rubbing_details.append({
                    'box': bbox_int,
                    'label': predicted_class_name,
                    'score': confidence,
                    'id': info.get('id', i + 1) # 使用原始ID
                })
            else:
                print(f"    已过滤 (置信度 < {min_confidence_threshold})")

        except Exception as e:
            print(f"  错误: 处理/预测拓片 {info.get('id', i+1)} 时出错: {e}")
            continue # Skip to next contour

    # --- 绘制结果 ---
    result_image_final = draw_results_on_image(original_image_color, kept_rubbing_details, contour_thickness)

    print(f"--- [核心预测 TFLite] 处理完成. 保留了 {kept_predictions_count} 个预测。 ---")
    return predictions_api_list, result_image_final, None # 返回结果列表, 带框的图像, 无错误信息


# --- 绘制结果函数 (从 predict_rubbings 中分离出来) ---
def draw_results_on_image(image, prediction_details, contour_thickness=DEFAULT_CONTOUR_THICKNESS):
    """在图像上绘制检测结果 (使用 prediction_details 列表)"""
    result_image = image.copy()
    if not prediction_details: # 如果没有预测结果，直接返回副本
        return result_image

    for pred in prediction_details:
        box = pred['box'] # [x_min, y_min, x_max, y_max]
        label = pred['label']
        score = pred['score']
        pred_id = pred.get('id', '?') # 获取ID

        color = (0, 0, 255) if label == 'animal_bone' else (255, 0, 0) # BGR: 红色 vs 蓝色

        # 绘制边界框
        cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), color, contour_thickness)

        # 准备标签文本
        label_text = f"{pred_id}:{label[0]}({score:.2f})" # ID:首字母(分数)

        # 计算文本大小以放置背景
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # 决定文本位置 (优先放在框上方，如果空间不足则放在下方)
        text_y = box[1] - 10
        rect_y = text_y - text_height - baseline
        if rect_y < 0: # 如果上方空间不足
            text_y = box[3] + text_height + baseline
            rect_y = box[3] + baseline

        # 绘制文本背景
        cv2.rectangle(result_image, (box[0], rect_y), (box[0] + text_width, text_y + baseline), color, -1)
        # 绘制文本 (白色)
        cv2.putText(result_image, label_text, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return result_image


# --- 主预测流程 (现在仅用于独立测试脚本) ---
if __name__ == "__main__":
    print("--- 独立脚本测试流程开始 (TFLite) ---")
    # 使用文件顶部的默认常量进行测试
    test_image = r"E:/oracle_talent/oracle_data/tokyo_data/27-41.jpg" # 或者选择一个实际存在的测试图片
    test_output_dir = r"./test_output_tflite" # 保存到当前目录下的 test_output_tflite

    # 检查测试文件是否存在
    if not os.path.exists(test_image):
        exit(f"错误：用于独立测试的图像文件不存在: {test_image}")

    os.makedirs(test_output_dir, exist_ok=True)

    # 1. 加载 TFLite 模型
    interpreter_instance, err = load_tflite_model(DEFAULT_TFLITE_MODEL_PATH)
    if err:
        exit(f"独立测试失败: {err}")

    # 2. 执行预测
    start_time = time.time()
    predictions, result_img, err_msg = predict_rubbings(
        test_image,
        interpreter_instance, # 传入 interpreter
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
    print(f"预测耗时: {time.time() - start_time:.2f} 秒")

    if err_msg:
        print(f"独立测试中预测出错: {err_msg}")
    else:
        print("\n--- 独立测试预测结果 (TFLite) ---")
        if predictions:
            for i, p in enumerate(predictions):
                print(f"  {i+1}: Label={p['label']}, Score={p['score']:.2f}, Box={p['box']}")
        else:
            print("  未检测到目标。")

        if result_img is not None:
            try:
                output_path = os.path.join(test_output_dir, "tflite_test_result_" + os.path.basename(test_image))
                is_success, im_buf_arr = cv2.imencode('.jpg', result_img)
                if is_success:
                    with open(output_path, 'wb') as f: im_buf_arr.tofile(f)
                    print(f"\n测试结果图像已保存到: {output_path}")
                else: print("错误：无法编码测试结果图像。")
            except Exception as e:
                print(f"保存测试结果时出错: {e}")
        else:
            print("预测函数未返回结果图像。")

    print("\n--- 独立脚本测试流程结束 (TFLite) ---")

