# -*- coding: utf-8 -*-

print("--- 脚本开始执行 (顶层) ---")

# --- Import essential modules for early checks FIRST ---
import os
import sys
print(f"Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

# --- Other Imports (within a try-except block) ---
try:
    print("尝试导入 TensorFlow...")
    import tensorflow as tf
    print(f"TensorFlow 版本: {tf.__version__}")

    print("尝试导入 Keras...")
    from tensorflow import keras
    print("Keras 导入成功 (来自 TensorFlow)")

    print("尝试导入 MobileNetV2 preprocess_input...")
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_preprocess
    print("MobileNetV2 preprocess_input 导入成功")

    print("尝试导入 NumPy...")
    import numpy as np
    print("NumPy 导入成功")

    print("尝试导入 OpenCV...")
    import cv2
    print(f"OpenCV 版本: {cv2.__version__}")

    # --- Matplotlib Import with Agg Backend ---
    print("尝试配置并导入 Matplotlib...")
    import matplotlib
    matplotlib.use('Agg') # Set backend *before* importing pyplot
    import matplotlib.pyplot as plt
    print("Matplotlib 导入成功 (使用 Agg 后端)")
    # --- End Matplotlib Import ---

    print("尝试导入 re...")
    import re
    print("re 导入成功")

    print("尝试导入 time...")
    import time
    print("time 导入成功")

    print("--- 所有库导入成功 ---") # Check if this gets printed now

except ImportError as e:
    print(f"!!! 库导入失败: {e} !!!")
    exit(f"错误：必要的库未能导入 - {e}")
except Exception as e:
    print(f"!!! 导入过程中发生未知错误: {e} !!!")
    # import traceback
    # traceback.print_exc() # Uncomment for detailed stack trace if needed
    exit(f"错误：导入时发生异常 - {e}")


# --- 分割函数 (完整定义) ---
def find_rubbings_and_calculate_area_for_prediction(image_path,
                                     threshold_type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                                     closing_kernel_size=5, closing_iterations=2,
                                     opening_kernel_size=3, opening_iterations=1,
                                     min_area_threshold=3000, max_area_threshold=10000000,
                                     max_letter_height=50, max_letter_width=50,
                                     border_margin=15, max_relative_size=0.9):
    """
    查找拓片轮廓，用于预测。使用与训练时相同的参数。 (增加调试打印)
    """
    # print(f"--- [分割函数] 尝试处理: {image_path} ---") # DEBUG
    try:
        if not os.path.exists(image_path):
             print(f"错误：[分割函数] 测试文件路径不存在 {image_path}")
             print(f"当前工作目录: {os.getcwd()}")
             return [], None
        # 使用 imdecode 处理可能包含中文的路径
        n = np.fromfile(image_path, np.uint8)
        img_gray = cv2.imdecode(n, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"错误：[分割函数] 无法解码灰度图像 {image_path}")
            return [], None
        # print(f"  [分割函数] 灰度图像加载成功, shape: {img_gray.shape}") # DEBUG
        img_color = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img_color is None:
            # print("  [分割函数] 无法解码彩色图像，将从灰度图转换。") # DEBUG
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_height, img_width = img_gray.shape[:2]
        # print(f"  [分割函数] 彩色图像加载/转换成功, shape: {img_color.shape}") # DEBUG

    except Exception as e:
        print(f"错误：[分割函数] 读取或解码图像时发生错误 {image_path}: {e}")
        return [], None

    # --- 1. 全局阈值处理 (Otsu) ---
    # print("  [分割函数] 应用阈值处理...") # DEBUG
    thresh_val, binary_mask = cv2.threshold(img_gray, 0, 255, threshold_type)
    # print(f"  [分割函数] 阈值处理完成, Otsu阈值: {thresh_val}") # DEBUG

    # --- 2. 形态学闭运算 ---
    if closing_kernel_size > 0 and closing_iterations > 0:
        # print(f"  [分割函数] 应用闭运算 (kernel={closing_kernel_size}, iter={closing_iterations})...") # DEBUG
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # --- 3. 形态学开运算 ---
    if opening_kernel_size > 0 and opening_iterations > 0:
        # print(f"  [分割函数] 应用开运算 (kernel={opening_kernel_size}, iter={opening_iterations})...") # DEBUG
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=opening_iterations)

    # --- 4. 查找轮廓 ---
    # print("  [分割函数] 查找轮廓...") # DEBUG
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"  [分割函数] 找到 {len(contours)} 个初始轮廓。") # DEBUG

    rubbing_info_list = []
    if contours:
        # print(f"  [分割函数] 开始过滤轮廓 (min_area={min_area_threshold}, ...)") # DEBUG
        filtered_out_count = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # --- Apply all filters ---
            if area < min_area_threshold: filtered_out_count += 1; continue
            if area > max_area_threshold: filtered_out_count += 1; continue
            if h < max_letter_height or w < max_letter_width: filtered_out_count += 1; continue
            if x < border_margin or y < border_margin or \
               (x + w) > (img_width - border_margin) or \
               (y + h) > (img_height - border_margin): filtered_out_count += 1; continue
            if w > img_width * max_relative_size or h > img_height * max_relative_size: filtered_out_count += 1; continue

            rubbing_info_list.append({'contour': contour, 'area': area, 'bbox': (x, y, w, h)})
        # print(f"  [分割函数] 轮廓过滤完成. 保留: {len(rubbing_info_list)}, 过滤掉: {filtered_out_count}") # DEBUG

    # print(f"--- [分割函数] 返回 {len(rubbing_info_list)} 个有效轮廓 ---") # DEBUG
    return rubbing_info_list, img_color
# --- 函数定义结束 ---

print("--- 函数定义完成 ---") # DEBUG

# --- 主预测流程 ---
if __name__ == "__main__":
    print("\n--- 进入 `if __name__ == \"__main__\":` ---") # DEBUG: Crucial check

    # --- 配置 ---
    print("--- 开始配置参数 ---") # DEBUG
    MODEL_PATH = "masked_patch_classifier_final.keras"
    IMAGE_TO_PREDICT = r"E:/oracle_talent/oracle_data/tokyo_data/27-41.jpg"
    OUTPUT_DIR = r"E:/oracle_agent/result"
    CLASSES = ['animal_bone', 'tortoise_shell']

    # --- 分割和过滤参数 ---
    THRESHOLD_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    CLOSING_KERNEL_SIZE = 5
    CLOSING_ITERATIONS = 2
    OPENING_KERNEL_SIZE = 3
    OPENING_ITERATIONS = 1
    MIN_AREA = 4000
    MAX_AREA_THRESHOLD = 10000000
    MAX_LETTER_HEIGHT = 50
    MAX_LETTER_WIDTH = 50
    BORDER_MARGIN = 15
    MAX_RELATIVE_SIZE = 0.9
    MIN_CONFIDENCE_THRESHOLD = 0.5
    PATCH_HEIGHT = 160
    PATCH_WIDTH = 160
    CONTOUR_THICKNESS = 3
    print(f"模型路径: {MODEL_PATH}") # DEBUG
    print(f"预测图像: {IMAGE_TO_PREDICT}") # DEBUG
    print(f"输出目录: {OUTPUT_DIR}") # DEBUG
    print(f"分割参数: MinArea={MIN_AREA}, ConfidenceThresh={MIN_CONFIDENCE_THRESHOLD}") # DEBUG
    print("--- 参数配置完成 ---") # DEBUG

    # --- 检查路径 ---
    print("--- 开始检查路径 ---") # DEBUG
    if not os.path.exists(MODEL_PATH):
        print(f"!!! 错误：找不到模型文件 {MODEL_PATH} !!!") # DEBUG
        exit(f"错误：找不到模型文件 {MODEL_PATH}")
    if not os.path.exists(IMAGE_TO_PREDICT):
        print(f"!!! 错误：找不到要预测的图片文件 '{IMAGE_TO_PREDICT}' !!!") # DEBUG
        exit(f"错误：找不到要预测的图片文件 '{IMAGE_TO_PREDICT}'")
    print("模型和图像文件路径检查通过。") # DEBUG

    # --- 创建输出目录 ---
    print("--- 开始创建输出目录 ---") # DEBUG
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"结果将保存到: {OUTPUT_DIR}")
    except Exception as e:
        print(f"!!! 创建输出目录时出错 '{OUTPUT_DIR}': {e} !!!") # DEBUG
        exit(f"创建输出目录时出错 '{OUTPUT_DIR}': {e}")

    # --- 1. 加载分类模型 ---
    print("\n--- 步骤 1: 准备加载分类模型 ---") # DEBUG: Print *before* the try block
    classification_model = None # Initialize to None
    try:
        print(f"尝试加载模型: {MODEL_PATH} ... (这步可能需要一些时间)") # DEBUG: Print right before load_model
        start_load_time = time.time()
        classification_model = keras.models.load_model(MODEL_PATH)
        end_load_time = time.time()
        print(f"--- 分类模型加载成功! 耗时: {end_load_time - start_load_time:.2f} 秒 ---") # DEBUG: Print on success
    except FileNotFoundError:
        print(f"!!! 加载模型失败: 文件未找到 {MODEL_PATH} !!!") # DEBUG
        exit(f"加载模型时出错: 文件未找到 {MODEL_PATH}")
    except OSError as e: # Catch potential OS errors during file reading
        print(f"!!! 加载模型失败: 发生 OSError (可能是文件损坏或权限问题): {e} !!!") # DEBUG
        exit(f"加载模型时出错 (OSError): {e}")
    except Exception as e:
        print(f"!!! 加载模型失败: 发生未知异常: {e} !!!") # DEBUG
        # import traceback
        # traceback.print_exc() # Uncomment for detailed stack trace if needed
        exit(f"加载分类模型时出错: {e}")

    if classification_model is None:
         print("!!! 严重错误: 模型对象仍然是 None，加载失败。 !!!")
         exit("模型加载失败，无法继续。")
    else:
         print("--- 模型对象已确认加载 (非 None) ---")

    # --- 2. 查找所有拓片轮廓 ---
    print(f"\n--- 步骤 2: 开始查找拓片轮廓: {IMAGE_TO_PREDICT} ---") # DEBUG
    try:
        rubbing_infos, result_image_from_find = find_rubbings_and_calculate_area_for_prediction(
            IMAGE_TO_PREDICT,
            threshold_type=THRESHOLD_TYPE,
            closing_kernel_size=CLOSING_KERNEL_SIZE,
            closing_iterations=CLOSING_ITERATIONS,
            opening_kernel_size=OPENING_KERNEL_SIZE,
            opening_iterations=OPENING_ITERATIONS,
            min_area_threshold=MIN_AREA,
            max_area_threshold=MAX_AREA_THRESHOLD,
            max_letter_height=MAX_LETTER_HEIGHT,
            max_letter_width=MAX_LETTER_WIDTH,
            border_margin=BORDER_MARGIN,
            max_relative_size=MAX_RELATIVE_SIZE
        )
    except Exception as e:
        print(f"!!! 调用 find_rubbings... 时发生意外错误: {e} !!!") # DEBUG
        exit()

    # --- 关键检查点 ---
    print(f"\n--- 分割函数调用完成 ---") # DEBUG
    if not rubbing_infos:
        print(f"!!! 关键问题: 未能检测到符合条件的拓片轮廓 (分割函数返回空列表)。") # DEBUG
        print(f"    请检查图像内容和分割/过滤参数:")
        print(f"    - 图像: {IMAGE_TO_PREDICT}")
        print(f"    - Min Area: {MIN_AREA}")
        # ... (print other parameters if needed) ...
        if result_image_from_find is not None:
             print("    分割函数返回了图像，但没有符合条件的轮廓。") # DEBUG
             # We set backend to Agg, so plt.show() will not work directly.
             # We can try saving the intermediate image instead.
             try:
                 debug_filename = os.path.join(OUTPUT_DIR, "debug_no_contours_found.png")
                 print(f"    尝试保存未找到轮廓的图像到: {debug_filename}")
                 cv2.imwrite(debug_filename, result_image_from_find)
                 # Or using imencode for paths with non-ASCII chars
                 # is_success, buf = cv2.imencode(".png", result_image_from_find)
                 # if is_success:
                 #     with open(debug_filename, 'wb') as f:
                 #         buf.tofile(f)
             except Exception as save_err:
                 print(f"    保存调试图像时出错: {save_err}")
        else:
            print("    分割函数未返回可显示的图像。")
        exit("--- 主流程因未找到轮廓而终止 ---")

    print(f"找到 {len(rubbing_infos)} 个拓片进行分类 (已应用面积过滤)。")

    # --- 加载原始图像 ---
    print("加载原始图像用于裁剪...") # DEBUG
    try:
        n_orig = np.fromfile(IMAGE_TO_PREDICT, np.uint8)
        original_image_color = cv2.imdecode(n_orig, cv2.IMREAD_COLOR)
        if original_image_color is None:
             original_image_gray = cv2.imdecode(n_orig, cv2.IMREAD_GRAYSCALE)
             if original_image_gray is not None:
                 original_image_color = cv2.cvtColor(original_image_gray, cv2.COLOR_GRAY2BGR)
             else: raise ValueError("无法加载原始图像 (彩色或灰度)")
        # print(f"  原始图像加载成功, shape: {original_image_color.shape}") # DEBUG
    except Exception as e: exit(f"加载原始图像用于裁剪时出错: {e}")

    # --- 3. 遍历拓片 ---
    all_predictions = []
    kept_predictions_count = 0
    result_image_final = result_image_from_find.copy() if result_image_from_find is not None else original_image_color.copy()
    print(f"\n--- 步骤 3: 开始遍历 {len(rubbing_infos)} 个找到的拓片 ---") # DEBUG

    for i, info in enumerate(rubbing_infos):
        print(f"\n--- 处理拓片 {i+1}/{len(rubbing_infos)} ---") # DEBUG
        contour = info['contour']
        area = info['area']
        x, y, w, h = info['bbox']
        # print(f"  原始索引: {i}, 面积: {area:.2f}, BBox: ({x},{y},{w},{h})") # DEBUG

        # --- 应用掩码和裁剪 ---
        try:
            mask = np.zeros(original_image_color.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            crop_x, crop_y, crop_w, crop_h = x, y, w, h
            if crop_w <= 0 or crop_h <= 0:
                 print(f"  警告: 无效的 BBox 尺寸 W={crop_w}, H={crop_h}. 跳过。")
                 continue
            crop_y_end = min(crop_y + crop_h, original_image_color.shape[0])
            crop_x_end = min(crop_x + crop_w, original_image_color.shape[1])
            crop_y = max(0, crop_y)
            crop_x = max(0, crop_x)
            masked_full_image = cv2.bitwise_and(original_image_color, original_image_color, mask=mask)
            rubbing_crop_masked = masked_full_image[crop_y:crop_y_end, crop_x:crop_x_end]
        except Exception as e:
            print(f"  错误: 应用掩码或裁剪时出错 for bbox ({x},{y},{w},{h}): {e}")
            continue

        if rubbing_crop_masked.size == 0:
            print("  警告: 裁剪后区域为空，跳过。")
            continue

        # --- 预处理 ---
        try:
            patch_resized = cv2.resize(rubbing_crop_masked, (PATCH_WIDTH, PATCH_HEIGHT), interpolation=cv2.INTER_AREA)
            patch_processed = patch_resized.astype('float32')
            patch_processed = model_preprocess(patch_processed)
            patch_batch = np.expand_dims(patch_processed, axis=0)
        except Exception as e:
            print(f"  错误: 预处理拓片时出错: {e}")
            continue

        # --- 进行分类预测 ---
        print("  进行分类预测...") # DEBUG
        try:
            start_time = time.time()
            if classification_model is None:
                print("  !!!错误：分类模型对象无效，无法预测!!!")
                continue
            predictions = classification_model.predict(patch_batch, verbose=0)
            end_time = time.time()
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASSES[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            print(f"  预测结果: {predicted_class_name} (置信度: {confidence:.4f}), 耗时: {end_time - start_time:.3f}s") # DEBUG

            # --- 置信度过滤和绘制 ---
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                print(f"  置信度 >= {MIN_CONFIDENCE_THRESHOLD}，保留此预测。") # DEBUG
                all_predictions.append({ 'id': i + 1, 'area': area, 'class': predicted_class_name, 'confidence': confidence, 'bbox': (x, y, w, h)})
                kept_predictions_count += 1
                try:
                    color = (255, 0, 0) if predicted_class_name == 'animal_bone' else (0, 0, 255) # Blue for animal, Red for tortoise? Check colors
                    cv2.drawContours(result_image_final, [contour], -1, color, CONTOUR_THICKNESS)
                    label_text = f"{i+1}:{predicted_class_name[0]}"
                    text_x, text_y = x, y - 10 if y > 20 else y + h + 15
                    cv2.putText(result_image_final, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                except Exception as draw_err:
                    print(f"  警告: 绘制轮廓/文本时出错: {draw_err}")
            else:
                print(f"  置信度 < {MIN_CONFIDENCE_THRESHOLD}，已过滤掉此预测。") # DEBUG

        except Exception as e:
            print(f"  错误: 进行分类预测时出错: {e}") # DEBUG
            print(f"  出错时的输入形状: {patch_batch.shape}")

    # --- 4. 显示并保存最终结果图像 ---
    print("\n--- 步骤 4: 所有拓片处理完成, 准备保存结果 ---") # DEBUG (Removed 'display' since Agg backend is used)
    if result_image_final is not None:
        final_title = f'检测并分类 (保留 {kept_predictions_count}/{len(rubbing_infos)}, Conf>={MIN_CONFIDENCE_THRESHOLD}, Area>={MIN_AREA})'
        print(f"最终结果: {final_title}") # DEBUG

        # --- 显示 (Commented out because Agg backend cannot show) ---
        # print("准备显示最终结果图像...") # DEBUG
        # try:
        #     plt.figure(figsize=(15, 15))
        #     plt.imshow(cv2.cvtColor(result_image_final, cv2.COLOR_BGR2RGB), interpolation='nearest')
        #     plt.title(final_title)
        #     plt.axis('off')
        #     plt.show() # This will likely raise an error with Agg backend
        #     print("图像显示窗口已弹出 (或将在脚本结束后显示)。") # DEBUG
        # except Exception as e:
        #     print(f"错误: 显示最终结果图像时出错 (可能是因为使用了 Agg 后端): {e}")

        # --- 保存 ---
        print("准备保存最终结果图像...") # DEBUG
        try:
            base_filename = os.path.basename(IMAGE_TO_PREDICT)
            safe_basename = re.sub(r'[\\/*?:"<>|]', '_', base_filename)
            output_filename = os.path.join(OUTPUT_DIR, "result_" + safe_basename)
            # Ensure extension (e.g., .jpg or .png)
            name, ext = os.path.splitext(output_filename)
            if not ext:
                ext = '.jpg' # Default to .jpg if no extension
                output_filename = name + ext

            print(f"尝试保存到: {output_filename}") # DEBUG
            # Using imencode to handle potential non-ASCII paths better
            is_success, im_buf_arr = cv2.imencode(ext, result_image_final) # Use the determined extension
            if is_success:
                with open(output_filename, 'wb') as f:
                    im_buf_arr.tofile(f)
                print(f"结果图像已保存到: {output_filename}") # DEBUG
            else:
                print(f"错误：无法使用 cv2.imencode 编码结果图像。") # DEBUG
        except Exception as e:
            print(f"错误: 保存结果图像时出错: {e}")
    else:
        print("错误: 最终结果图像对象为 None，无法保存。") # DEBUG

    # --- 总结 ---
    print("\n--- 最终保留的预测总结 ---") # DEBUG
    if all_predictions:
        for pred in all_predictions:
            print(f"  拓片 {pred['id']} (原始序号): 面积={pred['area']:.0f}, 类别={pred['class']}, 置信度={pred['confidence']:.2f}, BBox={pred['bbox']}")
    else:
        print("  没有预测结果满足设定的面积和置信度阈值。") # DEBUG

    print("\n--- 主流程正常结束 ---") # DEBUG

else:
    print("--- 脚本被导入，未执行主流程 ---") # DEBUG