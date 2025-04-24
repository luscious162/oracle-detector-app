import os
import sys
import numpy as np
import cv2
import json
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import time # 用于调试或特定逻辑

# --- 获取资源基础路径的函数 ---
def get_base_path():
    """ 获取资源文件的基础路径，兼容开发环境和 PyInstaller 打包环境 """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    else:
        return os.path.abspath(os.path.dirname(__file__))

BASE_DIR = get_base_path()
print(f"资源基础路径 (BASE_DIR): {BASE_DIR}")

# --- 导入预测逻辑模块 ---
try:
    import prediction_logic as logic_a
    print("成功导入 prediction_logic as logic_a")
except ImportError as e:
    print(f"警告: 无法导入 prediction_logic: {e}")
    logic_a = None

try:
    import predict_batch_test as logic_b
    print("成功导入 predict_batch_test as logic_b")
except ImportError as e:
    print(f"警告: 无法导入 predict_batch_test: {e}")
    logic_b = None

if logic_a is None and logic_b is None:
    raise ImportError("错误：两个预测逻辑脚本都未能成功导入！")

# --- 配置 ---
DEFAULT_TFLITE_MODEL_FILENAME = "oracle_detector_model.tflite"
DEFAULT_CLASSES = ["animal_bone", "tortoise_shell"]

# 尝试从模块获取类别
if logic_a and hasattr(logic_a, 'DEFAULT_CLASSES'):
    DEFAULT_CLASSES = logic_a.DEFAULT_CLASSES
elif logic_b and hasattr(logic_b, 'DEFAULT_CLASSES'):
    DEFAULT_CLASSES = logic_b.DEFAULT_CLASSES
else:
    print("警告：无法从 logic_a 或 logic_b 获取 DEFAULT_CLASSES，使用内置默认值。")

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(BASE_DIR, DEFAULT_TFLITE_MODEL_FILENAME)
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')

print(f"模型文件预期路径 (MODEL_PATH): {MODEL_PATH}")
print(f"模板文件夹预期路径: {template_dir}")
print(f"静态文件夹预期路径: {static_dir}")
print(f"使用的类别: {DEFAULT_CLASSES}")

# --- Flask 应用实例 ---
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 全局解释器变量 ---
interpreter = None
interpreter_load_error = None

# --- 加载 TFLite 解释器的函数 ---
def initialize_interpreter():
    global interpreter, interpreter_load_error # 声明修改全局变量
    loader_function = None
    if logic_a and hasattr(logic_a, 'load_tflite_model'):
        loader_function = logic_a.load_tflite_model
        print("使用 logic_a.load_tflite_model 加载模型")
    elif logic_b and hasattr(logic_b, 'load_tflite_model'):
        loader_function = logic_b.load_tflite_model
        print("使用 logic_b.load_tflite_model 加载模型")

    if loader_function:
        interpreter_instance, error_msg = loader_function(MODEL_PATH)
        if error_msg:
            interpreter_load_error = error_msg # 记录错误
            print(f"警告: TFLite Interpreter 加载失败: {interpreter_load_error}")
        elif interpreter_instance:
            interpreter = interpreter_instance # 赋值给全局变量
            print("Flask 应用 TFLite Interpreter 加载成功。")
    else:
        interpreter_load_error = "错误：未找到可用的 TFLite 模型加载函数 (load_tflite_model)"
        print(interpreter_load_error)

# --- 在应用启动时（模块加载时）初始化解释器 ---
print("--- 初始化 TFLite Interpreter (全局作用域) ---")
initialize_interpreter()
if interpreter is None:
    print("错误：全局 Interpreter 未能加载，应用可能无法处理预测请求。")
# ------------------------------------------------

# --- 函数 (allowed_file) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask 路由 ---
@app.route('/')
def splash():
    now = datetime.utcnow()
    return render_template('splash.html', now=now)

@app.route('/api/detect', methods=['POST'])
def detect():
    # ... (路由内部逻辑保持不变，检查全局 interpreter 变量) ...
    method = request.form.get('prediction_method', 'logic_a')
    print(f"[API] 请求使用方法: {method}")

    if 'image' not in request.files: return jsonify({"error": "未找到图像文件"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename): return jsonify({"error": "不支持的文件类型"}), 400

    # 检查全局 interpreter 是否加载
    if interpreter is None:
        err_msg = interpreter_load_error or "Interpreter 未成功加载。"
        print(f"错误: /api/detect - {err_msg}") # 添加日志
        return jsonify({"error": f"模型解释器不可用: {err_msg}"}), 500

    predictor = None
    constants_module = None
    if method == 'logic_b' and logic_b and hasattr(logic_b, 'predict_rubbings'):
        print("[API] 选择 logic_b (predict_batch_test.py) 进行预测")
        predictor = logic_b.predict_rubbings
        constants_module = logic_b
    elif logic_a and hasattr(logic_a, 'predict_rubbings'):
        print("[API] 选择 logic_a (prediction_logic.py) 进行预测")
        predictor = logic_a.predict_rubbings
        constants_module = logic_a
    else:
        return jsonify({"error": f"选择的预测方法 '{method}' 不可用或未正确导入。"}), 500

    if constants_module is None:
         return jsonify({"error": f"无法加载方法 '{method}' 的常量。"}), 500

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 传递全局 interpreter 给预测函数
        predictions_api, _, error_msg = predictor(
            filepath, interpreter, # 传递 interpreter
            classes=getattr(constants_module, 'DEFAULT_CLASSES', DEFAULT_CLASSES),
            patch_height=getattr(constants_module, 'DEFAULT_PATCH_HEIGHT', 160),
            patch_width=getattr(constants_module, 'DEFAULT_PATCH_WIDTH', 160),
            min_confidence_threshold=getattr(constants_module, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD', 0.5),
            threshold_type=getattr(constants_module, 'DEFAULT_THRESHOLD_TYPE', 'adaptive'),
            closing_kernel_size=getattr(constants_module, 'DEFAULT_CLOSING_KERNEL_SIZE', 3),
            closing_iterations=getattr(constants_module, 'DEFAULT_CLOSING_ITERATIONS', 1),
            opening_kernel_size=getattr(constants_module, 'DEFAULT_OPENING_KERNEL_SIZE', 3),
            opening_iterations=getattr(constants_module, 'DEFAULT_OPENING_ITERATIONS', 1),
            min_area_threshold=getattr(constants_module, 'DEFAULT_MIN_AREA', 100),
            max_area_threshold=getattr(constants_module, 'DEFAULT_MAX_AREA_THRESHOLD', 10000),
            max_letter_height=getattr(constants_module, 'DEFAULT_MAX_LETTER_HEIGHT', 100),
            max_letter_width=getattr(constants_module, 'DEFAULT_MAX_LETTER_WIDTH', 100),
            border_margin=getattr(constants_module, 'DEFAULT_BORDER_MARGIN', 10),
            max_relative_size=getattr(constants_module, 'DEFAULT_MAX_RELATIVE_SIZE', 0.8)
        )

        try: os.remove(filepath)
        except Exception as e: print(f"删除临时文件 {filepath} 时出错: {e}")

        if error_msg: return jsonify({"error": f"检测过程中发生错误: {error_msg}"}), 500
        return jsonify(predictions_api), 200

    except Exception as e:
        # 使用 app.logger 记录更详细的错误信息
        app.logger.error(f"API 检测失败 (方法: {method}): {str(e)}", exc_info=True)
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@app.route('/detect', methods=['GET', 'POST'])
def detection_interface():
    # ... (路由内部逻辑保持不变，检查全局 interpreter 变量) ...
    result_image_b64 = None
    predictions_for_html = None
    error_message = None
    request_processed = False
    selected_method = 'logic_a'

    if request.method == 'POST':
        request_processed = True
        selected_method = request.form.get('prediction_method', 'logic_a')
        print(f"[Web UI] 请求使用方法: {selected_method}")

        if 'image' not in request.files: error_message = "未找到图像文件"
        else:
            file = request.files['image']
            if file.filename == '': error_message = "未选择文件"
            elif not allowed_file(file.filename): error_message = "不支持的文件类型"
            # 检查全局 interpreter
            elif interpreter is None:
                err_msg = interpreter_load_error or "Interpreter 未成功加载。"
                print(f"错误: /detect - {err_msg}") # 添加日志
                error_message = f"模型解释器不可用: {err_msg}"
            else:
                predictor = None
                constants_module = None
                if selected_method == 'logic_b' and logic_b and hasattr(logic_b, 'predict_rubbings'):
                    print("[Web UI] 选择 logic_b (predict_batch_test.py) 进行预测")
                    predictor = logic_b.predict_rubbings
                    constants_module = logic_b
                elif logic_a and hasattr(logic_a, 'predict_rubbings'):
                    print("[Web UI] 选择 logic_a (prediction_logic.py) 进行预测")
                    predictor = logic_a.predict_rubbings
                    constants_module = logic_a
                else:
                    error_message = f"选择的预测方法 '{selected_method}' 不可用或未正确导入。"

                if predictor and constants_module:
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)

                        # 传递全局 interpreter
                        predictions_api, result_image, error_msg = predictor(
                            filepath, interpreter, # 传递 interpreter
                            classes=getattr(constants_module, 'DEFAULT_CLASSES', DEFAULT_CLASSES),
                            patch_height=getattr(constants_module, 'DEFAULT_PATCH_HEIGHT', 160),
                            patch_width=getattr(constants_module, 'DEFAULT_PATCH_WIDTH', 160),
                            min_confidence_threshold=getattr(constants_module, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD', 0.5),
                            threshold_type=getattr(constants_module, 'DEFAULT_THRESHOLD_TYPE', 'adaptive'),
                            closing_kernel_size=getattr(constants_module, 'DEFAULT_CLOSING_KERNEL_SIZE', 3),
                            closing_iterations=getattr(constants_module, 'DEFAULT_CLOSING_ITERATIONS', 1),
                            opening_kernel_size=getattr(constants_module, 'DEFAULT_OPENING_KERNEL_SIZE', 3),
                            opening_iterations=getattr(constants_module, 'DEFAULT_OPENING_ITERATIONS', 1),
                            min_area_threshold=getattr(constants_module, 'DEFAULT_MIN_AREA', 100),
                            max_area_threshold=getattr(constants_module, 'DEFAULT_MAX_AREA_THRESHOLD', 10000),
                            max_letter_height=getattr(constants_module, 'DEFAULT_MAX_LETTER_HEIGHT', 100),
                            max_letter_width=getattr(constants_module, 'DEFAULT_MAX_LETTER_WIDTH', 100),
                            border_margin=getattr(constants_module, 'DEFAULT_BORDER_MARGIN', 10),
                            max_relative_size=getattr(constants_module, 'DEFAULT_MAX_RELATIVE_SIZE', 0.8)
                        )

                        predictions_for_html = predictions_api
                        try: os.remove(filepath)
                        except Exception as e: print(f"删除临时文件 {filepath} 时出错: {e}")

                        if error_msg: error_message = f"处理图像时出错: {error_msg}"
                        elif result_image is not None:
                            is_success, buffer = cv2.imencode('.jpg', result_image)
                            if is_success:
                                result_image_b64 = base64.b64encode(buffer).decode('utf-8')
                            else:
                                print("警告: 无法将结果图像编码为 JPG")
                                error_message = "无法编码结果图像"

                    except Exception as e:
                        app.logger.error(f"Web UI 处理图像时出错 (方法: {selected_method}): {str(e)}", exc_info=True)
                        error_message = f"处理图像时发生意外错误: {str(e)}"

    model_filename = DEFAULT_TFLITE_MODEL_FILENAME
    model_status = "已加载" if interpreter else f"加载失败 ({interpreter_load_error or '未知错误'})"
    default_constants = logic_a or logic_b
    if default_constants:
        min_conf = getattr(default_constants, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD', 'N/A')
        min_area = getattr(default_constants, 'DEFAULT_MIN_AREA', 'N/A')
        model_note = f"当前默认配置: 置信度阈值 >= {min_conf}, 最小面积 >= {min_area}"
    else:
        model_note = "无法加载默认配置信息。"
    now = datetime.utcnow()

    return render_template(
        'index.html',
        model_filename=model_filename,
        model_status=model_status,
        model_note=model_note,
        error_message=error_message,
        request_processed=request_processed,
        result_image_b64=result_image_b64,
        predictions=predictions_for_html,
        now=now,
        current_method=selected_method,
        logic_a_available=bool(logic_a and hasattr(logic_a, 'predict_rubbings')),
        logic_b_available=bool(logic_b and hasattr(logic_b, 'predict_rubbings'))
    )

# --- 移除 if __name__ == '__main__' 块 ---
# Render 使用 Gunicorn 启动，不需要这个块来运行服务器。
# 本地测试可以通过直接运行 `flask run` (需要设置 FLASK_APP=app.py)
# 或者使用 `gunicorn app:app` 来模拟 Render 环境。
# def run_server(): ...
# if __name__ == '__main__': run_server()


