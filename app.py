import os
import sys # PyInstaller: 导入 sys
import threading # pywebview: 导入 threading
# import webview # 暂时注释掉 webview，先确保 Render 部署成功
import numpy as np
import cv2
import json
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import time # 用于等待服务器启动

# --- PyInstaller: 获取资源基础路径的函数 ---
def get_base_path():
    """ 获取资源文件的基础路径，兼容开发环境和 PyInstaller 打包环境 """
    # 注意：在 Render 环境中，没有 sys.frozen，所以会返回脚本目录
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 如果是 PyInstaller 打包后的环境 (_MEIPASS 是临时文件夹路径)
        return sys._MEIPASS
    else:
        # 如果是普通开发环境或 Render 环境 (获取当前脚本所在目录)
        return os.path.abspath(os.path.dirname(__file__))

BASE_DIR = get_base_path()
print(f"资源基础路径 (BASE_DIR): {BASE_DIR}") # 调试信息

# --- 修改: 使用别名导入两个逻辑 ---
try:
    # 导入原始逻辑 (别名 logic_a)
    import prediction_logic as logic_a
    print("成功导入 prediction_logic as logic_a")
except ImportError as e:
    print(f"警告: 无法导入 prediction_logic: {e}")
    logic_a = None

try:
    # 导入测试逻辑 (别名 logic_b)
    import predict_batch_test as logic_b
    print("成功导入 predict_batch_test as logic_b")
except ImportError as e:
    print(f"警告: 无法导入 predict_batch_test: {e}")
    logic_b = None

# 检查至少有一个逻辑被成功导入
if logic_a is None and logic_b is None:
    raise ImportError("错误：两个预测逻辑脚本都未能成功导入！请检查文件是否存在且无语法错误。")

# --- 配置 ---
# **修正:** 直接定义 TFLite 文件名，不再从 logic 模块获取路径常量
DEFAULT_TFLITE_MODEL_FILENAME = "oracle_detector_model.tflite"
DEFAULT_CLASSES = ["animal_bone", "tortoise_shell"] # 默认类别，确保与模型一致

# 尝试从导入的模块获取类别，如果失败则使用默认值
if logic_a and hasattr(logic_a, 'DEFAULT_CLASSES'):
    DEFAULT_CLASSES = logic_a.DEFAULT_CLASSES
elif logic_b and hasattr(logic_b, 'DEFAULT_CLASSES'):
    DEFAULT_CLASSES = logic_b.DEFAULT_CLASSES
else:
    print("警告：无法从 logic_a 或 logic_b 获取 DEFAULT_CLASSES，使用内置默认值。")


UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# **修正:** 使用 get_base_path() 和固定文件名构建模型路径
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

# --- 全局模型/解释器变量 ---
# **修正:** 变量名改为 interpreter 更清晰
interpreter = None
interpreter_load_error = None

# --- 函数 (allowed_file, initialize_model) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# **修正:** 函数名改为 initialize_interpreter，加载 TFLite
def initialize_interpreter():
    global interpreter, interpreter_load_error
    loader_function = None
    # 优先使用 logic_a 的加载器
    if logic_a and hasattr(logic_a, 'load_tflite_model'):
        loader_function = logic_a.load_tflite_model
        print("使用 logic_a.load_tflite_model 加载模型")
    # 如果 logic_a 不可用或没有加载函数，尝试 logic_b
    elif logic_b and hasattr(logic_b, 'load_tflite_model'):
        loader_function = logic_b.load_tflite_model
        print("使用 logic_b.load_tflite_model 加载模型")

    if loader_function:
        # 使用计算好的 MODEL_PATH
        interpreter, interpreter_load_error = loader_function(MODEL_PATH)
        if interpreter_load_error:
            print(f"警告: TFLite Interpreter 加载失败: {interpreter_load_error}")
        elif interpreter:
            print("Flask 应用 TFLite Interpreter 加载成功。")
    else:
        interpreter_load_error = "错误：未找到可用的 TFLite 模型加载函数 (load_tflite_model)"
        print(interpreter_load_error)

# --- Flask 路由 (@app.route...) ---
@app.route('/')
def splash():
    now = datetime.utcnow()
    return render_template('splash.html', now=now)

@app.route('/api/detect', methods=['POST'])
def detect():
    method = request.form.get('prediction_method', 'logic_a')
    print(f"[API] 请求使用方法: {method}")

    if 'image' not in request.files: return jsonify({"error": "未找到图像文件"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename): return jsonify({"error": "不支持的文件类型"}), 400

    # **修正:** 检查 interpreter 是否加载
    if interpreter is None:
        err_msg = interpreter_load_error or "Interpreter 未成功加载。"
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

        # **修正:** 传递 interpreter 给预测函数
        predictions_api, _, error_msg = predictor(
            filepath, interpreter, # 传递 interpreter
            classes=getattr(constants_module, 'DEFAULT_CLASSES', DEFAULT_CLASSES),
            patch_height=getattr(constants_module, 'DEFAULT_PATCH_HEIGHT', 160), # 使用模型尺寸
            patch_width=getattr(constants_module, 'DEFAULT_PATCH_WIDTH', 160),   # 使用模型尺寸
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
        app.logger.error(f"API 检测失败 (方法: {method}): {str(e)}", exc_info=True)
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@app.route('/detect', methods=['GET', 'POST'])
def detection_interface():
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
            # **修正:** 检查 interpreter
            elif interpreter is None:
                err_msg = interpreter_load_error or "Interpreter 未成功加载。"
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

                        # **修正:** 传递 interpreter
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

    # **修正:** 使用固定文件名，检查 interpreter 状态
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

# --- 运行 Flask 应用 (Render 会使用 gunicorn) ---
# **修正:** 移除 pywebview 相关代码，专注于 Render 部署
def run_server():
    # 在服务器启动前加载模型/解释器
    print("--- 初始化 TFLite Interpreter ---")
    initialize_interpreter()
    if interpreter is None:
        print("错误：Interpreter 未能加载，Flask 服务器可能无法正常工作。")

    # Render 会使用 gunicorn 命令启动，这里的 app.run 仅用于本地测试
    # 如果直接运行 python app.py，则会执行这里
    print("--- 启动 Flask 开发服务器 (用于本地测试) ---")
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    run_server()

