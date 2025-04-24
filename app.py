import os
import numpy as np
import cv2
import json
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from datetime import datetime

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
# 选择一个作为默认配置来源，或者在下面动态选择
# 这里我们假设基础配置相似，但会在调用时传递特定参数
if logic_a:
    DEFAULT_MODEL_PATH = logic_a.DEFAULT_MODEL_PATH
    DEFAULT_CLASSES = logic_a.DEFAULT_CLASSES # 假设类别相同
elif logic_b: # 如果 logic_a 导入失败，尝试从 logic_b 获取
    DEFAULT_MODEL_PATH = logic_b.DEFAULT_MODEL_PATH
    DEFAULT_CLASSES = logic_b.DEFAULT_CLASSES
else: # 如果两者都失败 (虽然上面有检查，但作为保险)
     DEFAULT_MODEL_PATH = "default_model.keras" # 需要提供一个后备路径
     DEFAULT_CLASSES = ["class1", "class2"] # 需要提供后备类别

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = DEFAULT_MODEL_PATH # 使用选择的默认模型路径

app = Flask(__name__)
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 全局模型变量 ---
model = None
model_load_error = None

# --- 函数 ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    global model, model_load_error
    # 尝试使用 logic_a 的加载器，如果失败则尝试 logic_b
    loader_function = None
    if logic_a and hasattr(logic_a, 'load_keras_model'):
        loader_function = logic_a.load_keras_model
        print("使用 logic_a.load_keras_model 加载模型")
    elif logic_b and hasattr(logic_b, 'load_keras_model'):
        loader_function = logic_b.load_keras_model
        print("使用 logic_b.load_keras_model 加载模型")

    if loader_function:
        model, model_load_error = loader_function(MODEL_PATH)
        if model_load_error:
            print(f"警告: 模型加载失败: {model_load_error}")
        elif model:
            print("Flask 应用模型加载成功。")
    else:
        model_load_error = "错误：未找到可用的模型加载函数 (load_keras_model)"
        print(model_load_error)


initialize_model()

# --- Splash Page Route ---
@app.route('/')
def splash():
    now = datetime.utcnow()
    return render_template('splash.html', now=now)

# --- API 端点 ---
@app.route('/api/detect', methods=['POST'])
def detect():
    # 获取选择的预测方法 (可以从 form data 或 query param 获取)
    # 默认使用 'logic_a'
    method = request.form.get('prediction_method', 'logic_a')
    print(f"[API] 请求使用方法: {method}")

    # 检查图像文件
    if 'image' not in request.files: return jsonify({"error": "未找到图像文件"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename): return jsonify({"error": "不支持的文件类型"}), 400

    # 检查模型
    if model is None:
        err_msg = model_load_error or "模型未成功加载。"
        return jsonify({"error": f"模型不可用: {err_msg}"}), 500

    predictor = None
    constants_module = None

    # 根据选择的方法确定要使用的函数和常量来源
    if method == 'logic_b' and logic_b and hasattr(logic_b, 'predict_rubbings'):
        print("[API] 选择 logic_b (predict_batch_test.py) 进行预测")
        predictor = logic_b.predict_rubbings
        constants_module = logic_b
    elif logic_a and hasattr(logic_a, 'predict_rubbings'): # 默认或显式选择 logic_a
        print("[API] 选择 logic_a (prediction_logic.py) 进行预测")
        predictor = logic_a.predict_rubbings
        constants_module = logic_a
    else:
        return jsonify({"error": f"选择的预测方法 '{method}' 不可用或未正确导入。"}), 500

    # 确保常量模块已加载
    if constants_module is None:
         return jsonify({"error": f"无法加载方法 '{method}' 的常量。"}), 500

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- 调用选择的预测函数，并传入对应的常量 ---
        predictions_api, _, error_msg = predictor(
            filepath,
            model,
            classes=getattr(constants_module, 'DEFAULT_CLASSES', DEFAULT_CLASSES), # 优先用特定模块的
            patch_height=getattr(constants_module, 'DEFAULT_PATCH_HEIGHT'),
            patch_width=getattr(constants_module, 'DEFAULT_PATCH_WIDTH'),
            min_confidence_threshold=getattr(constants_module, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD'),
            threshold_type=getattr(constants_module, 'DEFAULT_THRESHOLD_TYPE'),
            closing_kernel_size=getattr(constants_module, 'DEFAULT_CLOSING_KERNEL_SIZE'),
            closing_iterations=getattr(constants_module, 'DEFAULT_CLOSING_ITERATIONS'),
            opening_kernel_size=getattr(constants_module, 'DEFAULT_OPENING_KERNEL_SIZE'),
            opening_iterations=getattr(constants_module, 'DEFAULT_OPENING_ITERATIONS'),
            min_area_threshold=getattr(constants_module, 'DEFAULT_MIN_AREA'),
            max_area_threshold=getattr(constants_module, 'DEFAULT_MAX_AREA_THRESHOLD'),
            max_letter_height=getattr(constants_module, 'DEFAULT_MAX_LETTER_HEIGHT'),
            max_letter_width=getattr(constants_module, 'DEFAULT_MAX_LETTER_WIDTH'),
            border_margin=getattr(constants_module, 'DEFAULT_BORDER_MARGIN'),
            max_relative_size=getattr(constants_module, 'DEFAULT_MAX_RELATIVE_SIZE')
        )
        # ----------------------------------------------------

        try: os.remove(filepath)
        except Exception as e: print(f"删除临时文件 {filepath} 时出错: {e}")

        if error_msg: return jsonify({"error": f"检测过程中发生错误: {error_msg}"}), 500
        return jsonify(predictions_api), 200

    except Exception as e:
        app.logger.error(f"API 检测失败 (方法: {method}): {str(e)}", exc_info=True)
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


# --- Detection Interface Route ---
@app.route('/detect', methods=['GET', 'POST'])
def detection_interface():
    result_image_b64 = None
    predictions_for_html = None
    error_message = None
    request_processed = False
    selected_method = 'logic_a' # 默认值

    if request.method == 'POST':
        request_processed = True
        selected_method = request.form.get('prediction_method', 'logic_a') # 获取选择的方法
        print(f"[Web UI] 请求使用方法: {selected_method}")

        # 文件和模型检查 (同 API)
        if 'image' not in request.files: error_message = "未找到图像文件"
        else:
            file = request.files['image']
            if file.filename == '': error_message = "未选择文件"
            elif not allowed_file(file.filename): error_message = "不支持的文件类型"
            elif model is None:
                err_msg = model_load_error or "模型未成功加载。"
                error_message = f"模型不可用: {err_msg}"
            else:
                # --- 选择预测器和常量模块 ---
                predictor = None
                constants_module = None
                if selected_method == 'logic_b' and logic_b and hasattr(logic_b, 'predict_rubbings'):
                    print("[Web UI] 选择 logic_b (predict_batch_test.py) 进行预测")
                    predictor = logic_b.predict_rubbings
                    constants_module = logic_b
                elif logic_a and hasattr(logic_a, 'predict_rubbings'): # 默认或显式选择 logic_a
                    print("[Web UI] 选择 logic_a (prediction_logic.py) 进行预测")
                    predictor = logic_a.predict_rubbings
                    constants_module = logic_a
                else:
                    error_message = f"选择的预测方法 '{selected_method}' 不可用或未正确导入。"

                # 如果找到预测器，则继续处理
                if predictor and constants_module:
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)

                        # --- 调用选择的预测函数 ---
                        predictions_api, result_image, error_msg = predictor(
                            filepath,
                            model,
                            classes=getattr(constants_module, 'DEFAULT_CLASSES', DEFAULT_CLASSES),
                            patch_height=getattr(constants_module, 'DEFAULT_PATCH_HEIGHT'),
                            patch_width=getattr(constants_module, 'DEFAULT_PATCH_WIDTH'),
                            min_confidence_threshold=getattr(constants_module, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD'),
                            threshold_type=getattr(constants_module, 'DEFAULT_THRESHOLD_TYPE'),
                            closing_kernel_size=getattr(constants_module, 'DEFAULT_CLOSING_KERNEL_SIZE'),
                            closing_iterations=getattr(constants_module, 'DEFAULT_CLOSING_ITERATIONS'),
                            opening_kernel_size=getattr(constants_module, 'DEFAULT_OPENING_KERNEL_SIZE'),
                            opening_iterations=getattr(constants_module, 'DEFAULT_OPENING_ITERATIONS'),
                            min_area_threshold=getattr(constants_module, 'DEFAULT_MIN_AREA'),
                            max_area_threshold=getattr(constants_module, 'DEFAULT_MAX_AREA_THRESHOLD'),
                            max_letter_height=getattr(constants_module, 'DEFAULT_MAX_LETTER_HEIGHT'),
                            max_letter_width=getattr(constants_module, 'DEFAULT_MAX_LETTER_WIDTH'),
                            border_margin=getattr(constants_module, 'DEFAULT_BORDER_MARGIN'),
                            max_relative_size=getattr(constants_module, 'DEFAULT_MAX_RELATIVE_SIZE')
                        )
                        # --------------------------

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

    # 准备通用模板变量
    model_filename = os.path.basename(MODEL_PATH) if MODEL_PATH else "未知模型"
    model_status = "已加载" if model else f"加载失败 ({model_load_error or '未知错误'})"
    # 注意: model_note 现在可能需要根据选择的方法动态生成，如果参数不同的话
    # 为简单起见，先显示一个通用的或基于默认逻辑的
    default_constants = logic_a or logic_b # 获取第一个成功导入的模块
    min_conf = getattr(default_constants, 'DEFAULT_MIN_CONFIDENCE_THRESHOLD', 'N/A')
    min_area = getattr(default_constants, 'DEFAULT_MIN_AREA', 'N/A')
    model_note = f"当前默认配置: 置信度阈值 >= {min_conf}, 最小面积 >= {min_area}"
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
        # 将当前选择的方法传递给模板，以便下拉菜单可以显示当前选择
        current_method=selected_method,
        # 传递标志指示哪些方法可用
        logic_a_available=bool(logic_a and hasattr(logic_a, 'predict_rubbings')),
        logic_b_available=bool(logic_b and hasattr(logic_b, 'predict_rubbings'))
    )

# --- 运行 Flask 应用 ---
if __name__ == '__main__':
    # 部署时务必使用 debug=False
    # 使用 Waitress: waitress-serve --host=0.0.0.0 --port=5000 app:app
    app.run(debug=False, host='0.0.0.0', port=5000)