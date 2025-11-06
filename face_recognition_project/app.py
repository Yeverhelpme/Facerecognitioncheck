from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading  # 确保导入threading
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from face_system import FaceRecognitionSystem  # 导入人脸识别核心类
import time
from typing import Tuple, Optional  # 类型提示

# --------------------------
# 1. 初始化配置
# --------------------------
LURU_DIR = os.path.join("tupian", "luru")
SHIBIE_DIR = os.path.join("tupian", "shibie")

# 确保目录存在（自动创建）
os.makedirs(LURU_DIR, exist_ok=True)
os.makedirs(SHIBIE_DIR, exist_ok=True)

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大上传限制：50MB

# 全局初始化人脸识别系统（仅加载一次模型）
fr_system = FaceRecognitionSystem()

# --------------------------
# 2. 工具函数（图片处理相关）
# --------------------------
def get_base64_image(cv_img: np.ndarray) -> str:
    """将OpenCV格式图像（BGR）转为Base64字符串，用于前端展示"""
    try:
        # 转换颜色通道：BGR → RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"图片转Base64失败: {e}")
        # 生成错误提示图
        err_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(
            err_img,
            "图片处理失败",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # BGR红色
            2
        )
        return get_base64_image(err_img)


def draw_chinese_text(cv_img: np.ndarray, text: str, pos: Tuple[int, int], 
                     font_size: int = 20, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """在OpenCV图像上绘制中文（解决中文乱码问题）"""
    try:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        # 加载中文字体（若不存在则使用默认字体）
        font = ImageFont.truetype("simhei.ttf", font_size) if os.path.exists("simhei.ttf") else ImageFont.load_default()
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))  # BGR→RGB
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"绘制中文失败: {e}")
        return cv_img  # 失败时返回原图


# --------------------------
# 3. 定时清理图片逻辑（修复路径和引用）
# --------------------------
def clean_shibie_folder():
    """删除SHIBIE_DIR文件夹下的所有图片文件（10分钟循环一次）"""
    folder_path = SHIBIE_DIR  # 统一使用定义的路径
    if not os.path.exists(folder_path):
        print(f"清理提示：{folder_path}文件夹不存在，无需清理")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    deleted_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 跳过文件夹，只处理文件
        if not os.path.isfile(file_path):
            continue
        # 只删除图片文件
        if filename.lower().endswith(image_extensions):
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"已删除图片：{file_path}")
            except Exception as e:
                print(f"删除图片失败 {file_path}：{str(e)}")

    print(f"清理完成：共删除{deleted_count}张图片")
    # 10分钟后再次执行（修复Timer引用）
    threading.Timer(600, clean_shibie_folder).start()


# --------------------------
# 4. 接口实现
# --------------------------
@app.route("/register", methods=["POST"])
def register():
    """人脸录入接口：仅返回是否成功和姓名"""
    try:
        # 获取参数
        user_id = request.form.get("id")
        name = request.form.get("name")
        image_file = request.files.get("image")

        # 参数校验（失败时返回姓名和错误信息）
        if not user_id or not name or not image_file:
            return jsonify({
                               "success": False,
                               "name": name or "",
            "msg": "缺少参数：ID、姓名或图片"
            }), 400
        if not user_id.isdigit():
            return jsonify({
                "success": False,
                "name": name,
                "msg": "用户ID必须为数字"
            }), 400
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({
                "success": False,
                "name": name,
                "msg": "仅支持JPG、PNG格式图片"
            }), 400

        # 保存图片（仅后端临时使用，不返回路径）
        prefix = f"{name}_{user_id}_"
        existing_files = [f for f in os.listdir(LURU_DIR) if
                          f.startswith(prefix) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(existing_files) + 1
        image_filename = f"{prefix}{count}.jpg"
        save_path = os.path.join(LURU_DIR, image_filename)
        try:
            image_file.save(save_path)
        except Exception as e:
            return jsonify({
                "success": False,
                "name": name,
                "msg": f"图片保存失败：{str(e)}"
            }), 500

        # 调用录入逻辑
        success, msg = fr_system.register_face(user_id, name, save_path)

        # 成功时只返回success和name；失败时附加错误信息
        if success:
            return jsonify({
                "success": True,
                "name": name
            })
        else:
            return jsonify({
                "success": False,
                "name": name,
                "msg": msg
            })

    except Exception as e:
        return jsonify({
                           "success": False,
                           "name": name or "",
        "msg": f"服务器错误：{str(e)}"
        }), 500


@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"success": False, "msg": "缺少待识别图片"}), 400

        # 保存临时图片（仅后端临时使用，不返回路径）
        timestamp = int(time.time())
        random_suffix = os.urandom(4).hex()
        image_filename = f"temp_recognize_{timestamp}_{random_suffix}.jpg"
        save_path = os.path.join(SHIBIE_DIR, image_filename)
        image_file.save(save_path)

        # 调用识别逻辑
        status, results, msg = fr_system.recognize_face(save_path)

        # 精简results，只保留核心字段
        simplified_results = []
        for res in results:
            simplified_results.append({
                "name": res["name"],
                "similarity": res["similarity"],  # 保留供前端参考
                "bbox": res["bbox"],
                "checkin_msg": res["checkin_msg"]
            })

        # 生成原图Base64（前端绘制用）
        original_base64 = None
        try:
            img = cv2.imread(save_path)
            original_base64 = get_base64_image(img) if img is not None else None
        except Exception as e:
            print(f"原图转Base64失败：{str(e)}")

        # 只返回核心信息
        return jsonify({
            "success": True,
            "original_image_base64": original_base64,
            "results": simplified_results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "msg": f"服务器错误：{str(e)}"
        }), 500


@app.route("/checkin/stats", methods=["GET"])
def get_checkin_stats():
    try:
        stats = fr_system.get_checkin_stats()
        return jsonify({
            "success": True,
            "total": stats["total"],
            "checked_count": stats["checked_count"],
            "checked_list": stats["checked_list"],
            "unchecked_list": stats["unchecked_list"]
        })
    except Exception as e:
        return jsonify({"success": False, "msg": f"获取统计失败：{str(e)}"}), 500


@app.route("/checkin/reset", methods=["POST"])
def reset_checkin():
    try:
        success, msg = fr_system.reset_checkin()
        return jsonify({"success": success, "msg": msg})
    except Exception as e:
        return jsonify({"success": False, "msg": f"重置失败：{str(e)}"}), 500


# --------------------------
# 5. 启动服务
# --------------------------
if __name__ == "__main__":
    # 启动定时清理任务（首次执行）
    clean_shibie_folder()
    # 生产环境建议关闭debug，且不使用use_reloader
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)