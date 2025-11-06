import cv2
import numpy as np
import os
import csv
import warnings
import onnxruntime
from ultralytics import YOLO
import time
import json
from typing import List, Dict, Tuple, Optional  # 类型提示，提升代码可读性

# 屏蔽无关警告
warnings.filterwarnings("ignore",
                        message="Specified provider 'CUDAExecutionProvider' is not in available provider names.")

# --------------------------
# 全局配置（统一路径管理）
# --------------------------
BASE_DIR = os.path.dirname(__file__)
USER_DB_PATH = os.path.join(BASE_DIR, "user_database.json")  # 存储用户ID-姓名映射
CHECKIN_RECORD_PATH = os.path.join(BASE_DIR, "checkin_records.json")  # 持久化签到记录
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # 人脸图片保存目录（新增：确保目录存在）
os.makedirs(DATASET_DIR, exist_ok=True)  # 自动创建目录，避免保存图片时出错


# --------------------------
# 1. YOLO人脸检测器（优化边界框计算）
# --------------------------
class YoloDetector:
    def __init__(self, model_path="yolov8l-face.pt"):
        self.model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO模型文件不存在：{self.model_path}")
        self.model = YOLO(self.model_path)
        self.conf_threshold = 0.6

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """检测人脸并返回边界框（x1,y1,x2,y2,置信度）"""
        results = self.model(img, conf=self.conf_threshold)
        faces = []
        for result in results:
            for box in result.boxes:
                # 确保坐标为整数且在图片范围内（新增：边界检查）
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                conf = float(box.conf[0])
                faces.append((x1, y1, x2, y2, conf))
        return faces


# --------------------------
# 2. ArcFace特征提取器（增强输入检查）
# --------------------------
class ArcFaceExtractor:
    def __init__(self):
        model_dir = os.path.join(BASE_DIR, "models", "buffalo_l")
        self.arcface_path = os.path.join(model_dir, "w600k_r50.onnx")
        if not os.path.exists(self.arcface_path):
            raise FileNotFoundError(f"ArcFace模型文件不存在：{self.arcface_path}")

        self.session = onnxruntime.InferenceSession(
            self.arcface_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_feature(self, face_img: np.ndarray) -> np.ndarray:
        """提取人脸特征向量（新增：输入有效性检查）"""
        if face_img is None or face_img.size == 0:
            raise ValueError("输入人脸图片为空或无效")

        # 预处理（严格按照模型要求）
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (112, 112))
            face_transposed = face_resized.transpose(2, 0, 1)
            mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([128.0, 128.0, 128.0], dtype=np.float32).reshape(3, 1, 1)
            face_normalized = (face_transposed - mean) / std
            input_blob = np.expand_dims(face_normalized, axis=0).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"人脸预处理失败：{str(e)}")

        # 执行推理
        try:
            feature = self.session.run([self.output_name], {self.input_name: input_blob})[0][0]
        except Exception as e:
            raise RuntimeError(f"特征提取推理失败：{str(e)}")

        return feature / np.linalg.norm(feature)


# --------------------------
# 3. 特征库管理（增强容错性）
# --------------------------
class FeatureDB:
    def __init__(self, db_path="face_features.csv"):
        self.db_path = os.path.join(BASE_DIR, db_path)
        self.features: List[np.ndarray] = []  # 类型标注
        self.names: List[str] = []  # 类型标注
        self._load_db()

    def _load_db(self) -> None:
        """加载特征库（新增：跳过无效行，添加异常捕获）"""
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row_num, row in enumerate(reader, 1):  # 记录行号，方便定位错误
                    if len(row) != 2:
                        print(f"警告：特征库第{row_num}行格式错误（需2列），已跳过")
                        continue
                    name, feat_str = row
                    try:
                        feature = np.array(eval(feat_str), dtype=np.float32)
                        # 检查特征维度是否正确（ArcFace输出通常为512维）
                        if feature.shape != (512,):
                            print(f"警告：特征库第{row_num}行特征维度错误（应为512），已跳过")
                            continue
                        self.names.append(name)
                        self.features.append(feature)
                    except Exception as e:
                        print(f"警告：特征库第{row_num}行解析失败：{str(e)}，已跳过")
        except Exception as e:
            print(f"特征库加载失败：{str(e)}，将使用空特征库")
            self.features = []
            self.names = []

    def save_feature(self, name: str, feature: np.ndarray) -> None:
        """保存特征到CSV（新增：检查特征有效性）"""
        if feature.shape != (512,):
            raise ValueError(f"特征维度错误（应为512，实际为{feature.shape[0]}）")

        try:
            with open(self.db_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([name, feature.tolist()])
            self.names.append(name)
            self.features.append(feature)
        except Exception as e:
            raise RuntimeError(f"特征保存失败：{str(e)}")

    def match_feature(self, feature: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """匹配特征并返回最佳结果"""
        if not self.features:
            return "陌生人", 0.0
        if feature.shape != (512,):
            return "特征无效", 0.0

        similarities = [np.dot(feature, feat) for feat in self.features]
        max_sim = max(similarities)
        max_idx = similarities.index(max_sim)
        return (self.names[max_idx], float(max_sim)) if max_sim >= threshold else ("陌生人", float(max_sim))


# --------------------------
# 4. 核心系统（增强鲁棒性）
# --------------------------
class FaceRecognitionSystem:
    def __init__(self):
        self.detector = YoloDetector()
        self.extractor = ArcFaceExtractor()
        self.db = FeatureDB()

        # 加载用户数据库（ID-姓名映射，持久化）
        self.user_database: Dict[str, Dict[str, str]] = self._load_user_database()
        # 加载签到记录（持久化）
        self.checkin_records: Dict[str, Dict[str, str]] = self._load_checkin_records()

    # --------------------------
    # 数据持久化相关方法（增强容错）
    # --------------------------
    def _load_user_database(self) -> Dict[str, Dict[str, str]]:
        """加载用户ID-姓名映射（新增：异常处理）"""
        if not os.path.exists(USER_DB_PATH):
            return {}
        try:
            with open(USER_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 验证数据格式（确保是dict且每个值包含"name"）
                if not isinstance(data, dict):
                    raise ValueError("用户数据库格式错误（应为字典）")
                for uid, info in data.items():
                    if not isinstance(info, dict) or "name" not in info:
                        raise ValueError(f"用户ID {uid} 格式错误（缺少name字段）")
                return data
        except Exception as e:
            print(f"用户数据库加载失败：{str(e)}，将使用空数据库")
            return {}

    def _save_user_database(self) -> None:
        """保存用户数据库（新增：异常处理）"""
        try:
            with open(USER_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(self.user_database, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"用户数据库保存失败：{str(e)}")

    def _load_checkin_records(self) -> Dict[str, Dict[str, str]]:
        """加载签到记录（新增：异常处理）"""
        if not os.path.exists(CHECKIN_RECORD_PATH):
            return {}
        try:
            with open(CHECKIN_RECORD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("签到记录格式错误（应为字典）")
                return data
        except Exception as e:
            print(f"签到记录加载失败：{str(e)}，将使用空记录")
            return {}

    def _save_checkin_records(self) -> None:
        """保存签到记录（新增：异常处理）"""
        try:
            with open(CHECKIN_RECORD_PATH, "w", encoding="utf-8") as f:
                json.dump(self.checkin_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"签到记录保存失败：{str(e)}")

    # --------------------------
    # 用户管理相关方法（新增关键功能）
    # --------------------------
    def get_user_id_by_name(self, name: str) -> Optional[str]:
        """通过姓名查询用户ID（返回第一个匹配项，新增：支持模糊查询提示）"""
        matches = [uid for uid, info in self.user_database.items() if info["name"] == name]
        if len(matches) == 0:
            # 提示可能的相似姓名（解决因空格/大小写导致的匹配失败）
            similar_names = [info["name"] for info in self.user_database.values() if name in info["name"]]
            if similar_names:
                print(f"警告：未找到姓名「{name}」，但存在相似姓名：{similar_names}")
            return None
        elif len(matches) > 1:
            print(f"警告：姓名「{name}」对应多个用户ID：{matches}，将使用第一个")
        return matches[0]

    def is_user_exist(self, user_id: str) -> bool:
        """检查用户ID是否已存在（新增：防止重复录入）"""
        return str(user_id) in self.user_database

    def register_face(self, user_id, name, img_path):
            """同一ID和姓名可重复录入，每次录入新增特征到特征库"""
            if not os.path.exists(img_path):
                return False, f"图片不存在：{img_path}"
            img = cv2.imread(img_path)
            if img is None:
                return False, f"图片无法打开：{img_path}"

            # 检测人脸（每次录入要求1张人脸）
            face_boxes = self.detector.detect(img)
            if len(face_boxes) != 1:
                return False, f"需1张人脸，实际检测到{len(face_boxes)}张"

            # 扩展人脸区域
            x1, y1, x2, y2, conf = face_boxes[0]
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - int(0.1 * w))
            y1 = max(0, y1 - int(0.1 * h))
            x2 = min(img.shape[1], x2 + int(0.1 * w))
            y2 = min(img.shape[0], y2 + int(0.1 * h))
            face_img = img[y1:y2, x1:x2]

            try:
                # 提取特征并保存（关键：每次录入都新增特征，不覆盖）
                feature = self.extractor.extract_feature(face_img)
                self.db.save_feature(name, feature)  # 特征库会累加保存

                # 更新用户映射（即使ID已存在，也更新姓名映射，确保一致性）
                self.user_database[str(user_id)] = {"name": name}
                self._save_user_database()

                # 保存图片时添加序号（区分同一用户的不同样本）
                sample_count = len([f for f in self.db.names if f == name])  # 统计该用户的样本数
                save_path = os.path.join(DATASET_DIR, f"{name}_{user_id}_sample{sample_count}.jpg")
                cv2.imwrite(save_path, face_img)

                return True, f"录入成功（第{sample_count}个样本）"
            except Exception as e:
                return False, f"特征提取失败：{str(e)}"

    def get_all_users(self) -> List[Dict[str, str]]:
        """获取所有用户列表（适配前端展示）"""
        return [{"id": uid, "name": info["name"]} for uid, info in self.user_database.items()]

    # --------------------------
    # 识别+签到联动方法（优化逻辑）
    # --------------------------
    def recognize_face(self, img_path: str, threshold: float = 0.6) -> Tuple[str, List[Dict], str]:
        """识别后自动签到，返回包含签到状态的结果"""
        if not os.path.exists(img_path):
            return "error", [], f"图片不存在：{img_path}"
        img = cv2.imread(img_path)
        if img is None:
            return "error", [], f"图片无法打开：{img_path}"

        face_boxes = self.detector.detect(img)
        if len(face_boxes) == 0:
            return "no_face", [], "未检测到人脸"

        results = []
        for box in face_boxes:
            x1, y1, x2, y2, _ = box
            # 扩展人脸区域
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - int(0.1 * w))
            y1 = max(0, y1 - int(0.1 * h))
            x2 = min(img.shape[1], x2 + int(0.1 * w))
            y2 = min(img.shape[0], y2 + int(0.1 * h))
            face_img = img[y1:y2, x1:x2]

            try:
                feature = self.extractor.extract_feature(face_img)
                name, sim = self.db.match_feature(feature, threshold)
                sim_float = float(sim)
                checkin_msg = ""

                # 只有匹配成功（非陌生人且相似度达标）才执行签到
                if name != "陌生人" and sim_float >= threshold:
                    user_id = self.get_user_id_by_name(name)
                    if user_id:
                        # 执行签到
                        _, checkin_msg = self.checkin(user_id, name)
                    else:
                        checkin_msg = f"未找到「{name}」的用户ID，无法签到"
                else:
                    checkin_msg = "未匹配到有效用户，无法签到"

                # 结果中添加签到状态（供前端展示）
                results.append({
                    "name": name,
                    "similarity": sim_float,
                    "bbox": (x1, y1, x2, y2),
                    "message": f"识别成功（相似度：{sim_float:.4f}）",
                    "checkin_msg": checkin_msg
                })
            except Exception as e:
                results.append({
                    "name": "error",
                    "similarity": 0.0,
                    "bbox": (x1, y1, x2, y2),
                    "message": f"特征提取失败：{str(e)}",
                    "checkin_msg": "识别失败，无法签到"
                })

        return "success", results, f"共检测到{len(face_boxes)}张人脸"

    def checkin(self, user_id: str, name: str) -> Tuple[bool, str]:
        """签到方法：持久化记录，避免重复签到"""
        user_id_str = str(user_id)
        if user_id_str not in self.checkin_records:
            self.checkin_records[user_id_str] = {
                "name": name,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_checkin_records()
            return True, "签到成功"
        return False, "已签到"

    def get_checkin_stats(self):
        all_users = self.get_all_users()
        checked_user_ids = set(self.checkin_records.keys())
        checked_count = len(checked_user_ids)
        unchecked_count = len(all_users) - checked_count  # 确保用总人数减已签到人数
        unchecked_users = [
            user for user in all_users
            if user["id"] not in checked_user_ids
        ]
        return {
            "total": len(all_users),
            "checked_count": checked_count,
            "checked_list": list(self.checkin_records.values()),
            "unchecked_count": unchecked_count,
            "unchecked_list": unchecked_users
        }
    def reset_checkin(self) -> Tuple[bool, str]:
        """重置签到记录（用于重新开始）"""
        self.checkin_records = {}
        self._save_checkin_records()
        return True, "签到记录已重置"