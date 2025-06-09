import cv2
import torch
import numpy as np
from yolo_face import FaceDetector


class FaceProcessor:
    def __init__(self):
        """
        初始化人脸处理器
        - 自动检测可用设备(GPU/CPU)
        - 加载人脸检测器(YOLO模型)
        - 加载人脸识别模型(InceptionResnetV1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备

        # 初始化人脸检测器(YOLO模型)
        self.detector = FaceDetector(model_path='D:/biyesheji1/biyesheji/models/best.onnx')
        
        # 单独加载ONNX模型用于推理
        self.onnx_session = cv2.dnn.readNetFromONNX('D:/biyesheji1/biyesheji/models/vggface2.onnx')

    def preprocess_face(self, face_img):
        """
        人脸预处理流程
        参数:
            face_img: 输入的人脸图像(BGR格式)
        返回:
            预处理后的tensor(归一化后的RGB图像)
        处理步骤:
            1. 人脸对齐(基于关键点)
            2. 调整大小和颜色空间转换
            3. 转换为tensor并归一化
        """
        # 1. 人脸对齐(使用关键点检测)
        aligned_face = self.align_face(face_img)

        # 2. 调整大小和颜色空间转换
        face_img = cv2.resize(aligned_face, (160, 160))  # 调整为模型输入尺寸
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # 3. 转换为Tensor并归一化
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()  # HWC转CHW
        return (face_tensor - 127.5) / 128.0  # 归一化到[-1,1]范围

    def align_face(self, face_img):
        """
        人脸对齐方法
        参数:
            face_img: 输入的人脸图像
        返回:
            对齐后的人脸图像
        处理步骤:
            1. 检测人脸关键点(使用YOLO或默认位置)
            2. 计算仿射变换矩阵
            3. 应用变换进行对齐
        """
        # 1. 获取关键点(优先使用检测器的关键点)
        detections = self.detector.detect(face_img)
        if detections and 'landmarks' in detections[0]:
            landmarks = detections[0]['landmarks']
            src_points = np.array([
                landmarks[0],  # 左眼
                landmarks[1],  # 右眼
                landmarks[2],  # 鼻尖
                landmarks[3],  # 左嘴角
                landmarks[4]  # 右嘴角
            ], dtype=np.float32)
        else:
            # 回退到默认关键点位置
            h, w = face_img.shape[:2]
            src_points = np.array([
                [w * 0.3, h * 0.4],  # 左眼
                [w * 0.7, h * 0.4],  # 右眼
                [w * 0.5, h * 0.6],  # 鼻尖
                [w * 0.3, h * 0.8],  # 左嘴角
                [w * 0.7, h * 0.8]  # 右嘴角
            ], dtype=np.float32)

        # 2. 定义目标关键点位置(标准脸)
        h, w = face_img.shape[:2]
        dst_points = np.array([
            [0.35, 0.35],  # 左眼
            [0.65, 0.35],  # 右眼
            [0.5, 0.5],  # 鼻尖
            [0.35, 0.65],  # 左嘴角
            [0.65, 0.65]  # 右嘴角
        ], dtype=np.float32)
        dst_points[:, 0] *= w
        dst_points[:, 1] *= h

        # 3. 计算仿射变换矩阵
        M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]

        # 4. 应用变换(使用Lanczos插值)
        return cv2.warpAffine(face_img, M, (w, h), flags=cv2.INTER_LANCZOS4)

    def normalize_lighting(self, face_img):
        """
        改进的光照归一化方法
        参数:
            face_img: 输入的人脸图像
        返回:
            光照归一化后的图像
        处理步骤:
            1. 转换到LAB颜色空间
            2. 对L通道进行CLAHE增强
            3. 白平衡处理
            4. 合并通道并转换回BGR
        """
        # 0. 输入验证
        if face_img is None or face_img.size == 0:
            raise ValueError("输入图像为空")

        # 1. 转换到LAB空间并应用CLAHE
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建CLAHE对象
        l = clahe.apply(l)  # 增强亮度通道

        # 2. 确保通道一致性
        a = cv2.resize(a, (l.shape[1], l.shape[0])).astype(l.dtype)
        b = cv2.resize(b, (l.shape[1], l.shape[0])).astype(l.dtype)

        # 3. 白平衡处理
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        a = (a - (avg_a - 128) * 0.5).clip(0, 255).astype(l.dtype)
        b = (b - (avg_b - 128) * 0.5).clip(0, 255).astype(l.dtype)

        # 4. 合并通道并转换回BGR
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)  # 归一化到0-255

    def get_face_embedding(self, face_tensor):
        """
        获取人脸特征向量(512维)
        参数:
            face_tensor: 预处理后的人脸tensor
        返回:
            512维的人脸特征向量(PyTorch tensor)
        """
        # 使用ONNX进行推理
        face_np = face_tensor.numpy().transpose(1, 2, 0)
        blob = cv2.dnn.blobFromImage(face_np)
        self.onnx_session.setInput(blob)
        embedding = self.onnx_session.forward().flatten()
        return torch.from_numpy(embedding).to(self.device)

    def detect_faces(self, frame):
        """
        人脸检测接口
        参数:
            frame: 输入图像帧
        返回:
            检测到的人脸信息列表
        """
        return self.detector.detect(frame)  # 调用YOLO检测器
