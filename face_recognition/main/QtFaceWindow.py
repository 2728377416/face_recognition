import sys
import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QBrush, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QLineEdit, QTextEdit, QMessageBox)
from face_utils import FaceProcessor
from csv_utils import save_face_to_csv, check_face


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, float)  # 添加帧率参数
    detection_result = pyqtSignal(list)  # [(name, similarity, bbox, confidence)]

    def __init__(self):
        super().__init__()
        self.processor = FaceProcessor()
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.frame_queue = None
        self.skip_frames = 1  # 跳帧处理，每n帧处理一次

    def start_capture(self, source=0, skip_frames=1):
        self.running = True
        self.skip_frames = max(1, skip_frames)
        # 在Windows上优先使用DShow后端
        if sys.platform == 'win32':
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        
        # 优化视频捕获参数
        if source == 0:  # 只有摄像头才设置这些参数
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 设置缓冲区大小为1，减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame_count = 0
        self.last_time = time.time()
        self.start()

    def run(self):
        from collections import deque
        import threading
        
        # 使用双端队列作为缓冲区
        self.frame_queue = deque(maxlen=2)
        
        # 视频捕获线程
        def capture_thread():
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if len(self.frame_queue) == self.frame_queue.maxlen:
                        self.frame_queue.popleft()
                    self.frame_queue.append(frame)
                else:
                    self.running = False
        
        # 处理线程
        def process_thread():
            process_count = 0
            while self.running:
                if self.frame_queue:
                    frame = self.frame_queue[-1]  # 只处理最新帧
                    process_count += 1
                    
                    # 跳帧处理
                    if process_count % self.skip_frames != 0:
                        continue
                        
                    # 计算FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_time >= 1.0:  # 每秒更新一次FPS
                        self.fps = self.frame_count / (current_time - self.last_time)
                        self.frame_count = 0
                        self.last_time = current_time
                    
                    # 异步处理人脸检测
                    detections = self.processor.detect_faces(frame)
                    results = []
                    for det in detections:
                        face_img = frame[det['coordinates'][1]:det['coordinates'][3],
                                   det['coordinates'][0]:det['coordinates'][2]]
                        if face_img.size > 0:
                            embedding = self.processor.get_face_embedding(
                                self.processor.preprocess_face(face_img))
                            name, similarity = check_face(embedding.cpu())
                            results.append((name, similarity, det['coordinates']))
                    
                    # 发送结果和当前FPS
                    self.detection_result.emit(results)
                    self.frame_ready.emit(frame, self.fps)
        
        # 启动线程
        threading.Thread(target=capture_thread, daemon=True).start()
        threading.Thread(target=process_thread, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class FaceRecognitionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置窗口图标
        from PyQt5.QtGui import QIcon
        self.setWindowIcon(QIcon('qt_img/icon.jpg'))
        self.init_ui()
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.detection_result.connect(self.update_detections)
        self.recognition_results = []
        self.current_frame = None
        self.enrollment_mode = False
        self.is_video_recognition = False
        self.stop_video_recognition = False
        # 添加FPS显示标签
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.fps_label.setStyleSheet("""
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 16px;
                background: rgba(0, 0, 0, 0.5);
                padding: 2px 5px;
                border-radius: 3px;
            }
        """)
        # 在视频标签上叠加FPS显示
        self.video_label_layout = QVBoxLayout()
        self.video_label_layout.addWidget(self.fps_label)
        self.video_label_layout.addStretch()
        self.video_label.setLayout(self.video_label_layout)
        
        # 性能优化相关变量
        self.last_frame_time = 0
        self.smooth_fps = 0
        self.frame_times = []

    def init_ui(self):
        self.setWindowTitle("智能人脸识别系统")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        # 设置背景图
        palette = main_widget.palette()
        palette.setBrush(main_widget.backgroundRole(),
                         QBrush(QPixmap("qt_img/background.jpg").scaled(
                             self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)))
        main_widget.setPalette(palette)
        main_widget.setAutoFillBackground(True)

        # 设置控件样式
        main_widget.setStyleSheet("""
            QLabel, QPushButton, QTextEdit, QLineEdit {
                background: rgba(255, 255, 255, 0.7);
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background: rgba(200, 230, 255, 0.9);
                border: 1px solid #64B5F6;
            }
            QTextEdit {
                background: rgba(255, 255, 255, 0.85);
            }
        """)
        main_layout = QHBoxLayout()

        # 左侧视频区
        left_panel = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.video_label)

        # 右侧控制面板
        right_panel = QVBoxLayout()

        # 控制按钮
        control_group = QWidget()
        control_layout = QVBoxLayout()

        # 录入部分
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入姓名")
        self.name_input.setVisible(False)
        self.capture_btn = QPushButton("拍照录入")
        self.capture_btn.setCheckable(True)
        self.capture_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white;
            }
            QPushButton:hover { 
                background-color: #1976D2;
            }
            QPushButton:checked { 
                background-color: #FF5722;
            }
            QPushButton:checked:hover {
                background-color: #E64A19;
            }
        """)
        self.image_input_btn = QPushButton("图片录入")
        self.video_input_btn = QPushButton("视频录入")

        # 识别部分
        self.recognition_btn = QPushButton("实时识别")
        self.image_recog_btn = QPushButton("图片识别")
        self.video_recog_btn = QPushButton("视频识别")
        self.recognition_btn.setCheckable(True)
        self.recognition_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white;
            }
            QPushButton:hover { 
                background-color: #388E3C;
            }
            QPushButton:checked { 
                background-color: #F44336;
            }
            QPushButton:checked:hover {
                background-color: #D32F2F;
            }
        """)

        # 录入按钮组
        input_group = QHBoxLayout()
        input_group.addWidget(self.capture_btn)
        input_group.addWidget(self.image_input_btn)
        input_group.addWidget(self.video_input_btn)

        # 识别按钮组
        recog_group = QHBoxLayout()
        recog_group.addWidget(self.recognition_btn)
        recog_group.addWidget(self.image_recog_btn)
        recog_group.addWidget(self.video_recog_btn)

        control_layout.addWidget(self.name_input)
        control_layout.addLayout(input_group)
        control_layout.addLayout(recog_group)
        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)

        # 历史记录区
        history_group = QWidget()
        history_layout = QVBoxLayout()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(QLabel("识别结果:"))
        history_layout.addWidget(self.history_text)
        history_group.setLayout(history_layout)
        right_panel.addWidget(history_group)

        # 组合布局
        main_layout.addLayout(left_panel, 70)
        main_layout.addLayout(right_panel, 30)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 连接信号
        self.recognition_btn.toggled.connect(self.toggle_recognition)
        self.capture_btn.toggled.connect(self.toggle_enrollment)
        self.image_input_btn.clicked.connect(self.load_image_for_enrollment)
        self.video_input_btn.clicked.connect(self.load_video_for_enrollment)
        self.image_recog_btn.clicked.connect(self.load_image_for_recognition)
        self.video_recog_btn.clicked.connect(self.toggle_video_recognition)

    def toggle_recognition(self, checked):
        if checked:
            if self.enrollment_mode:
                self.reset_enrollment_state()
            self.recognition_btn.setText("停止识别")
            self.video_thread.start_capture()
        else:
            self.recognition_btn.setText("开始识别")
            self.video_thread.stop()

    def toggle_enrollment(self, checked):
        if checked:
            self.enrollment_mode = True
            self.capture_btn.setText("点击拍照")
            self.name_input.setVisible(True)
            self.name_input.setFocus()
            self.video_thread.start_capture()
        else:
            if self.enrollment_mode and self.current_frame is not None:
                name = self.name_input.text()
                if name and self.recognition_results:
                    for _, _, bbox in self.recognition_results:
                        x1, y1, x2, y2 = bbox
                        face_img = self.current_frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            embedding = self.video_thread.processor.get_face_embedding(
                                self.video_thread.processor.preprocess_face(face_img))
                            save_face_to_csv(name, embedding.cpu())
                            self.history_text.append(f"{name} 拍照录入成功")

                            # 显示录入成功的框
                            cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            self.update_frame(self.current_frame)
            self.reset_enrollment_state()

    def toggle_video_recognition(self):
        if self.is_video_recognition:
            # 停止识别
            self.stop_video_recognition = True
            self.video_recog_btn.setText("视频识别")
            self.video_recog_btn.setStyleSheet("")
            self.is_video_recognition = False
        else:
            # 开始识别
            if self.enrollment_mode:
                self.reset_enrollment_state()

            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
            if file_path:
                self.stop_video_recognition = False
                self.process_video_for_recognition(file_path)

    def reset_enrollment_state(self):
        self.enrollment_mode = False
        self.capture_btn.setText("拍照录入")
        self.capture_btn.setChecked(False)
        self.name_input.setVisible(False)
        self.video_thread.stop()

    def update_frame(self, frame, fps):
        """优化后的帧更新函数，添加FPS显示"""
        # 平滑FPS计算
        current_time = time.time()
        if self.last_frame_time > 0:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 10:
                self.frame_times.pop(0)
            avg_frame_time = np.mean(self.frame_times)
            self.smooth_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        self.last_frame_time = current_time
        
        # 更新FPS显示
        display_fps = self.smooth_fps if self.smooth_fps > 0 else fps
        self.fps_label.setText(f"FPS: {display_fps:.1f}")
        
        # 保留原有处理逻辑
        self.current_frame = frame
        if self.recognition_results:
            for name, similarity, bbox in self.recognition_results:
                x1, y1, x2, y2 = bbox
                color = (0, 0, 255) if self.enrollment_mode else (255, 0, 0)
                thickness = 3 if self.enrollment_mode else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                try:
                    from PIL import ImageFont, ImageDraw, Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype("simsun.ttc", 16)
                    text = "未知人脸" if name == "Unknown" else f"{name} 相似度: {similarity:.2f}"
                    if self.is_video_recognition:
                        draw.text((x2 - 150, y1 - 20), text, font=font, fill=(255, 0, 0))
                    else:
                        draw.text((x1, y1 - 20), text, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except:
                    text = "未知人脸" if name == "Unknown" else f"{name} {similarity:.2f}"
                    if self.is_video_recognition:
                        cv2.putText(frame, text, (x2 - 100, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 优化图像显示
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_detections(self, results):
        self.recognition_results = results
        if results:
            for name, similarity, _ in results:
                self.history_text.append(f"{name} - 相似度: {similarity:.2f}")

    def save_enrollment(self):
        name = self.name_input.text()
        if name and self.recognition_results and self.current_frame is not None:
            for _, _, bbox in self.recognition_results:
                x1, y1, x2, y2 = bbox
                face_img = self.current_frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    embedding = self.video_thread.processor.get_face_embedding(
                        self.video_thread.processor.preprocess_face(face_img))
                    save_face_to_csv(name, embedding.cpu())
                    self.history_text.append(f"{name} 录入成功")
            self.capture_btn.setChecked(False)

    def load_image_for_enrollment(self):
        from PyQt5.QtWidgets import QFileDialog, QInputDialog

        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            return

        # 缩放并检测人脸
        frame = self.scale_frame(frame)
        detections = self.video_thread.processor.detect_faces(frame)

        if not detections:
            QMessageBox.warning(self, "提示", "未检测到人脸")
            return

        # 标记所有人脸框
        for det in detections:
            x1, y1, x2, y2 = det['coordinates']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示图片
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # 简单输入姓名对话框
        name, ok = QInputDialog.getText(self, '录入人脸', '请输入姓名:')
        if ok and name:
            for det in detections:
                x1, y1, x2, y2 = det['coordinates']
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    embedding = self.video_thread.processor.get_face_embedding(
                        self.video_thread.processor.preprocess_face(face_img))
                    save_face_to_csv(name, embedding.cpu())
                    self.history_text.append(f"{name} 从图片录入成功")

    def load_video_for_enrollment(self):
        from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov)")
        if not file_path:
            return

        # 读取最后一帧
        cap = cv2.VideoCapture(file_path)
        last_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        cap.release()

        if last_frame is None:
            QMessageBox.warning(self, "错误", "无法读取视频文件")
            return

        # 缩放并检测人脸
        last_frame = self.scale_frame(last_frame)
        detections = self.video_thread.processor.detect_faces(last_frame)

        if not detections:
            QMessageBox.warning(self, "提示", "未检测到人脸")
            return

        # 标记所有人脸框
        for det in detections:
            x1, y1, x2, y2 = det['coordinates']
            cv2.rectangle(last_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示最后一帧
        height, width, channel = last_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(last_frame.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # 使用简单输入框获取姓名
        name, ok = QInputDialog.getText(self, '录入人脸', '请输入姓名:')
        if ok and name:
            for det in detections:
                x1, y1, x2, y2 = det['coordinates']
                face_img = last_frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    try:
                        embedding = self.video_thread.processor.get_face_embedding(
                            self.video_thread.processor.preprocess_face(face_img))
                        save_face_to_csv(name, embedding.cpu())

                        # 标记录入成功的人脸框为绿色
                        cv2.rectangle(last_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        q_img = QImage(last_frame.data, width, height, bytes_per_line,
                                       QImage.Format_RGB888).rgbSwapped()
                        self.video_label.setPixmap(QPixmap.fromImage(q_img))

                        self.history_text.append(f"{name} 从视频录入成功")
                    except Exception as e:
                        QMessageBox.warning(self, "错误", f"人脸录入失败: {str(e)}")

    def load_image_for_recognition(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.png *.jpeg)")
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                self.process_frame_for_recognition(frame)

    def process_frame_for_enrollment(self, frame):
        detections = self.video_thread.processor.detect_faces(frame)
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det['coordinates']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "录入中", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            face_img = frame[detections[0]['coordinates'][1]:detections[0]['coordinates'][3],
                       detections[0]['coordinates'][0]:detections[0]['coordinates'][2]]
            if face_img.size > 0:
                embedding = self.video_thread.processor.get_face_embedding(
                    self.video_thread.processor.preprocess_face(face_img))
                save_face_to_csv(self.name_input.text(), embedding.cpu())
                self.history_text.append(f"{self.name_input.text()} 从媒体文件录入成功")

                scaled_frame = self.scale_frame(frame)
                height, width, channel = scaled_frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(scaled_frame.data, width, height, bytes_per_line,
                               QImage.Format_RGB888).rgbSwapped()
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def process_frame_for_recognition(self, frame):
        detections = self.video_thread.processor.detect_faces(frame)
        results = []
        for det in detections:
            face_img = frame[det['coordinates'][1]:det['coordinates'][3],
                       det['coordinates'][0]:det['coordinates'][2]]
            if face_img.size > 0:
                embedding = self.video_thread.processor.get_face_embedding(
                    self.video_thread.processor.preprocess_face(face_img))
                name, similarity = check_face(embedding.cpu())
                results.append((name, similarity, det['coordinates']))

        self.recognition_results = results

        if results:
            # 获取原始图像尺寸
            original_height, original_width = frame.shape[:2]

            # 基准参数（针对1080p优化）
            BASE_RESOLUTION = 1080  # 基准分辨率高度
            BASE_FONT_SIZE = 24  # 基准字体大小
            BASE_BOX_THICKNESS = 3  # 基准框线粗细

            # 计算动态比例
            resolution_ratio = original_height / BASE_RESOLUTION

            # 动态调整参数（设置最小值）
            font_size = max(16, int(BASE_FONT_SIZE * resolution_ratio))
            box_thickness = max(2, int(BASE_BOX_THICKNESS * resolution_ratio))

            for name, similarity, bbox in results:
                x1, y1, x2, y2 = bbox

                # 绘制人脸框（使用动态粗细）
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), box_thickness)

                # 准备显示文本
                text = f"{name} {similarity:.2f}" if name != "Unknown" else "未知人脸"

                try:
                    # 使用PIL显示中文（动态字体大小）
                    from PIL import ImageFont, ImageDraw, Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype("simsun.ttc", font_size)

                    # 计算文本宽度
                    text_width = draw.textlength(text, font=font)

                    # 调整文本位置防止超出边界
                    text_x = min(max(10, x1), int(frame.shape[1] - text_width - 10))
                    text_y = max(font_size + 5, y1 - 5)

                    # 添加文本背景增强可读性
                    draw.rectangle(
                        [(text_x - 5, text_y - font_size - 5),
                         (text_x + text_width + 5, text_y + 5)],
                        fill=(0, 0, 0)  # 黑色背景
                    )

                    # 绘制文本
                    draw.text((text_x, text_y - font_size), text,
                              font=font, fill=(255, 255, 255))
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"PIL文本渲染失败: {e}, 使用OpenCV回退")
                    # OpenCV回退方案
                    font_scale = font_size / 40  # 将字体大小转换为OpenCV的字体比例
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_thickness)

                    # 调整文本位置防止超出边界
                    text_x = min(max(10, x1), int(frame.shape[1] - text_width - 10))
                    text_y = max(text_height + 5, y1 - 5)

                    # 添加文本背景
                    cv2.rectangle(frame,
                                  (text_x - 5, text_y - text_height - 5),
                                  (text_x + text_width + 5, text_y + 5),
                                  (0, 0, 0), -1)

                    # 绘制文本
                    cv2.putText(frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), box_thickness)

        # 显示处理后的图片
        scaled_frame = self.scale_frame(frame)
        height, width, channel = scaled_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(scaled_frame.data, width, height, bytes_per_line,
                       QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # 添加到历史记录
        for name, similarity, _ in results:
            self.history_text.append(f"图片识别: {name} - 相似度: {similarity:.2f}")
    
    def toggle_recognition(self, checked):
        if checked:
            if self.enrollment_mode:
                self.reset_enrollment_state()
            self.recognition_btn.setText("停止识别")
            # 增加跳帧数为4，提高流畅度
            self.video_thread.start_capture(skip_frames=4)
        else:
            self.recognition_btn.setText("开始识别")
            self.video_thread.stop()
            self.fps_label.setText("FPS: --")

    def toggle_enrollment(self, checked):
        if checked:
            self.enrollment_mode = True
            self.capture_btn.setText("点击拍照")
            self.name_input.setVisible(True)
            self.name_input.setFocus()
            # 录入模式不需要高帧率，设置跳帧数为1保证质量
            self.video_thread.start_capture(skip_frames=1)
        else:
            if self.enrollment_mode and self.current_frame is not None and self.recognition_results:
                name = self.name_input.text()
                if name:
                    for _, _, bbox in self.recognition_results:
                        x1, y1, x2, y2 = bbox
                        face_img = self.current_frame[y1:y2, x1:x2]
                        if face_img.size > 0:
                            try:
                                embedding = self.video_thread.processor.get_face_embedding(
                                    self.video_thread.processor.preprocess_face(face_img))
                                save_face_to_csv(name, embedding.cpu())
                                self.history_text.append(f"{name} 拍照录入成功")
                                
                                # 显示录入成功的绿色框
                                cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                height, width = self.current_frame.shape[:2]
                                bytes_per_line = 3 * width
                                q_img = QImage(self.current_frame.data, width, height, bytes_per_line,
                                               QImage.Format_RGB888).rgbSwapped()
                                self.video_label.setPixmap(QPixmap.fromImage(q_img))
                                
                                QMessageBox.information(self, "提示", f"{name} 录入成功")
                            except Exception as e:
                                QMessageBox.warning(self, "错误", f"录入失败: {str(e)}")
            self.reset_enrollment_state()

    def process_video_for_recognition(self, video_path):
        self.video_recog_btn.setText("停止识别")
        self.video_recog_btn.setStyleSheet("background-color: #F44336; color: white;")
        self.is_video_recognition = True
        self.stop_video_recognition = False

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        process_every_n_frames = 5  # 降低处理频率提高性能

        # 基准参数（针对1080p优化）
        BASE_RESOLUTION = 1080
        BASE_FONT_SIZE = 24
        BASE_BOX_THICKNESS = 3
        MIN_FONT_SIZE = 16
        MIN_BOX_THICKNESS = 1

        # 获取显示区域尺寸
        display_size = self.video_label.size()
        display_width = display_size.width()
        display_height = display_size.height()

        while cap.isOpened() and not self.stop_video_recognition:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                original_height, original_width = frame.shape[:2]

                # 计算动态缩放比例（保持宽高比）
                scale = min(display_width / original_width, display_height / original_height)

                # 动态调整参数（基于原始分辨率）
                resolution_ratio = original_height / BASE_RESOLUTION
                font_scale = max(MIN_FONT_SIZE, BASE_FONT_SIZE * resolution_ratio)
                thickness = max(MIN_BOX_THICKNESS, int(BASE_BOX_THICKNESS * resolution_ratio))

                # 执行缩放（设置10%缓冲区间）
                if scale < 0.9 or scale > 1.1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                    current_scale = scale
                else:
                    current_scale = 1.0

                # 人脸检测和识别
                detections = self.video_thread.processor.detect_faces(frame)
                results = []
                for det in detections:
                    x1, y1, x2, y2 = det['coordinates']
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        embedding = self.video_thread.processor.get_face_embedding(
                            self.video_thread.processor.preprocess_face(face_img))
                        name, similarity = check_face(embedding.cpu())
                        results.append((name, similarity, (x1, y1, x2, y2)))

                # 更新显示
                self.recognition_results = results
                if results:
                    for name, similarity, bbox in results:
                        x1, y1, x2, y2 = bbox

                        # 绘制人脸框（动态粗细）
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (255, 0, 0), thickness)

                        # 准备显示文本
                        text = f"{name} {similarity:.2f}" if name != "Unknown" else "未知人脸"

                        # 使用PIL实现高质量文本渲染
                        try:
                            from PIL import ImageFont, ImageDraw, Image
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_img)

                            # 动态字体大小
                            adjusted_font_size = int(font_scale * current_scale)
                            font = ImageFont.truetype("simsun.ttc", adjusted_font_size)

                            # 计算文本边界
                            text_width = draw.textlength(text, font=font)
                            text_height = adjusted_font_size

                            # 自动调整位置（防止超出画面）
                            text_x = min(max(5, x1), int(frame.shape[1] - text_width - 5))
                            text_y = max(text_height + 5, y1 - 5)

                            # 绘制文本背景（半透明）
                            bg_color = (0, 0, 0, 128)  # 半透明黑色
                            draw.rectangle(
                                [(text_x - 3, text_y - text_height - 3),
                                 (text_x + text_width + 3, text_y + 3)],
                                fill=bg_color
                            )

                            # 绘制文本
                            draw.text((text_x, text_y - text_height), text,
                                      font=font, fill=(255, 255, 255))
                            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        except Exception as e:
                            # OpenCV回退方案
                            font_scale_opencv = max(0.5, font_scale / 40)
                            (text_width, text_height), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale_opencv, thickness)

                            # 位置调整
                            text_x = min(max(5, x1), int(frame.shape[1] - text_width - 5))
                            text_y = max(text_height + 5, y1 - 5)

                            # 绘制背景
                            cv2.rectangle(
                                frame,
                                (text_x - 3, text_y - text_height - 3),
                                (text_x + text_width + 3, text_y + 3),
                                (0, 0, 0), -1
                            )

                            # 绘制文本
                            cv2.putText(
                                frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale_opencv,
                                (255, 255, 255), thickness
                            )

                # 显示处理后的帧
                height, width = frame.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line,
                               QImage.Format_RGB888).rgbSwapped()
                self.video_label.setPixmap(QPixmap.fromImage(q_img))

                # 记录结果
                for name, similarity, _ in results:
                    self.history_text.append(f"视频帧 {frame_count}: {name} ({similarity:.2f})")
            # 计算并显示处理FPS
            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            self.fps_label.setText(f"处理FPS: {current_fps:.1f}")
            # 保持UI响应
            QApplication.processEvents()
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        self.is_video_recognition = False
        self.video_recog_btn.setText("视频识别")
        self.video_recog_btn.setStyleSheet("")
        cv2.destroyAllWindows()

    def scale_frame(self, frame):
        """缩放帧以适应窗口大小"""
        max_width = self.video_label.width()
        max_height = self.video_label.height()

        if frame.shape[1] > max_width or frame.shape[0] > max_height:
            scale = min(max_width / frame.shape[1], max_height / frame.shape[0])
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        return frame

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionUI()
    window.show()
    sys.exit(app.exec_())
