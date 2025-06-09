# main.py（主程序）
import sys
import cv2
import time
import csv
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer,QRect
from PyQt5.QtGui import (QImage, QPixmap, QBrush, QFont, QColor, QPainter, 
                         QFontDatabase)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget,
                            QPushButton, QTextEdit, QMessageBox)
import numpy as np
from face_utils import FaceProcessor
from csv_utils import check_face

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, float)
    detection_result = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.processor = FaceProcessor()
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.is_monitoring = False
        self.skip_frames = 2

    def start_capture(self, source=0):
        self.running = True
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                
                if source == 0:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                if self.cap.isOpened():
                    print(f"摄像头成功打开，重试次数: {retry_count}")
                    self.start()
                    return
                else:
                    raise RuntimeError("无法打开摄像头")
                    
            except Exception as e:
                retry_count += 1
                print(f"摄像头打开失败，重试 {retry_count}/{max_retries}: {str(e)}")
                if self.cap:
                    self.cap.release()
                time.sleep(1)
        
        print("摄像头初始化失败，请检查设备连接")
        self.running = False

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            if self.frame_count % self.skip_frames != 0:
                continue

            frame = cv2.flip(frame, 1)
            
            detections = []
            if self.is_monitoring:
                faces = self.processor.detect_faces(frame)
                for face in faces:
                    x1, y1, x2, y2 = face['coordinates']
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size > 0:
                        try:
                            processed = self.processor.preprocess_face(face_img)
                            embedding = self.processor.get_face_embedding(processed)
                            name, similarity = check_face(embedding.cpu())
                            detections.append((name, similarity, (x1, y1, x2, y2)))
                        except Exception as e:
                            print(f"特征提取失败: {str(e)}")
                            continue

            self.detection_result.emit(detections)
            self.frame_ready.emit(frame, self.fps if self.fps else 30)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class AccessControlUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能门禁管理系统")
        self.setGeometry(100, 100, 1280, 800)
        self.init_ui()
        self.init_system()
        
        # 加载中文字体
        QFontDatabase.addApplicationFont("msyh.ttf")
        
    def init_system(self):
        self.log_file = "access_log.csv"
        self.door_open = False
        self.unknown_counter = 0
        self.last_face_time = 0
        self.warning_triggered = False
        self.show_warning_overlay = False
        self.recognition_results = []
        
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.detection_result.connect(self.handle_detection)
        
        self.door_timer = QTimer()
        self.door_timer.timeout.connect(self.auto_close_door)
        self.warning_timer = QTimer()
        self.warning_timer.timeout.connect(self.check_unknown_duration)
        
        self.video_thread.start_capture()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧监控区
        left_panel = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(1024, 768)
        
        self.status_light = QLabel()
        self.status_light.setFixedSize(50, 50)
        self.update_door_status(False)
        
        self.monitor_btn = QPushButton("启动识别")
        self.monitor_btn.setCheckable(True)
        self.monitor_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                font-size: 18px;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #F44336;
            }
        """)
        
        left_panel.addWidget(self.status_light, alignment=Qt.AlignCenter)
        left_panel.addWidget(self.video_label)
        left_panel.addWidget(self.monitor_btn)

        # 右侧日志区
        right_panel = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background: #1E1E1E;
                color: #FFFFFF;
                font-family: Microsoft YaHei;
                font-size: 12px;
            }
        """)
        
        right_panel.addWidget(QLabel("系统日志:"))
        right_panel.addWidget(self.log_area)

        main_layout.addLayout(left_panel, 70)
        main_layout.addLayout(right_panel, 30)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.monitor_btn.toggled.connect(self.toggle_monitoring)

    def update_door_status(self, is_open):
        color = QColor(0, 255, 0) if is_open else QColor(255, 0, 0)
        pixmap = QPixmap(50, 50)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 50, 50)
        painter.end()
        self.status_light.setPixmap(pixmap)

    def toggle_monitoring(self, checked):
        self.video_thread.is_monitoring = checked
        if checked:
            self.monitor_btn.setText("停止识别")
            self.warning_timer.start(1000)
            self.log_event("SYSTEM", "人脸识别已启动")
        else:
            self.monitor_btn.setText("启动识别")
            self.warning_timer.stop()
            self.log_event("SYSTEM", "人脸识别已停止")

    def handle_detection(self, results):
        current_time = time.time()
        has_known_face = False
        
        if results and self.video_thread.is_monitoring:
            self.recognition_results = results
            for name, similarity, _ in results:
                if name != "Unknown" and similarity >= 0.6:
                    has_known_face = True
                    if not self.door_open:
                        self.open_door(name)
                    self.last_face_time = current_time
                    break

            if not has_known_face:
                self.unknown_counter += 1
                self.log_event("WARNING", f"检测到未授权人员 ({self.unknown_counter}/10秒)")
            else:
                self.unknown_counter = 0
        else:
            self.unknown_counter = 0

    def check_unknown_duration(self):
        """检查未授权人员持续时间"""
        if self.unknown_counter >= 10 and not self.warning_triggered:
            self.trigger_unknown_warning()
            self.warning_triggered = True
        elif self.unknown_counter < 10:
            self.warning_triggered = False

    def trigger_unknown_warning(self):
        """触发安全警报"""
        self.log_event("SECURITY", "警告：未授权人员持续存在超过10秒！")
        self.show_warning_overlay = True
        QTimer.singleShot(5000, lambda: setattr(self, 'show_warning_overlay', False))
        
        try:
            import winsound
            winsound.Beep(2000, 1000)
        except:
            pass
        
        alert = QMessageBox()
        alert.setWindowTitle("安全警报")
        alert.setText("""
        <html>
        <h2 style='color:red;'>⚠️ 安全警报！</h2>
        <p>检测到未授权人员持续存在！</p>
        </html>
        """)
        alert.setIcon(QMessageBox.Critical)
        alert.exec_()
        self.unknown_counter = 0

    def open_door(self, name):
        self.door_open = True
        self.update_door_status(True)
        self.door_timer.start(4000)
        self.log_event("ACCESS", f"门已开启 - 识别用户: {name}")

    def auto_close_door(self):
        if time.time() - self.last_face_time >= 4:
            self.door_open = False
            self.update_door_status(False)
            self.door_timer.stop()
            self.log_event("SYSTEM", "门已自动关闭")

    def log_event(self, event_type, detail):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, event_type, detail])
            self.log_area.append(f"[{timestamp}] {event_type}: {detail}")
        except Exception as e:
            print(f"日志记录失败: {e}")

    def update_frame(self, frame, fps):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.door_open:
            painter.setPen(QColor(0, 255, 0))
            painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
            painter.drawText(20, 50, "门已开启")

        painter.setPen(QColor(255, 255, 0))
        painter.setFont(QFont("Microsoft YaHei", 14))
        painter.drawText(w-150, 30, f"FPS: {fps:.1f}")

        for name, similarity, (x1, y1, x2, y2) in self.recognition_results:
            color = (QColor(0, 255, 0) if (name != "Unknown" and self.video_thread.is_monitoring) 
                    else QColor(255, 0, 0) if self.video_thread.is_monitoring 
                    else QColor(0, 0, 255))
            
            painter.setPen(color)
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            
            if self.video_thread.is_monitoring:
                label = f"{name}({similarity:.2f})" if name != "Unknown" else "未知人员"
                text_rect = QRect(x1, y1-30, 200, 30)
                painter.fillRect(text_rect, color)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Microsoft YaHei", 12))
                painter.drawText(text_rect, Qt.AlignLeft|Qt.AlignVCenter, label)

        # 绘制警告覆盖层
        if self.show_warning_overlay:
            overlay = QPixmap(pixmap.size())
            overlay.fill(QColor(255, 0, 0, 100))
            
            if int(time.time() * 2) % 2 == 0:
                overlay_painter = QPainter(overlay)
                overlay_painter.setPen(QColor(255, 255, 255))
                overlay_painter.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
                text = "安全警报！未授权人员"
                text_rect = overlay_painter.boundingRect(overlay.rect(), Qt.AlignCenter, text)
                overlay_painter.drawText(text_rect, Qt.AlignCenter, text)
                overlay_painter.end()
            
            painter.drawPixmap(0, 0, overlay)

        painter.end()
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AccessControlUI()
    window.show()
    sys.exit(app.exec_())
