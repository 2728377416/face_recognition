from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path='E:/biyesheji/biyesheji/models/best.onnx'):
        # 自动检测模型格式
        if model_path.endswith('.onnx'):
            self.model = YOLO(model_path, task='detect')
        else:
            self.model = YOLO(model_path)
        print(f"Loaded model: {model_path}")
        self.iou_threshold = 0.45  # 默认iou阈值
        self.conf_threshold = 0.75  # 默认置信度阈值

    def detect(self, image):
        results = self.model(image, verbose=False, iou=self.iou_threshold, conf=self.conf_threshold)
        faces = []
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == 'face':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    faces.append({
                        'coordinates': (x1, y1, x2, y2),
                        'color': (255, 0, 0),
                        'confidence': float(box.conf)  # 添加置信度信息
                    })
        return faces
