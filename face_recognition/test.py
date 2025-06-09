import time
import cv2
import numpy as np
import onnxruntime as ort

# 配置参数
MODEL_PATH = "models/best.onnx"
IMAGE_PATH = "tests_datas_jpg_video/test.jpg"
WARMUP = 10   # 预热次数（避免初始化的干扰）
REPEAT = 100  # 正式测试次数

# 加载模型
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])  # CPU
# session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])  # GPU

# 获取输入输出名称
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# 准备输入数据（YOLOv8的默认输入尺寸为640x640）
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = cv2.resize(image, (640, 640))          # 调整尺寸
input_tensor = input_tensor.transpose(2, 0, 1)         # HWC -> CHW
input_tensor = np.expand_dims(input_tensor, axis=0)    # 添加batch维度
input_tensor = input_tensor.astype(np.float32) / 255.0 # 归一化

# 预热（Warmup）
for _ in range(WARMUP):
    session.run(output_names, {input_name: input_tensor})

# 正式测速
start_time = time.perf_counter()
for _ in range(REPEAT):
    outputs = session.run(output_names, {input_name: input_tensor})
end_time = time.perf_counter()

# 计算平均耗时（仅推理，不含预处理/后处理）
avg_inference_time = (end_time - start_time) / REPEAT
print(f"Average inference time: {avg_inference_time * 1000:.2f} ms")

# 计算FPS
fps = 1 / avg_inference_time
print(f"FPS: {fps:.2f}")