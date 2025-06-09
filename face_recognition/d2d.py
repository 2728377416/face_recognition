import cv2
import numpy as np
import onnxruntime as ort
import time

# 配置参数
MODEL_PATH = "models/best.onnx"  # 模型路径
IMAGE_PATH = "tests_datas_jpg_video/test.jpg"       # 测试图像路径
WARMUP = 10                   # 预热次数
REPEAT = 100                  # 正式测试次数
CONF_THRESH = 0.25            # 置信度阈值
IOU_THRESH = 0.45             # NMS的IOU阈值

# ---------------------------------- 初始化模型 ----------------------------------
def init_model():
    # 创建ONNX Runtime会话（GPU加速）
    providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    return session, input_name, output_names

# ---------------------------------- 预处理函数 ----------------------------------
def preprocess(image):
    # Resize + BGR2RGB +归一化 (YOLOv8官方预处理方式)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

# ---------------------------------- 后处理函数（含NMS） ----------------------------------
def postprocess(outputs, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    处理YOLOv8输出:
    outputs: [1, 84, 8400] 格式 (xywh + 80类COCO概率)
    返回: [N, 6] 格式的检测结果 (x1, y1, x2, y2, conf, class_id)
    """
    # 解析输出
    predictions = np.squeeze(outputs[0]).T  # [8400, 84]
    
    # 过滤低置信度检测框
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > conf_thresh
    predictions = predictions[mask]
    scores = scores[mask]
    
    # 获取类别ID
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # 提取框坐标 (xywh -> xyxy)
    boxes = predictions[:, :4]
    boxes[:, 0] -= boxes[:, 2] / 2  # x_center -> x1
    boxes[:, 1] -= boxes[:, 3] / 2  # y_center -> y1
    boxes[:, 2] += boxes[:, 0]      # width -> x2
    boxes[:, 3] += boxes[:, 1]      # height -> y2
    
    # 执行NMS
    keep_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, iou_thresh)
    
    if len(keep_indices) == 0:
        return np.array([])
    
    # 合并结果
    detections = np.column_stack([
        boxes[keep_indices],
        scores[keep_indices],
        class_ids[keep_indices]
    ])
    return detections

# ---------------------------------- 主测试流程 ----------------------------------
def main():
    # 初始化模型
    session, input_name, output_names = init_model()
    
    # 读取测试图像
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image {IMAGE_PATH} not found")
    
    # 预热（避免首次运行耗时异常）
    for _ in range(WARMUP):
        input_tensor = preprocess(image)
        outputs = session.run(output_names, {input_name: input_tensor})
        _ = postprocess(outputs)
    
    # 正式测试
    total_time = 0.0
    for _ in range(REPEAT):
        start_time = time.perf_counter()
        
        # --- 完整流程 ---
        # 1. 预处理
        input_tensor = preprocess(image)
        # 2. 推理
        outputs = session.run(output_names, {input_name: input_tensor})
        # 3. 后处理
        detections = postprocess(outputs)
        # -----------------
        
        total_time += time.perf_counter() - start_time
    
    # 统计结果
    avg_time = total_time / REPEAT
    fps = 1 / avg_time
    
    print(f"\n端到端性能报告:")
    print(f"- 平均耗时: {avg_time * 1000:.2f} ms")
    print(f"- FPS     : {fps:.2f}")
    
    # 输出检测结果示例（可选）
    if len(detections) > 0:
        print("\n示例检测结果 (x1, y1, x2, y2, conf, class_id):")
        print(detections[0])  # 打印第一个检测框

if __name__ == "__main__":
    main()