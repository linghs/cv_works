
from ultralytics import YOLO

# 初始化模型
model = YOLO("yolov8n.pt")  # 选择一个预训练模型（如 yolov8n.pt，yolov8s.pt 等）

# 开始训练
model.train(data='data.yaml', epochs=50, imgsz=640)

