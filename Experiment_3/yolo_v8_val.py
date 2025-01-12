
from ultralytics import YOLO
path = "./HelmetDetection/val/images/hard_hat_workers1082.png"


model = YOLO("runs/detect/train4/weights/best.pt")  # 加载训练得到的最佳模型
results = model.predict(path)

print(type(results))
output_dir = "path_to_output_directory.png"  # 指定保存目录
results[0].save(output_dir)  # 保存图像

