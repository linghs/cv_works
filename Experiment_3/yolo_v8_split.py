import os
import random
import shutil

# 设置路径
dataset_dir = './HelmetDetection/images'  # 图像文件夹
labels_dir = './HelmetDetection/labels'  # 标签文件夹

train_dir = './HelmetDetection/train'
val_dir = './HelmetDetection/val'

# 创建训练集和验证集文件夹
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

# 获取图像和标签文件列表
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 随机打乱图像列表
random.seed(42)  # 固定随机种子，确保每次运行结果相同
random.shuffle(image_files)

# 划分比例（例如 80% 训练集，20% 验证集）
train_size = int(0.8 * len(image_files))
train_images = image_files[:train_size]
val_images = image_files[train_size:]

# 复制文件到训练集和验证集文件夹
def copy_files(image_list, src_dir, dest_img_dir, dest_label_dir):
    for image_name in image_list:
        label_name = os.path.splitext(image_name)[0] + '.txt'
        
        # 复制图像和标签文件
        shutil.copy(os.path.join(src_dir, image_name), os.path.join(dest_img_dir, image_name))
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(dest_label_dir, label_name))

# 复制训练集和验证集文件
copy_files(train_images, dataset_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
copy_files(val_images, dataset_dir, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

