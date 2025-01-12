import os
import xml.etree.ElementTree as ET

# 提供目录路径
image_dir = "HelmetDetection/annotations"
output_dir = "HelmetDetection/labels"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义class_id映射（这里假设 'helmet' 类的 ID 是 0）
class_mapping = {'helmet': 0}


def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取文件名和图像大小
    filename = root.find('filename').text
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)

    # 准备写入的文本数据
    output_data = []

    # 处理所有object标签
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = class_mapping.get(class_name, -1)  # 默认类 ID 为 -1
        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算中心点和宽高
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        obj_width = (xmax - xmin) / width
        obj_height = (ymax - ymin) / height

        # 格式化为所需格式
        output_data.append(f"{class_id} {x_center} {y_center} {obj_width} {obj_height}")

    # 将结果写入txt文件
    output_txt_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    with open(output_txt_file, 'w') as f:
        for line in output_data:
            f.write(line + '\n')


def convert_all_xmls(image_dir):
    # 遍历图像目录并转换所有XML文件
    for root_dir, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_file = os.path.join(root_dir, file)
                convert_annotation(xml_file)
                print(f"Converted {file}")


# 执行转换
convert_all_xmls(image_dir)

