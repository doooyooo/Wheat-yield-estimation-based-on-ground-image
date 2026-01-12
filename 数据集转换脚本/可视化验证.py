import cv2
import os
import numpy as np


def visualize_yolo_labels(image_dir, label_dir, output_dir=None, num_classes=5):
    """
    可视化YOLO格式的标注框

    Args:
        image_dir: 图像文件夹路径
        label_dir: YOLO标注文件(.txt)文件夹路径
        output_dir: 保存可视化结果的文件夹路径(可选)
        num_classes: 类别数量，用于确定颜色
    """
    # 定义不同类别的颜色
    color_list = [
        (200, 0, 0),  # 红色
        (0, 200, 0),  # 绿色
        (0, 0, 200),  # 蓝色
        (200, 200, 0),  # 黄色
        (0, 200, 200),  # 青色
    ]

    # 创建输出目录(如果指定)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # 构建图像和标注文件路径
        img_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            print(f"警告: 未找到对应的标注文件 {label_file}")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        height, width = img.shape[:2]

        # 读取标注文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 绘制所有标注框
        for line in lines:
            line = line.strip().split()
            if not line:
                continue

            # 解析标注信息
            class_id = int(line[0])
            x_center = float(line[1])
            y_center = float(line[2])
            box_w = float(line[3])
            box_h = float(line[4])

            # 将归一化坐标转换为像素坐标
            x1 = int((x_center - box_w / 2) * width)
            y1 = int((y_center - box_h / 2) * height)
            x2 = int((x_center + box_w / 2) * width)
            y2 = int((y_center + box_h / 2) * height)

            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # 绘制边界框
            color = color_list[class_id % len(color_list)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 添加类别标签
            cv2.putText(img, f"{class_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # # 显示图像
        # cv2.imshow(f"Image: {image_file}", img)
        # cv2.waitKey(0)

        # 保存可视化结果(如果指定了输出目录)
        if output_dir:
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, img)

    # cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    image_dir = r"E:\dataset\spike\images_1024"  # 调整后的图像文件夹
    label_dir = r"E:\dataset\spike\txt"  # YOLO标注文件夹
    output_dir = r"E:\dataset\spike\visualized"  # 可视化结果保存文件夹(可选)

    visualize_yolo_labels(image_dir, label_dir, output_dir)