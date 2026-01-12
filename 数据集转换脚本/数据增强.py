import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import json
import shutil
from tqdm import tqdm


class YOLODataAugmentor:
    """YOLO数据集增强器，支持多种图像增强操作并同步更新标注"""

    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir,
                 class_names=None, augment_times=1, seed=None):
        """
        初始化YOLO数据集增强器

        Args:
            image_dir: 原始图像文件夹路径
            label_dir: 原始标签文件夹路径
            output_image_dir: 增强后图像保存路径
            output_label_dir: 增强后标签保存路径
            class_names: 类别名称列表，默认为None(使用类别ID)
            augment_times: 每张图像增强的次数
            seed: 随机数种子，用于可重复的增强
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        self.class_names = class_names
        self.augment_times = augment_times

        # 创建输出目录
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 定义增强方法列表
        self.augment_methods = [
            self.flip_horizontal,
            self.flip_vertical,
            self.brightness_adjust,
            self.contrast_adjust,
            self.saturation_adjust,
            self.hue_adjust,
            self.add_gaussian_noise,
            self.add_salt_pepper_noise,
            self.elastic_transform
        ]

    def process_dataset(self):
        """处理整个数据集"""
        # 获取所有图像文件
        image_files = [f for f in os.listdir(self.image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"找到 {len(image_files)} 张图像")

        # 处理每张图像
        for image_file in tqdm(image_files, desc="处理图像"):
            image_path = os.path.join(self.image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)

            # 检查标签文件是否存在
            if not os.path.exists(label_path):
                print(f"警告: 未找到对应的标签文件 {label_file}，跳过")
                continue

            # 读取图像和标签
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图像 {image_path}，跳过")
                continue

            height, width = image.shape[:2]
            labels = self.read_yolo_labels(label_path, width, height)

            # 保存原始图像和标签的副本
            base_name = os.path.splitext(image_file)[0]
            orig_image_output_path = os.path.join(self.output_image_dir, f"{base_name}.png")
            orig_label_output_path = os.path.join(self.output_label_dir, f"{base_name}.txt")

            cv2.imwrite(orig_image_output_path, image)
            self.write_yolo_labels(orig_label_output_path, labels, width, height)

            # 应用增强
            for i in range(self.augment_times):
                # 随机选择增强方法
                augment_method = random.choice(self.augment_methods)

                # 应用增强
                augmented_image, augmented_labels = augment_method(image.copy(), labels.copy())

                # 保存增强后的图像和标签
                augmented_image_output_path = os.path.join(
                    self.output_image_dir, f"{base_name}_aug{i + 1}.png")
                augmented_label_output_path = os.path.join(
                    self.output_label_dir, f"{base_name}_aug{i + 1}.txt")

                cv2.imwrite(augmented_image_output_path, augmented_image)
                self.write_yolo_labels(augmented_label_output_path, augmented_labels, width, height)

    def read_yolo_labels(self, label_path, image_width, image_height):
        """
        读取YOLO格式的标签文件

        Args:
            label_path: 标签文件路径
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            标签列表，每个标签格式为 [class_id, x1, y1, x2, y2]
        """
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                if not line:
                    continue

                class_id = int(line[0])
                x_center = float(line[1])
                y_center = float(line[2])
                w = float(line[3])
                h = float(line[4])

                # 转换为边界框坐标(x1, y1, x2, y2)
                x1 = (x_center - w / 2) * image_width
                y1 = (y_center - h / 2) * image_height
                x2 = (x_center + w / 2) * image_width
                y2 = (y_center + h / 2) * image_height

                labels.append([class_id, x1, y1, x2, y2])

        return labels

    def write_yolo_labels(self, label_path, labels, image_width, image_height):
        """
        将标签写入YOLO格式的文件，使用真实图像尺寸进行归一化

        Args:
            label_path: 标签文件路径
            labels: 标签列表，每个标签格式为 [class_id, x1, y1, x2, y2]
            image_width: 图像宽度
            image_height: 图像高度
        """
        with open(label_path, 'w') as f:
            for label in labels:
                class_id, x1, y1, x2, y2 = label

                # 计算中心点和宽高
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                # 使用真实图像尺寸进行归一化
                x_center_norm = x_center / image_width
                y_center_norm = y_center / image_height
                w_norm = w / image_width
                h_norm = h / image_height

                # 写入文件，保留6位小数
                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} "
                        f"{w_norm:.6f} {h_norm:.6f}\n")

    def flip_horizontal(self, image, labels):
        """水平翻转"""
        flipped_image = cv2.flip(image, 1)
        height, width = image.shape[:2]

        flipped_labels = []
        for label in labels:
            class_id, x1, y1, x2, y2 = label
            # 水平翻转边界框
            new_x1 = width - x2
            new_x2 = width - x1
            flipped_labels.append([class_id, new_x1, y1, new_x2, y2])

        return flipped_image, flipped_labels

    def flip_vertical(self, image, labels):
        """垂直翻转"""
        flipped_image = cv2.flip(image, 0)
        height, width = image.shape[:2]

        flipped_labels = []
        for label in labels:
            class_id, x1, y1, x2, y2 = label
            # 垂直翻转边界框
            new_y1 = height - y2
            new_y2 = height - y1
            flipped_labels.append([class_id, x1, new_y1, x2, new_y2])

        return flipped_image, flipped_labels

    # def rotate(self, image, labels, angle_range=(-20, 20)):
    #     """随机旋转"""
    #     height, width = image.shape[:2]
    #     angle = random.uniform(*angle_range)
    #
    #     # 计算旋转矩阵
    #     center = (width // 2, height // 2)
    #     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    #
    #     # 执行旋转
    #     cos = np.abs(rotation_matrix[0, 0])
    #     sin = np.abs(rotation_matrix[0, 1])
    #
    #     # 计算新图像的宽度和高度
    #     new_width = int((height * sin) + (width * cos))
    #     new_height = int((height * cos) + (width * sin))
    #
    #     # 调整旋转矩阵的平移部分
    #     rotation_matrix[0, 2] += (new_width / 2) - center[0]
    #     rotation_matrix[1, 2] += (new_height / 2) - center[1]
    #
    #     # 应用旋转
    #     rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    #
    #     # 旋转边界框
    #     rotated_labels = []
    #     for label in labels:
    #         class_id, x1, y1, x2, y2 = label
    #
    #         # 获取四个角点坐标
    #         points = np.array([
    #             [x1, y1],
    #             [x2, y1],
    #             [x2, y2],
    #             [x1, y2]
    #         ], dtype=np.float32)
    #
    #         # 添加一列1，用于仿射变换
    #         points_affine = np.hstack([points, np.ones((4, 1))])
    #
    #         # 应用变换
    #         transformed_points = np.dot(rotation_matrix, points_affine.T).T
    #
    #         # 计算新边界框
    #         min_x = np.min(transformed_points[:, 0])
    #         min_y = np.min(transformed_points[:, 1])
    #         max_x = np.max(transformed_points[:, 0])
    #         max_y = np.max(transformed_points[:, 1])
    #
    #         # 确保边界框在图像范围内
    #         min_x = max(0, min_x)
    #         min_y = max(0, min_y)
    #         max_x = min(new_width - 1, max_x)
    #         max_y = min(new_height - 1, max_y)
    #
    #         rotated_labels.append([class_id, min_x, min_y, max_x, max_y])
    #
    #     return rotated_image, rotated_labels

    def brightness_adjust(self, image, labels, brightness_range=(0.5, 1.5)):
        """亮度调整"""
        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 调整亮度
        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_image = enhancer.enhance(brightness_factor)

        # 转回OpenCV格式
        adjusted_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        # 标签不变
        return adjusted_image, labels

    def contrast_adjust(self, image, labels, contrast_range=(0.5, 1.5)):
        """对比度调整"""
        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 调整对比度
        contrast_factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(contrast_factor)

        # 转回OpenCV格式
        adjusted_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        # 标签不变
        return adjusted_image, labels

    def saturation_adjust(self, image, labels, saturation_range=(0.5, 1.5)):
        """饱和度调整"""
        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 调整饱和度
        saturation_factor = random.uniform(*saturation_range)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced_image = enhancer.enhance(saturation_factor)

        # 转回OpenCV格式
        adjusted_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        # 标签不变
        return adjusted_image, labels

    def hue_adjust(self, image, labels, hue_range=(-15, 15)):
        """色调调整"""
        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 调整色调
        hue_shift = int(random.uniform(*hue_range))
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

        # 转回BGR颜色空间
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # 标签不变
        return adjusted_image, labels

    def add_gaussian_noise(self, image, labels, mean=0, sigma_range=(5, 20)):
        """添加高斯噪声"""
        sigma = random.uniform(*sigma_range)
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # 标签不变
        return noisy, labels

    def add_salt_pepper_noise(self, image, labels, amount_range=(0.001, 0.01)):
        """添加椒盐噪声"""
        amount = random.uniform(*amount_range)
        noisy = np.copy(image)

        # 添加盐噪声
        num_salt = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

        # 添加椒噪声
        num_pepper = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0

        # 标签不变
        return noisy, labels

    def elastic_transform(self, image, labels, alpha_range=(100, 200), sigma_range=(8, 12)):
        """弹性变换"""
        alpha = random.uniform(*alpha_range)
        sigma = random.uniform(*sigma_range)

        height, width = image.shape[:2]

        # 创建随机位移场
        dx = cv2.GaussianBlur(
            (np.random.rand(height, width) * 2 - 1) * alpha,
            (0, 0), sigma)
        dy = cv2.GaussianBlur(
            (np.random.rand(height, width) * 2 - 1) * alpha,
            (0, 0), sigma)

        # 创建网格
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # 应用变换
        transformed_image = cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 对边界框应用相同的变换
        transformed_labels = []
        for label in labels:
            class_id, x1, y1, x2, y2 = label

            # 计算四个角点
            points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            # 应用变换
            transformed_points = np.zeros_like(points)
            for i, (x, y) in enumerate(points):
                x = int(np.clip(x, 0, width - 1))
                y = int(np.clip(y, 0, height - 1))
                transformed_points[i] = [map_x[y, x], map_y[y, x]]

            # 计算新边界框
            min_x = np.min(transformed_points[:, 0])
            min_y = np.min(transformed_points[:, 1])
            max_x = np.max(transformed_points[:, 0])
            max_y = np.max(transformed_points[:, 1])

            # 确保边界框在图像范围内
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(width - 1, max_x)
            max_y = min(height - 1, max_y)

            transformed_labels.append([class_id, min_x, min_y, max_x, max_y])

        return transformed_image, transformed_labels


if __name__ == "__main__":
    # 配置参数
    config = {
        "image_dir": r"F:\2025wheatdata\yolov10\datasets\kernel\images\train",  # 原始图像文件夹
        "label_dir": r"F:\2025wheatdata\yolov10\datasets\kernel\labels\train",  # 原始标签文件夹
        "output_image_dir": r"F:\2025wheatdata\yolov10\datasets\kernel\aug_images",  # 增强后图像保存路径
        "output_label_dir": r"F:\2025wheatdata\yolov10\datasets\kernel\aug_labels",  # 增强后标签保存路径
        "class_names": ["0"],  # 类别名称
        "augment_times": 3,  # 每张图像增强的次数
        "seed": 42  # 随机种子，确保结果可重复
    }

    # 创建增强器实例并处理数据集
    augmentor = YOLODataAugmentor(**config)
    augmentor.process_dataset()

    print("数据集增强完成！")