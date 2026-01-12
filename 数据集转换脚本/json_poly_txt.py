# coding:utf-8

import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

"""
1. One row per object
2. Each row is class x1 y1 x2 y2 ... xn yn format.
3. Coordinates must be in normalized format (from 0 - 1). 
   If your coordinates are in pixels, divide each x by image width and each y by image height.
4. Class numbers are zero-indexed (start from 0).
"""
# 本脚本用于将包含多边形标注的JSON文件生成对应的txt文件
# 需要输入图片文件夹位置，json文件夹位置，即将存放的txt文件夹位置
# labelme 中预设的类别名和类别 id 的对应关系
label_idx_map = {"spike": 0, "1": 1, "2": 2, "3": 3, "4": 4}
color_list = [[200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0], [0, 200, 200], [200, 0, 200], [0, 0, 0],
              [128, 128, 0]]


def labelme_to_yolo(img_dir, json_dir, save_dir, img_save_dir, resize_images=True, target_size=1024):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)

    name_list = os.listdir(json_dir)
    for name in name_list:
        if name.startswith('.'):
            continue
        save_path = os.path.join(save_dir, name.replace(".json", ".txt"))
        im_path = os.path.join(img_dir, name.replace(".json", ".jpg"))
        json_path = os.path.join(json_dir, name)
        img_save_path = os.path.join(img_save_dir, name.replace(".json", ".png"))

        # 检查图像文件是否存在
        if not os.path.exists(im_path):
            print(f"警告: 图像文件 {im_path} 不存在，跳过")
            continue

        try:
            # 读取原始图像并获取尺寸
            im = cv2.imread(im_path)
            if im is None:
                print(f"警告: 无法读取图像 {im_path}，跳过")
                continue

            orig_height, orig_width = im.shape[:2]

            # 确定用于归一化的尺寸
            norm_width = target_size if resize_images else orig_width
            norm_height = target_size if resize_images else orig_height

            # 根据参数决定是否调整图像大小
            if resize_images:
                # 调整图像大小
                im_resized = cv2.resize(im, (target_size, target_size))
                cv2.imwrite(img_save_path, im_resized)

                # 计算缩放比例
                scale_x = target_size / orig_width
                scale_y = target_size / orig_height
            else:
                # 不调整图像大小，直接保存原图
                cv2.imwrite(img_save_path, im)

                # 缩放比例为1
                scale_x = 1.0
                scale_y = 1.0

            # 读取JSON文件
            label_dict = json.load(open(json_path, 'r'))
            loc_info_list = label_dict["shapes"]
            label_info_list = list()

            for loc_info in loc_info_list:
                obj_name = loc_info.get("label")
                label_id = label_idx_map.get(obj_name)
                if label_id is None:
                    print(f"警告: 未找到类别 {obj_name} 的映射，跳过此标注")
                    continue

                # 获取多边形的所有点
                points = loc_info.get("points")
                if not points or len(points) < 3:  # 至少需要3个点来形成多边形
                    print(f"警告: 多边形点数不足，跳过此标注")
                    continue

                # 计算多边形的边界框
                points_array = np.array(points, dtype=np.float32)
                x_min, y_min = np.min(points_array, axis=0)
                x_max, y_max = np.max(points_array, axis=0)

                # 计算边界框的中心点和宽高
                box_w = x_max - x_min
                box_h = y_max - y_min
                x_center = x_min + box_w / 2
                y_center = y_min + box_h / 2

                # 根据图像缩放比例调整边界框坐标，并使用正确的归一化尺寸
                x_center_scaled = x_center * scale_x / norm_width
                y_center_scaled = y_center * scale_y / norm_height
                box_w_scaled = box_w * scale_x / norm_width
                box_h_scaled = box_h * scale_y / norm_height

                # 构建YOLO格式的行数据
                label_info = [
                    str(label_id),
                    str(x_center_scaled),
                    str(y_center_scaled),
                    str(box_w_scaled),
                    str(box_h_scaled)
                ]
                label_info_list.append(label_info)

                # 绘制调整后的边界框（用于调试）
                display_img = im_resized if resize_images else im
                x1 = int((x_center_scaled - box_w_scaled / 2) * norm_width)
                y1 = int((y_center_scaled - box_h_scaled / 2) * norm_height)
                x2 = int((x_center_scaled + box_w_scaled / 2) * norm_width)
                y2 = int((y_center_scaled + box_h_scaled / 2) * norm_height)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color_list[label_id % len(color_list)], 2)

            # 将标注信息写入txt文件
            if label_info_list:
                with open(save_path, 'w') as f:
                    for label_info in label_info_list:
                        label_str = ' '.join(label_info)
                        f.write(label_str + '\n')
                print(f"成功生成: {save_path}")
            else:
                print(f"警告: {json_path} 中没有有效标注")

            # 保存带有边界框的图像（用于调试）
            # cv2.imwrite(img_save_path.replace(".png", "_debug.png"), display_img)

        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")


if __name__ == "__main__":
    # 图像文件夹
    image_dir = r"E:\dataset\spike\plantsadd\images"
    # labelme 的标注结果
    json_dir = r"E:\dataset\spike\plantsadd\jsons"
    # yolo 使用的 txt 结果
    save_dir = r"E:\dataset\spike\plantsadd\txt"
    # 调整大小后的图像保存位置
    img_save_dir = r"E:\dataset\spike\plantsadd\images_png"

    # 设置为False可禁用图像调整
    labelme_to_yolo(image_dir, json_dir, save_dir, img_save_dir, resize_images=False)