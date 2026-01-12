from ultralytics import YOLOv10
import os
import cv2
import pandas as pd
from tqdm import tqdm

# # model = YOLOv10("yolov10n.pt")
# #yolov10项目讲解链接：https://www.bilibili.com/video/BV1Kx4y1H7fL/?spm_id_from=333.999.0.0
# model = YOLOv10(r"runs/detect/train/weights/best.pt")
# results = model.predict(r"F:\2025wheatdata\2025大范围出差\黄淮海5月手机照片\spike\np10_spike_1.jpg")
# results[0].show()
# results[0].save("output_image.jpg")

# 配置参数
IMAGE_FOLDER = r"F:\2025wheatdata\2025大范围出差\黄淮海5月手机照片\spike_resize"  # 输入图片文件夹路径
OUTPUT_IMAGE_FOLDER = r"F:\2025wheatdata\2025大范围出差\黄淮海5月手机照片\spike_predict"  # 输出图片文件夹路径
OUTPUT_CSV_PATH = r"F:\2025wheatdata\2025大范围出差\spike_predict.csv"  # 输出CSV表格路径
MODEL_PATH = r"runs/detect/spike_train_300_0701/weights/best.pt"  # 模型路径，可以替换为其他YOLOv8模型
CONFIDENCE_THRESHOLD = 0.3  # 置信度阈值
FONT_SCALE = 1.2  # 字体大小
FONT_THICKNESS = 2  # 字体粗细
BACKGROUND_PADDING = 10  # 文字背景边距

# 创建输出目录
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

# 加载模型
model = YOLOv10(MODEL_PATH)

# 支持的图片格式
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# 获取所有图片文件
image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if os.path.isfile(os.path.join(IMAGE_FOLDER, f))
               and os.path.splitext(f)[1].lower() in IMAGE_EXTS]

# 存储所有图片的统计结果
all_results = []

# 遍历所有图片进行预测
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # 预测
    results = model(image_path)

    # 获取预测结果
    for result in results:
        img = result.orig_img.copy()  # 原始图像
        boxes = result.boxes  # 边界框信息

        # 统计各类目标数量
        object_count = {}

        # 筛选置信度符合要求的边界框
        filtered_boxes = []
        for box in boxes:
            conf = float(box.conf)
            if conf >= CONFIDENCE_THRESHOLD:
                filtered_boxes.append(box)

                # 更新目标统计
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                if cls_name in object_count:
                    object_count[cls_name] += 1
                else:
                    object_count[cls_name] = 1

        # 计算总目标数
        total_objects = sum(object_count.values())

        # 绘制检测框和标签
        for box in filtered_boxes:
            # 边界框坐标
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # 置信度
            conf = float(box.conf)

            # 类别名称
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]

            # 标签文本
            label = f"{cls_name}: {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签背景
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)

            # 绘制标签文本
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 准备统计信息文本
        stats_text = [f"Total objects: {total_objects}"]

        # 计算文本区域的总高度和最大宽度
        max_text_width = 0
        total_text_height = 0
        for text in stats_text:
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            max_text_width = max(max_text_width, text_width)
            total_text_height += text_height

        # 添加行间距
        total_text_height += (len(stats_text) - 1) * 10

        # 计算文本区域的位置和大小
        text_area_width = max_text_width + 2 * BACKGROUND_PADDING
        text_area_height = total_text_height + 2 * BACKGROUND_PADDING
        text_area_x = (img.shape[1] - text_area_width) // 2
        text_area_y = BACKGROUND_PADDING

        # 绘制白色背景
        cv2.rectangle(img, (text_area_x, text_area_y),
                      (text_area_x + text_area_width, text_area_y + text_area_height),
                      (255, 255, 255), -1)

        # 绘制统计信息文本
        y_offset = text_area_y + BACKGROUND_PADDING + int(cv2.getTextSize(stats_text[0],
                                                                          cv2.FONT_HERSHEY_SIMPLEX,
                                                                          FONT_SCALE,
                                                                          FONT_THICKNESS)[0][1])
        for text in stats_text:
            cv2.putText(img, text, (text_area_x + BACKGROUND_PADDING, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
            y_offset += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0][1] + 10

        # 保存结果图像
        output_path = os.path.join(OUTPUT_IMAGE_FOLDER, image_file)
        cv2.imwrite(output_path, img)

        # 记录当前图片的统计结果
        result_dict = {"image_file": image_file, "total_objects": total_objects}
        result_dict.update(object_count)
        all_results.append(result_dict)

# 将统计结果转换为DataFrame并保存为CSV
if all_results:
    df = pd.DataFrame(all_results)

    # 重新排列列，确保image_file和total_objects在前两列
    cols = ['image_file', 'total_objects'] + [col for col in df.columns if col not in ['image_file', 'total_objects']]
    df = df[cols]

    # 保存CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"统计结果已保存到 {OUTPUT_CSV_PATH}")
else:
    print("没有处理任何图片或未检测到目标")