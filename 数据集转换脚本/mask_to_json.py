import cv2  # 导入OpenCV库进行图像处理
import os   # 导入os模块以与操作系统交互
import json  # 导入json模块以处理JSON数据
import sys  # 导入sys模块以访问命令行参数


def func(file: str) -> dict:
    # 使用OpenCV读取图像文件
    png = cv2.imread(file)
    
    # 将图像转换为灰度图
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    
    # 对灰度图应用二值化阈值处理
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # 在二值图像中查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个包含图像元数据的字典
    dic = {
        "version": "5.0.1",
        "flags": {},
        "shapes": list(),
        "imagePath": os.path.basename(file),
        "imageHeight": png.shape[0],
        "imageWidth": png.shape[1]
    }

    # 遍历轮廓并提取多边形点
    for contour in contours:
        temp = list()
        for point in contour[2:]:
            if len(temp) > 1 and temp[-2][0] * temp[-2][1] * int(point[0][0]) * int(point[0][1]) != 0 and (int(point[0][0]) - temp[-2][0]) * (temp[-1][1] - temp[-2][1]) == (int(point[0][1]) - temp[-2][1]) * (temp[-1][0] - temp[-1][0]):
                temp[-1][0] = int(point[0][0])
                temp[-1][1] = int(point[0][1])
            else:
                temp.append([int(point[0][0]), int(point[0][1])])
        
        # 将多边形信息添加到字典中
        dic["shapes"].append({
            "label": "result",
            "points": temp,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    # 返回最终的字典
    return dic

if __name__ == "__main__":
    # 检查命令行参数的数量是否不等于3
    if len(sys.argv) != 3:
        raise ValueError("mask文件或目录 输出路径")

    # 检查输入路径是否为目录
    if os.path.isdir(sys.argv[1]):
        # 遍历目录中的每个文件
        for file in os.listdir(sys.argv[1]):
            # 在写模式下打开一个JSON文件，并使用func函数处理图像
            with open(os.path.join(sys.argv[2], os.path.splitext(file)[0] + ".json"), mode='w', encoding="utf-8") as f:
                json.dump(func(os.path.join(sys.argv[1], file)), f)
    else:
        # 在写模式下打开一个JSON文件，并使用func函数处理图像
        with open(os.path.join(sys.argv[2], os.path.splitext(os.path.basename(sys.argv[1]))[0] + ".json"), mode='w', encoding="utf-8") as f:
            json.dump(func(sys.argv[1]), f)


