import os
import shutil
from PIL import Image
import numpy as np
#判定两个文件夹内的文件是否相同
file_path1 = r"E:\dataset\spike\plantsadd\jsons/"# 已知 内容较少的文件夹+/
file_path2 = r"E:\dataset\spike\plantsadd\images/"
temp_path = r"E:\dataset\kernel\temp/"# 临时文件夹，存放被移除的数据
os.makedirs(r"E:\dataset\kernel\temp/",exist_ok=True)# 临时文件夹，存放被移除的数据
# 获取第一个文件夹中的所有文件名（去除后缀）
f1 = []
for filename in os.listdir(file_path1):
    # 分割文件名和扩展名
    name_without_ext = os.path.splitext(filename)[0]
    f1.append(name_without_ext)

# 将两个文件夹内的文件名不同的提出来（基于去除后缀后的名称）
for filename2 in os.listdir(file_path2):
    # 获取第二个文件夹中当前文件的名称（去除后缀）
    name2_without_ext = os.path.splitext(filename2)[0]
    # 如果在第一个文件夹中找不到对应的文件名（去除后缀后），则移动该文件
    if name2_without_ext not in f1:
        shutil.move(os.path.join(file_path2, filename2), os.path.join(temp_path, filename2))  # 文件夹需要创建

shutil.rmtree(temp_path)
