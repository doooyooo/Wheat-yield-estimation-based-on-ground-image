import os
import shutil

# 定义路径
image_folder = "images/"
label_folder = "labels/"
train_image_folder = "dataset/image/train/"
val_image_folder = "dataset/image/val/"
train_label_folder = "dataset/label/train/"
val_label_folder = "dataset/label/val/"

# # 检查并创建目标文件夹
# if not os.path.exists(train_image_folder):
#     os.makedirs(train_image_folder)
# if not os.path.exists(val_image_folder):
#     os.makedirs(val_image_folder)
# if not os.path.exists(train_label_folder):
#     os.makedirs(train_label_folder)
# if not os.path.exists(val_label_folder):
#     os.makedirs(val_label_folder)

# 获取image文件夹内的所有jpg文件和labels文件夹内的所有txt文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

# 获取前5000张jpg图片和对应的txt文件，其余的放到val文件夹中
train_image_files = image_files[:5000]
val_image_files = image_files[5000:]

# 获取前5000个txt文件，其余的放到val文件夹中
train_label_files = label_files[:5000]
val_label_files = label_files[5000:]

# 复制图像文件和标签文件到train和val文件夹中
for image_file, label_file in zip(train_image_files, train_label_files):
    source_image_path = os.path.join(image_folder, image_file)
    source_label_path = os.path.join(label_folder, label_file)
    dest_image_path = os.path.join(train_image_folder, image_file)
    dest_label_path = os.path.join(train_label_folder, label_file)

    shutil.copy(source_image_path, dest_image_path)
    shutil.copy(source_label_path, dest_label_path)

for image_file, label_file in zip(val_image_files, val_label_files):
    source_image_path = os.path.join(image_folder, image_file)
    source_label_path = os.path.join(label_folder, label_file)
    dest_image_path = os.path.join(val_image_folder, image_file)
    dest_label_path = os.path.join(val_label_folder, label_file)

    shutil.copy(source_image_path, dest_image_path)
    shutil.copy(source_label_path, dest_label_path)
