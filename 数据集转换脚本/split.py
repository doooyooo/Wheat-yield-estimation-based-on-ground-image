import os
import shutil

# 定义输入路径
labels_path = r'D:\项目合集\ultralyticsSWIM\data\labels'
images_path = r'D:\项目合集\ultralyticsSWIM\data\images'

# 确保路径存在
if not os.path.exists(labels_path):
    raise ValueError(f"Labels path {labels_path} does not exist.")
if not os.path.exists(images_path):
    raise ValueError(f"Images path {images_path} does not exist.")

# 创建train和val文件夹
for folder in ['train', 'val']:
    train_folder_path = os.path.join(labels_path, folder)
    val_folder_path = os.path.join(images_path, folder)
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(val_folder_path, exist_ok=True)
    print(f"Created {folder} folder at: {train_folder_path}")
    print(f"Created {folder} folder at: {val_folder_path}")

# 获取所有的txt文件和对应的jpg文件
txt_files = sorted([f for f in os.listdir(labels_path) if f.endswith('.txt') and os.path.isfile(os.path.join(labels_path, f))])
jpg_files = sorted([f for f in os.listdir(images_path) if f.endswith('.JPG') and os.path.isfile(os.path.join(images_path, f))])

print("TXT files found:")
print(txt_files)
print("JPG files found:")
print(jpg_files)

# 确保文件名称一一对应
txt_basenames = [os.path.splitext(f)[0] for f in txt_files]
jpg_basenames = [os.path.splitext(f)[0] for f in jpg_files]
matching_basenames = set(txt_basenames).intersection(set(jpg_basenames))

print("Matching basenames:")
print(matching_basenames)

# 根据匹配的文件名过滤文件列表
txt_files = [f + '.txt' for f in matching_basenames]
jpg_files = [f + '.jpg' for f in matching_basenames]

print("Filtered TXT files:")
print(txt_files)
print("Filtered JPG files:")
print(jpg_files)

# 按照顺序划分数据集
num_files = len(txt_files)
split_index = int(num_files * 0.9)

train_txt_files = txt_files[:split_index]
val_txt_files = txt_files[split_index:]

print("Train TXT files:")
print(train_txt_files)
print("Val TXT files:")
print(val_txt_files)

# 将文件复制到对应的文件夹
for txt_file in train_txt_files:
    base_name = os.path.splitext(txt_file)[0]
    jpg_file = base_name + '.jpg'
    
    # 复制txt文件
    shutil.copy(os.path.join(labels_path, txt_file), os.path.join(labels_path, 'train', txt_file))
    # 复制jpg文件
    shutil.copy(os.path.join(images_path, jpg_file), os.path.join(images_path, 'train', jpg_file))
    print(f"Copied {txt_file} and {jpg_file} to train folder.")

for txt_file in val_txt_files:
    base_name = os.path.splitext(txt_file)[0]
    jpg_file = base_name + '.jpg'
    
    # 复制txt文件
    shutil.copy(os.path.join(labels_path, txt_file), os.path.join(labels_path, 'val', txt_file))
    # 复制jpg文件
    shutil.copy(os.path.join(images_path, jpg_file), os.path.join(images_path, 'val', jpg_file))
    print(f"Copied {txt_file} and {jpg_file} to val folder.")

print("文件已成功划分并复制到对应的文件夹中。")
