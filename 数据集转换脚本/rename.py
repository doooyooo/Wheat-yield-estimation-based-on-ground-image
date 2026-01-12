import os
import re

# 本代码用于修改文件名称,用于找到文件括号里面的数字作为新的文件名

def rename_files_in_folder(folder_path):
    # 正则表达式，用于匹配括号中的数字
    pattern = re.compile(r'\((\d+)\)')

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否是文件（而不是目录）
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            # 使用正则表达式查找括号中的数字
            match = pattern.search(filename)
            if match:
                new_name = match.group(1) + os.path.splitext(filename)[1]  # 保留文件扩展名
                new_file_path = os.path.join(folder_path, new_name)
                os.rename(file_path, new_file_path)
                print(f'Renamed: {filename} -> {new_name}')
            else:
                print(f'No match found in: {filename}')

# 使用示例
folder_path = 'bones_alljson'  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)
