import os

def get_unique_files(folder1, folder2):
    # 获取文件夹中的文件列表
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    
    # 提取文件名（不包括扩展名）并存储在集合中
    file_names1 = {os.path.splitext(file)[0] for file in files1}
    file_names2 = {os.path.splitext(file)[0] for file in files2}
    
    # 找出文件夹1中未重名的文件名
    unique_files1 = file_names1 - file_names2
    
    # 找出文件夹2中未重名的文件名
    unique_files2 = file_names2 - file_names1
    
    # 删除文件夹1中未重名的json文件
    for file_name in unique_files1:
        file_path = os.path.join(folder1, file_name + ".json")
        print("Deleting file:", file_path)  # 添加调试输出
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("File does not exist:", file_path)
    
    # 删除文件夹2中未重名的jpg文件
    for file_name in unique_files2:
        file_path = os.path.join(folder2, file_name + ".jpg")
        print("Deleting file:", file_path)  # 添加调试输出
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print("File does not exist:", file_path)
    
    return unique_files1, unique_files2

# 输入两个文件夹的路径
folder1_path = r"D:\项目合集\数据预处理\bones_alljson"
folder2_path = r"D:\项目合集\数据预处理\bones_allimages"

# 调用函数并输出结果
unique_files1, unique_files2 = get_unique_files(folder1_path, folder2_path)
print("文件夹1中未重名的文件名:")
for file_name in unique_files1:
    print(file_name)

print("\n文件夹2中未重名的文件名:")
for file_name in unique_files2:
    print(file_name)
