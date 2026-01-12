import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
# yolov10项目讲解链接：https://www.bilibili.com/video/BV1Kx4y1H7fL/?spm_id_from=333.999.0.0
# VOC类别名称映射到数字0到19
class_names = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 
    'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 
    'train': 18, 'tvmonitor': 19
}
# class_names=["person","bird"]
TRAIN_RATIO = 80  # 训练集和测试集的比例为 8:2

def clear_hidden_files(path):
    """
    清除隐藏文件
    :param path: 文件路径
    """
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    """
    将标注框从 VOC 格式转换为 YOLO 格式
    :param size: 图像的宽和高
    :param box: 标注框的位置 (xmin, xmax, ymin, ymax)
    :return: 转换后的标注框 (x, y, w, h)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    """
    将单个图像的 XML 标注文件转换为 YOLO 格式的 txt 文件
    :param image_id: 图像ID（文件名）
    """
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id)
    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_names or int(difficult) == 1:
            continue
        cls_id = class_names[cls]  # 使用字典查找类别的 ID
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()

wd = os.getcwd()
data_base_dir = os.path.join(wd, "VOCdevkit/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_space_dir = os.path.join(data_base_dir, "VOC2007/")
if not os.path.isdir(work_space_dir):
    os.mkdir(work_space_dir)
annotation_dir = os.path.join(work_space_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_space_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_space_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov5_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov5_images_dir):
    os.mkdir(yolov5_images_dir)
clear_hidden_files(yolov5_images_dir)
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov5_labels_dir):
    os.mkdir(yolov5_labels_dir)
clear_hidden_files(yolov5_labels_dir)
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
if not os.path.isdir(yolov5_images_train_dir):
    os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_images_test_dir = os.path.join(yolov5_images_dir, "val/")
if not os.path.isdir(yolov5_images_test_dir):
    os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
if not os.path.isdir(yolov5_labels_train_dir):
    os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)
yolov5_labels_test_dir = os.path.join(yolov5_labels_dir, "val/")
if not os.path.isdir(yolov5_labels_test_dir):
    os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)

train_file = open(os.path.join(wd, "yolov5_train.txt"), 'w')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'w')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov5_train.txt"), 'a')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'a')
list_imgs = os.listdir(image_dir)  # 列出所有图像文件
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
        prob = random.randint(1, 100)
        print("Probability: %d" % prob)
        if(prob < TRAIN_RATIO):  # 训练集
            if os.path.exists(annotation_path):
                train_file.write(image_path + '\n')
                convert_annotation(nameWithoutExtention)  # 转换标签
                copyfile(image_path, yolov5_images_train_dir + voc_path)
                copyfile(label_path, yolov5_labels_train_dir + label_name)
        else:  # 测试集
            if os.path.exists(annotation_path):
                test_file.write(image_path + '\n')
                convert_annotation(nameWithoutExtention)  # 转换标签
                copyfile(image_path, yolov5_images_test_dir + voc_path)
                copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()
