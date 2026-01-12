from ultralytics import YOLOv10
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型配置文件
model_yaml_path = r"ultralytics\cfg\models\v10\yolov10m.yaml"
#数据集配置文件
data_yaml_path = 'ultralytics\cfg\datasets\gwhd.yaml'
# data_yaml_path = 'ultralytics\cfg\datasets\kernel.yaml'
# #预训练模型
# pre_model_name = 'yolov10n.pt'
#yolov10项目讲解链接：https://www.bilibili.com/video/BV1Kx4y1H7fL/?spm_id_from=333.999.0.0
if __name__ == '__main__':
    #加载预训练模型
    model = YOLOv10(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          epochs=300,
                          batch=8,
                          workers=0,
                          )