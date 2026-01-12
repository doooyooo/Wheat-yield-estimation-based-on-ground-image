import argparse
import base64
import json
import os
import os.path as osp
 
import imgviz
import PIL.Image
 
from labelme.logger import logger
from labelme import utils


import glob
 
# 最前面加入导包
import yaml
 
 
def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )
 
    parser = argparse.ArgumentParser()
    ###############################################增加的语句##############################
    # parser.add_argument("json_file")
    parser.add_argument("--json_dir",default="D:/2021file/Biye/Mask_RCNN-master/samples/Mydata")
    ###############################################end###################################
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()
 
    ###############################################增加的语句##############################
    assert args.json_dir is not None and len(args.json_dir) > 0
    # json_file = args.json_file
    json_dir = args.json_dir
 
    if osp.isfile(json_dir):
        json_list = [json_dir] if json_dir.endswith('.json') else []
    else:
        json_list = glob.glob(os.path.join(json_dir, '*.json'))
    ###############################################end###################################
 
    for json_file in json_list:
        json_name = osp.basename(json_file).split('.')[0]
        out_dir = args.out if (args.out is not None) else osp.join(osp.dirname(json_file), json_name)
        ###############################################end###################################
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
 
        data = json.load(open(json_file))
        imageData = data.get("imageData")
 
        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)
 
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )
 
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
 
        lbl_viz = imgviz.label2rgb(
            lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
        )
 
        PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
        utils.lblsave(osp.join(out_dir, "label.png"), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))
 
        with open(osp.join(out_dir, "label_names.txt"), "w") as f:
            for lbl_name in label_names:
                f.write(lbl_name + "\n")
 
        logger.info("Saved to: {}".format(out_dir))
        #######
        #增加了yaml生成部分
        logger.warning('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)
        logger.info('Saved to: {}'.format(out_dir))
 
 
 
 
if __name__ == "__main__":
    main()