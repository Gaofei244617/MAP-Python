#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
将COCO格式的所有json文件中的ground truth box，解析到一个txt文件中
txt文件每行记录一个box, 格式: <image_name> <obj_type> <left> <top> <right> <bottom>
"""

import json
import time

############################# 配置参数 #################################
# 搜索目录
json_file = "H:/dataset/COCO/annotations/instances_train2017.json"
# output_file
output_file = "gt_boxes.txt"
########################################################################

in_f = open(json_file, "r")
data = json.load(in_f) # 将json文件解析为一个字典



# 解析成 {id:"image_name"}
image_dic = {}
for key, value in data.items():
    if key == "images":
        for img in value:
            image_dic[img["id"]] = img["file_name"][:-4]

# 解析成 {id:"category_name"}
category_dic = {}
for key, value in data.items():
    if key == "categories":
        for cat in value:
            category_dic[cat["id"]] = cat["name"]

# 解析ground truth box信息, 输出到txt文件
out_f = open(output_file, "w")
for key, value in data.items():
    if key == "annotations":
        for box in value:
            image_name = image_dic[box["image_id"]]
            obj_type = category_dic[box["category_id"]]
            # COCO标注左上角坐标和宽高 ("bbox": [x, y, width, height])
            xmin = str(box["bbox"][0])
            ymin = str(box["bbox"][1])
            xmax = xmin + str(box["bbox"][2])
            ymax = ymin +str(box["bbox"][3])
            line = image_name + " " + obj_type + " " + xmin + " " + ymin + " " + xmax + " " + ymax
            out_f.write(line + "\n")
            print(image_name)
out_f.close()
