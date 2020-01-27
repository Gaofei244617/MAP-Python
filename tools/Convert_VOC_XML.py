#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
将Pascal VOC格式的所有xml文件中的ground truth box，解析到一个txt文件中
txt文件每行记录一个box, 格式: <image_name> <obj_type> <left> <top> <right> <bottom>
"""

import os
import time
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

############################# 配置参数 ###########################################
# xml文件夹名称
xml_folder_name = "Annotations" 
# 搜索目录
path = "G:/map/map/dataset"
# output_file
output_file = "gt_boxes.txt"
########################################################################

f = open(output_file,"w")
# root: 所指的是当前正在遍历的这个文件夹的本身的地址
# dirs: 是一个list，内容是该文件夹中所有的目录的名字(不包括子目录)
# files: 同样是list, 内容是该文件夹中所有的文件(不包括子目录)
# starttime = time.time()
count = 0
for root, dirs, files in os.walk(path):
    if xml_folder_name in root.split("/") or xml_folder_name in root.split("\\"):
        total = len(files)
        for xml_file in files:
            count = count + 1
            tree = ET.parse(root + "/" + xml_file) # 载入xml文件
            root_node = tree.getroot()  # 获取根节点
            for item in root_node.findall("object"):
                bndbox = item.find("bndbox")
                xmin = bndbox.find("xmin").text 
                ymin = bndbox.find("ymin").text
                xmax = bndbox.find("xmax").text
                ymax = bndbox.find("ymax").text
                obj_type = item.find("name").text
                # <image_name> <obj_type> <left> <top> <bottom> <right>
                line = xml_file[:-4] + " " + obj_type + " " + xmin + " " + ymin + " " + xmax + " " + ymax
                f.write(line + "\n")
            print("%d / %d  %s"%(count, total, xml_file))
f.close()
# endtime = time.time()
# print(endtime - starttime)