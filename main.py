#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import voc_eval as voc
import get_data as dat
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

################################################################
config_file = "config.xml"
################################################################

tree = ET.parse(config_file)
map_type = tree.find("map_type").text  # map计算方式
map_iou_thresh = float(tree.find("map_iou_thresh").text) # map IoU阈值
scenes = tree.findall("scene")
ret = {} # 所有mAP相关数据
for scene in scenes:
    gt_file = scene.find("gt_box").text
    scene_name = scene.attrib["name"]
    det_files = [] 
    class_list = []
    for dt_file in scene.findall("detection"):
        det_files.append(dt_file.text)
        class_list.append(dt_file.attrib["name"])
    # gt_data : [("image_name", "obj_type", xmin, ymin, xmax, ymax), ...]
    gt_data = dat.get_gtBoxes(gt_file) # 从txt文件解析gt boxes数据
    # det_data：[("image_name", "obj_type", confidence, xmin, ymin, xmax, ymax), ...]
    det_data = dat.get_detBoxes(det_files, class_list) # 从txt文件解析detect boxes数据
    for obj_type in class_list:
        # 计算AP相关数据
        conf, tp, fp, fn, rec, prec, ap = voc.voc_eval(det_data, gt_data, obj_type, map_iou_thresh, map_type)
        if scene_name in ret.keys():
            ret[scene_name][obj_type] = (conf, tp, fp, fn, rec, prec, ap)   
        else:
            ret[scene_name] = {obj_type: (conf, tp, fp, fn, rec, prec, ap)}

# output
for scene_name, obj in ret.items():
    for obj_type, item in obj.items():
        print("%s %s %f"%(scene_name, obj_type, item[6])) 