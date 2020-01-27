#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 获取ground truth boxes
# 入参gt_file：记录gt boxes的文件名 
# 文件每一行：<image_name> <obj_type> <xmin> <ymin> <xmax> <ymax>
# 返回值：[("image_name", "obj_type", xmin, ymin, xmax, ymax), ...]
def get_gtBoxes(gt_file):
    gt_data = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            d = line.strip().split(' ')
            image_name = d[0]
            obj_type = d[1]
            xmin = float(d[2])
            ymin = float(d[3])
            xmax = float(d[4])
            ymax = float(d[5])
            gt_data.append((image_name, obj_type, xmin, ymin, xmax, ymax))
    return gt_data

# 获取检测结果
# 入参det_files：存储detect boxes的文件名的路径，文件以obj_type命名，每个类别一个文件
# 文件每一行：<image_name> <confidence> <xmin> <ymin> <xmax> <ymax>
# 入参class_list：目标类别列表，["obj_type", ...]
# 返回值：[("image_name", "obj_type", confidence, xmin, ymin, xmax, ymax), ...]
def get_detBoxes(det_files, class_list):
    det_data = []
    for obj_type, det_file in zip(class_list, det_files):
        with open(det_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = line.strip().split(' ')
                image_name = d[0]
                confidence = float(d[1])
                xmin = float(d[2])
                ymin = float(d[3])
                xmax = float(d[4])
                ymax = float(d[5])
                det_data.append((image_name, obj_type, confidence, xmin, ymin, xmax, ymax))
    return det_data

