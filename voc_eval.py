#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import get_data as dat

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# AP VOC2007
def cal_ap_VOC2007(rec, prec):
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
    return ap

# AP VOC2012
def cal_ap_VOC2012(rec, prec):
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(det_data, gt_boxes, obj_type, iou, map_type):
    """
    Top level function that does the PASCAL VOC evaluation.
    assume det_data: [("imagename", "obj_type", confidenc, xmin, ymin, xmax, ymax), ...]
    assume gt_boxes: [("imagename", "obj_type", xmin, ymin, xmax, ymax), ...]
    obj_type: Category name 
    [iou_thresh]: Overlap threshold (default = 0.5)
    [map_type]: To use VOC2007's 11 point AP or VOC2012 AP
    """
    # recs : {"imagename": [{"obj_type":"obj_type", "bbox":[xmin,ymin,xmax,ymax]}, ...]}
    recs = {} 
    for item in gt_boxes:
        if item[0] in recs.keys():
            recs[item[0]].append({"obj_type":item[1], "bbox":[item[2], item[3], item[4], item[5]]})
        else:
            recs[item[0]] = [{"obj_type":item[1], "bbox":[item[2], item[3], item[4], item[5]]}]
    
    # extract gt objects for this class
    class_recs = {}   # {"imagename" : {'bbox': [[xmin,ymin,xmax,ymax],...], 'det': [False,...]}}
    npos = 0    # gt box的数量
    for imagename in recs.keys():
        # [{"obj_type":"obj_type", "bbox":[xmin,ymin,xmax,ymax]}, ...], obj_type只有一类
        R = [obj for obj in recs[imagename] if obj['obj_type'] == obj_type]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det}

    # read dets
    image_ids = [x[0] for x in det_data if x[1] == obj_type]  # imagename list
    confidence = np.array([x[2] for x in det_data if x[1] == obj_type]) # 置信度list
    BB = np.array([x[3:] for x in det_data if x[1] == obj_type]) # [[xmin,ymin,xmax,ymax], ...]

    # sort by confidence
    sorted_ind = np.argsort(-confidence) # 置信度list索引按照置信度降序排序
    sorted_scores = np.sort(-confidence) # 置信度降序排序, 后续未使用此变量
    BB = BB[sorted_ind, :]  # bound box按照置信度降序排序
    image_ids = [image_ids[x] for x in sorted_ind] # imagename按照置信度降序排序

    # go down dets and mark TPs and FPs
    if map_type == "COCO":
        iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    else:
        iou_list = [iou]

    ap = 0.0
    mrec = 0.0
    mprec = 0.0  
    for iou_thresh in iou_list:
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            R["det"] = [False] * len(R["det"])
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute IoU
                # 相交区域
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni    # IoU
                ovmax = np.max(overlaps)   # 最大交并比
                jmax = np.argmax(overlaps)
            
            # 如果最大IoU大于阈值
            if ovmax > iou_thresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        mrec = mrec + rec / len(iou_list)
        mprec = mprec + prec / len(iou_list)
        if map_type == "VOC2007":
            p = cal_ap_VOC2007(rec, prec)
        else:
            p = cal_ap_VOC2012(rec, prec)
        ap = ap + p / len(iou_list)

    if map_type == "COCO":
        return sorted_scores, np.nan, np.nan, np.nan, mrec, mprec, ap

    return sorted_scores, tp, fp, npos-tp, mrec, mprec, ap

def cal_map_all(config_file):
    tree = ET.parse(config_file)
    map_type = tree.find("map_type").text  # map计算方式
    map_iou_thresh = float(tree.find("map_iou_thresh").text) # map IoU阈值
    scenes = tree.findall("scene")
    # 计算所有mAP相关数据 {"scene":{"obj_type":(conf, tp, fp, fn, rec, prec, ap)}}
    ret = {} 
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
            conf, tp, fp, fn, rec, prec, ap = voc_eval(det_data, gt_data, obj_type, map_iou_thresh, map_type)
            if scene_name in ret.keys():
                ret[scene_name][obj_type] = (conf, tp, fp, fn, rec, prec, ap)   
            else:
                ret[scene_name] = {obj_type: (conf, tp, fp, fn, rec, prec, ap)}
    return ret

