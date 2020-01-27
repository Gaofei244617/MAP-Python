#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np

def voc_ap(rec, prec, map_type):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if map_type == "VOC2007":
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    elif map_type == "VOC2012":
        # correct AP calculation
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

def voc_eval(det_data,
             gt_boxes,
             obj_type,
             iou_thresh,
             map_type):
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
    # sorted_scores = np.sort(-confidence) # 置信度降序排序, 后续未使用此变量
    BB = BB[sorted_ind, :]  # bound box按照置信度降序排序
    image_ids = [image_ids[x] for x in sorted_ind] # imagename按照置信度降序排序

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
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
    ap = voc_ap(rec, prec, map_type)

    return rec, prec, tp, fp, npos-tp, ap
