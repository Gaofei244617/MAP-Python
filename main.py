#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import voc_eval as ap
import time

########################## Config ######################################
config_file = "config.xml"
################################################################

# 计算map
ret = ap.cal_map_all(config_file)

# output表格
scenes = list(ret.keys())
obj_types = []
for scene_name, obj in ret.items():
    for obj_type in obj.keys():
        if not obj_type in obj_types:
            obj_types.append(obj_type)
scenes.append("Average")
obj_types.append("Average")
data = np.full((len(scenes), len(obj_types)), np.nan, dtype = float)

# output
for scene_name, obj in ret.items():
    for obj_type, item in obj.items():
        ap = item[6]
        x = scenes.index(scene_name)
        y = obj_types.index(obj_type)
        data[x][y] = ap

# 求平均值       
for i in range(len(scenes)-1):
    ds = [d for d in data[i, :-1] if d > 0 and d != np.nan]
    if len(ds) > 0:
        avg = np.mean(ds)
        data[i, -1] = avg

for i in range(len(obj_types)):
    ds = [d for d in data[:, i] if d > 0 and d != np.nan]
    if len(ds) > 0:
        avg = np.mean(ds)
        data[-1, i] = avg

dt = pd.DataFrame(data, index = scenes, columns = obj_types)
pd.set_option('display.width', None)
print(dt.T)