"""
NeuroSegⅡ
Crop the large - field image to NeuroSegⅡ recognizable size

Written by GuanJiang Hang
"""
import cv2
import numpy as np


def get_start_end(length0,step,overlap):
    x_ = np.arange(0, length0, step - overlap)
    if x_[-2]+step>length0:
        x_ = np.delete(x_,-1)
    x_start = x_
    x_end = x_ + step
    x_end[-1] = np.minimum(x_end[-1], length0)
    return x_start,x_end


filename = ''
step = 128
overlap = 28
im_ori = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
print("shape is %d,%d",im_ori.shape[0],im_ori.shape[1])
im_stacks = []


x_start,x_end = get_start_end(im_ori.shape[1],step,overlap)
dd = np.concatenate(([x_start],[x_end]),0)
xse = dd.T
y_start,y_end = get_start_end(im_ori.shape[0],step,overlap)
dd2 = np.concatenate(([y_start],[y_end]),0)
yse = dd2.T
for ix in xse:
    for iy in yse:
        im_temp = im_ori[iy[0]:iy[1],ix[0]:ix[1]]
        name_save = "%d_%d.tif"%(iy[0],ix[0])
        path="temptifs/"
        print(name_save)
        cv2.imwrite(path+name_save,im_temp)


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1
    return keep
