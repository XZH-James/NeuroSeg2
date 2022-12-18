"""
NeuroSegⅡ
Jointing the segmented large-field image

Written by GuanJiang Hang
"""

import cv2
import numpy as np
import re
import os


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


filename = ''  # original tiff file
npy = ''  # folder saving the temp results, npy file

files = os.listdir(npy)
im_ori = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

print("shape is %d,%d", im_ori.shape[0], im_ori.shape[1])
step = 128
overlap = 28
thresh = 0.4

bboxs_all = np.array([])
scores_all = np.array([])
contours_all = list([])
c_i = 0
from skimage.measure import find_contours

for i in range(len(files)):
    file = files[i]
    file_split = re.split(r'_|\.', file)
    y_start = int(file_split[0])
    x_start = int(file_split[1])
    data_temp = np.load(npy + file, allow_pickle=True).item()
    bboxs = data_temp['rois']
    scores = np.array([data_temp['scores']])
    scores = scores.T
    masks = data_temp['masks']
    for ii in np.arange(masks.shape[2]):
        mask = masks[:, :, ii]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        verts = np.fliplr(contours[0]) - 1
        verts[:, 0] = verts[:, 0] + x_start
        verts[:, 1] = verts[:, 1] + y_start
        contours_all.append(verts)
        c_i = c_i + 1
    if np.size(bboxs) == 0:
        continue

    bboxs[:, 0] = bboxs[:, 0] + y_start
    bboxs[:, 2] = bboxs[:, 2] + y_start
    bboxs[:, 1] = bboxs[:, 1] + x_start
    bboxs[:, 3] = bboxs[:, 3] + x_start
    if bboxs_all.size == 0:
        bboxs_all = bboxs
        scores_all = scores
    else:
        bboxs_all = np.vstack([bboxs_all, bboxs])
        scores_all = np.vstack([scores_all, scores])

dets = np.hstack([bboxs_all, scores_all])


im_ori = cv2.normalize(im_ori, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

image_np = cv2.cvtColor(im_ori, cv2.COLOR_GRAY2BGR)

index_left = py_cpu_nms(dets, thresh)
contours_left = list(contours_all[i] for i in index_left)
for cc in contours_left:
    pts = cc.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image_np, [pts], True, (0, 0, 255), 1)
cv2.imwrite("nms_joint.tiff", image_np)
cv2.imshow('show', image_np)
cv2.waitKey()
