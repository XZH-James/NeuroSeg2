#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Check the modified GT to make sure the masks are correct.

Written by YuKun Wu
"""
import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat
from . import draw_neuron_boundary_with_index
from . import draw_neuron_boundary
data_type = 'Neurofinder/train'
# choose the mode to determine whether to display the index
mode = 'no_index'  # 'index' or 'no_index'
# Set the number of image to display
nf_id = 5
img_list_path = './NeuroSeg2_png/max_intensity_image/'+data_type
img_list =os.listdir(img_list_path)
# the path where save the ground truth
mat_path_home = os.path.join('./Markings_NeuroSeg2', data_type, 'Grader1')
# the path where save the modifed ground truth
save_modified_Markings =os.path.join('./Markings_NeuroSeg2_m', data_type, 'Grader1')
# Making files to store modified Markings
os.makedirs(save_modified_Markings, exist_ok=True)
for img in enumerate(img_list[nf_id:nf_id+1],1):
    img_path = os.path.join(img_list_path,img[1])
    ori_img = cv2.imread(img_path, 1)
    if '_max' in img[1]:
        neuron_id = img[1][:-8]
    else:
        neuron_id = img[1][:-9]  # '_mean'
    print(f"Displaying the image from {data_type.split('/')[0]}—{data_type.split('/')[1]} dataset.\n")
    print(f"number: {img[0]}, ID：{neuron_id}\n")
    mat_name = 'FinalMasks_'+neuron_id +'.mat'
    mat_path = os.path.join(mat_path_home,mat_name)
    try:  # If file_name is saved in '-v7.3' format
        mat = h5py.File(mat_path, 'r')
    except OSError:  # If file_name is not saved in '-v7.3' format
        mat = loadmat(mat_path)
    if data_type.split('/')[1] == 'test':
        FinalMasks = np.array(mat["FinalMasks"]).transpose([2, 1, 0])
    else:
        FinalMasks = np.array(mat["FinalMasks"])
    background = cv2.imread(img_path, 1)
    if mode == 'index':
        neuron_border = draw_neuron_boundary_with_index(background,FinalMasks)
    else:
        neuron_border = draw_neuron_boundary(background, FinalMasks, (0, 0, 175))
    # show the original image and the image with boudary
    Disp = np.hstack((ori_img, neuron_border))
    cv2.namedWindow(f'{neuron_id}', 0)
    cv2.resizeWindow(
        f'{neuron_id}',
        (2 * ori_img.shape[1],
         ori_img.shape[0]))
    cv2.imshow(f'{neuron_id}', Disp)
    cv2.waitKey(0)