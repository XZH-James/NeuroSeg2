#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Visualize and modify GT from STNeuroNet

Written by YuKun Wu
"""
import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat, savemat
from . import Disp_neuron_boundary
from . import Disp_neuron_boundary_with_index
data_type = 'Neurofinder/train'
img_list_path = os.path.join('./NeuroSeg_png/mean_image', data_type)
img_list = os.listdir(img_list_path)
# the path where save the ground truth
mat_path_home = os.path.join('./Markings_NeuroSeg', data_type, 'Grader1')
# the path where save the modifed ground truth
save_modified_Markings =os.path.join('./Markings_NeuroSeg_m', data_type, 'Grader1')
# Making files to store modified Markings
os.makedirs(save_modified_Markings, exist_ok=True)
log = []
# Set the number of image to change
nf_id = 5
# Set the number of image to be modified at a time
num = 1
for img in enumerate(img_list[nf_id:nf_id + num], 1):
    img_path = os.path.join(img_list_path, img[1])
    # Extract the number of the data
    ori_img = cv2.imread(img_path, 1)
    if '_max' in img[1]:
        neuron_id = img[1][:-8]
    else:  # '_mean'
        neuron_id = img[1][:-4]
    print(f"Modifying the image from {data_type.split('/')[0]}—{data_type.split('/')[1]} dataset.\n")
    print(f"number: {img[0]}, ID：{neuron_id}\n")
    mat_name = 'FinalMasks' + neuron_id + '.mat'
    mat_path = os.path.join(mat_path_home, mat_name)
    try:  # If file_name is saved in '-v7.3' format
        mat = h5py.File(mat_path, 'r')
    except OSError:
        mat = loadmat(mat_path)
    if data_type.split('/')[1] == 'test':
        FinalMasks = np.array(mat["FinalMasks"]).transpose([2, 1, 0])
    else:
        FinalMasks = np.array(mat["FinalMasks"])

    background1 = cv2.imread(img_path, 1)
    nb_no_index = Disp_neuron_boundary(background1, FinalMasks, (0, 0, 175))
    background2 = cv2.imread(img_path, 1)
    nb_with_index = Disp_neuron_boundary_with_index(background2, FinalMasks)

    Disp = np.hstack((nb_with_index,ori_img,nb_no_index))
    # show neuron boudary with index, original image, neuron boudary no index in turn
    cv2.namedWindow(f'{neuron_id}', 0)
    cv2.resizeWindow(
        f'{neuron_id}',
        (3 * ori_img.shape[1],
         ori_img.shape[0]))
    cv2.imshow(f'{neuron_id}', Disp)
    cv2.waitKey(0)

    print("Please input the id of the neuron to be removed:")
    delet_neuron_list = input()
    delet_neuron_array_ = np.unique(
        np.array(delet_neuron_list.split(","),dtype=int))  # Prevent duplicate input
    delet_neuron_array = delet_neuron_array_ - \
        np.ones(delet_neuron_array_.shape, dtype=int)
    # Saving modified Markings in .mat
    modified_mat_path = os.path.join(save_modified_Markings, mat_name)
    FinalMasks_deleted = np.delete(FinalMasks, delet_neuron_array, axis=0)

    background3 = cv2.imread(img_path, 1)
    nb_modified = Disp_neuron_boundary(background3, FinalMasks_deleted)

    Disp_finish = np.hstack((nb_with_index, nb_modified))
    # Compare the picture of neuron boudary with index to the modified image
    cv2.namedWindow(f'{neuron_id}_modefied', 0)
    cv2.resizeWindow(
        f'{neuron_id}_modefied',
        (2 * ori_img.shape[1],
         ori_img.shape[0]))
    cv2.imshow(f'{neuron_id}_modefied', Disp_finish)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    savemat(modified_mat_path, mdict={'FinalMasks': FinalMasks_deleted})
    message = [neuron_id] + sorted(delet_neuron_array_.tolist())
    log.append(message)
    log_path = os.path.join(
        save_modified_Markings,
        f"{data_type.split('/')[0]}_{data_type.split('/')[1]}.csv")
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write(f'{neuron_id},{log[img[0]-1][1:]}\n')
            f.close()
    else:
        with open(log_path, 'w') as f:
            f.write('neuron_id,deleted neurons\n')
            f.write(f'{neuron_id},{log[img[0]-1][1:]}\n')
            f.close()
print("END of DELETING")
