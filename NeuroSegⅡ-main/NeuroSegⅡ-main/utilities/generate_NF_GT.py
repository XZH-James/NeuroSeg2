#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Convert mat files into png images
Create the Neurofinder Ground Truth Image which can be used in Neuroseg
the download path of GT:
https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/Neurofinder

Written by YuKun Wu
"""
import re
import h5py
import numpy as np
from scipy.io import loadmat
import os
import cv2


for data_type in ['Neurofinder/train', 'Neurofinder/test']:
    data_file = os.path.join('./Markings/', data_type, 'Grader1/')
    Marking_list = os.listdir(data_file)
    for mat in Marking_list:
        # find the id of the neurofinder dataset
        NF_id = re.findall(re.compile(r'\d{5}'), mat)[0]
        mat_path = os.path.join(data_file, mat)
        try:  # If file_name is saved in '-v7.3' format
            mat = h5py.File(mat_path, 'r')
        except OSError:
            mat = loadmat(mat_path)
        if data_type == 'Neurofinder/test/':
            FinalMasks = np.array(mat['FinalMasks']).transpose([2, 0, 1])
        else:
            FinalMasks = np.array(mat['FinalMasks'])
        background = np.zeros(FinalMasks.shape)
        neuron_gt = np.zeros(FinalMasks[0].shape)
        for i in range(FinalMasks.shape[0]):
            neuron = FinalMasks[i]
            xpix, ypix = np.where(neuron == 1.0)
            neuron_gt[xpix, ypix] = 1000 + i
        # save the ground truth in PNG format
        save_path_home = os.path.join('./NeuroSeg_data/', data_type)
        if not os.path.exists(save_path_home):
            os.makedirs(save_path_home)
        if data_type == 'Neurofinder/test/':
            save_path = os.path.join(
                save_path_home, NF_id + '.test_gtFine_instanceIds.png')
        else:
            save_path = os.path.join(
                save_path_home, NF_id + '_gtFine_instanceIds.png')
        cv2.imwrite(save_path, neuron_gt.astype(np.uint16))
