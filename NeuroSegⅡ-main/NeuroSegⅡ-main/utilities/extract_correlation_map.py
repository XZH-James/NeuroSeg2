#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Extract correlation map from the result of Suite2P
and save it in in the case of PNG(8-bit),TIFF format.

Written by YuKun Wu
"""
import os
import numpy as np
import cv2
import imageio
Data = 'Neurofinder'
neuron_idlist = ['100', '101', '200', '201', '401']
for Layer in ['train','test']:
    for experiment_id in neuron_idlist:
        if Layer == 'test':
            neuron_id = '0' + experiment_id[0] + '.' + experiment_id[1:] + '.test'
        else:
            neuron_id = '0' + experiment_id[0] + '.' + experiment_id[1:]
        cellpose_model = 'cyto'
        class_ = 'with_classifier'
        # load the result of Suite2P
        ops = np.load(f'./{Data}_{class_}/{Layer}/{neuron_id}/suite2p/plane0/ops.npy',
            allow_pickle=True).item()
        # Extract correlation map
        img = np.array(ops['Vcorr'], dtype='float32')
        # Save the reult in TIFF format
        correlation_dir = f'./correlation map_TIFF/{Data}/{Layer}'
        os.makedirs(correlation_dir, exist_ok=True)
        img_path = f'{correlation_dir}/{neuron_id}.tiff'
        imageio.imsave(img_path, img)
        # normalization
        nor_img = 255 * (img - np.min(img)) / \
            (np.max(img) - np.min(img))
        img_PNG = np.uint8(nor_img)
        # Save the reult in PNG(8-bit) format
        correlation_dir_ = f'./correlation map_PNG/{Data}/{Layer}'
        os.makedirs(correlation_dir_, exist_ok=True)
        img_path_png = f'{correlation_dir_}/{neuron_id}.png'
        cv2.imwrite(img_path_png, img_PNG)
