#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Convert h5 files to Maximum projected images in batches.

Written by YuKun Wu
"""
import os.path
import tqdm
import h5py
import numpy as np
from scipy.io import loadmat


def h5_maxImg(video_home, group_name):
    """
    Args:
        @param video_home: the path where stores the .h5 files
        @param group_name: the name of the group in .h5 files which store the video
    Returns:
        maxImg_list: a list which store the Maximum projected images generate from .h5 files in order
        maxImg: numpy.ndarray of uint8, shape = (frames,x,y)
    """
    video_list = os.listdir(video_home)
    maxImg_list = []
    for video_id in video_list:
        # Check whether the file is stored in H5 format
        if video_id[-2:] == 'h5':
            video_path = os.path.join(video_home, video_id)
            try:
                f = h5py.File(video_path, 'r')
            except OSError:
                f = loadmat(video_path)
            # Get the group below it based on the level 1 group name
            video = np.array(f[group_name])
            video_frame = video.shape[0]
            # Todo: Generate the maximum projected image
            # Initialize the maximum projected image and convert it to 1D for
            # easy comparison of pixel values
            maxImg = np.concatenate(np.zeros(video[0, :, :].shape), axis=0)
            for frame in tqdm.tqdm(range(video_frame)):
                img = video[frame, :, :]
                ori_img = np.concatenate(img, axis=0)
                for i in range(ori_img.shape[0]):
                    if maxImg[i] < ori_img[i]:
                        maxImg[i] = ori_img[i]
            maxImg = 255 * (maxImg - np.min(maxImg)) / \
                (np.max(maxImg) - np.min(maxImg))  # normalization
            maxImg = maxImg.reshape(video[0, :, :].shape).astype(
                np.uint8)  # reshape to 2D
            maxImg_list.append(maxImg)
    return maxImg_list
