#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Convert h5 files to Mean value images in batches.

Written by YuKun Wu
"""
import os.path
import cv2
import tqdm
import h5py
import numpy as np
import scipy
from scipy.io import loadmat


def h5_meanImg(video_home, group_name):
    """
    Args:
        @param video_home: the path where stores the .h5 files
        @param group_name: the name of the group in .h5 files which store the video
    Returns:
        meanImg_list: a list which store the Mean value images generate from .h5 files in order
        meanImg: numpy.ndarray of uint8, shape = (frames,x,y)
    """
    video_list = os.listdir(video_home)
    meanImg_list = []
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
            # Todo: Generate mean image
            meanImg = np.concatenate(video.sum(axis=0))
            meanImg = 255 * (meanImg - np.min(meanImg)) / \
                (np.max(meanImg) - np.min(meanImg))  # normalization
            meanImg = meanImg.reshape(video[0, :, :].shape).astype(
                np.uint8)  # reshape to 2D
            meanImg_list.append(meanImg)
    return meanImg_list
