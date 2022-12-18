#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Save the Neurofinder dataset in the format of H5

Written by ZheHao Xu
"""
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm


def tiff_h5(root_dir, save_path, group_name):
    """
    Args:
        @param root_dir: the path where stores the .tiff files
        @param save_path: the path to save the .h5 dataset
        @param group_name: Name of the dataset, example:'data','mov'
    Output:
        nf_set(HDF5 dataset, shape = (frames,x,y), dtype = 'uint16')
        """
    pics = sorted(os.listdir(root_dir))
    video_data = []
    for pic in tqdm(pics):
        # In case there is a picture can not be opened
        try:
            video_data.append(plt.imread(root_dir + '\\' + pic).tolist())
        except Exception:
            print(pic + " pic wrong")
    nf_set = h5py.File(save_path, 'w')
    nf_set.create_dataset(nf_set, group_name, video_data, dtype='uint16')
    nf_set.close()
