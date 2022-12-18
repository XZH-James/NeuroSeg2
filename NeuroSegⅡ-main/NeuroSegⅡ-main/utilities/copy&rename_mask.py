#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Script to copy and rename the Ground Truth in batch,
which can be easily used in the delet GT process.


Written by YuKun Wu
"""
import os
import shutil


def copy_files(datatype, ori_path, path_new):
    for filename in os.listdir(ori_path):
        if datatype == 'ABO':
            if '_FPremoved' in filename:
                for i in range(4):
                    group = filename.replace(
                        '_FPremoved', '').split('.mat')[0] + f'_{i}.mat'
                    shutil.copyfile(
                        os.path.join(
                            ori_path, filename), os.path.join(
                            path_new, group))
                    print(filename, "copied as", group)
        else:  # Neurofinder
            for i in range(7):
                if 'test' in ori_path:
                    group = filename.split('.mat')[0] + '_test' + f'_{i}.mat'
                else:
                    group = filename.split('.mat')[0] + f'_{i}.mat'
                shutil.copyfile(
                    os.path.join(
                        ori_path, filename), os.path.join(
                        path_new, group))
                print(filename, "copied as", group)


if __name__ == '__main__':
    src_dir = os.getcwd()
    for NF_type in ['train', 'test']:
        mat_path = os.path.join(
            src_dir,
            'Markings/Neurofinder',
            NF_type,
            'Grader1')
        NeuroSeg_mask = os.path.join(
            src_dir, 'Markings/Neurofinder', NF_type, 'Grader1')
        os.makedirs(NeuroSeg_mask, exist_ok=True)
        copy_files('Neurofinder', mat_path, NeuroSeg_mask)
