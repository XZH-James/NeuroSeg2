#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Generate data set lst file

Written by GuanJiang Hang
"""
import os


path = 'hybrid data/train'
filedir = './' + path + '/Img8bit/'
files = os.listdir(filedir)
xml_files = []
for file in files:
    if file[-4:] == '.png':  # Check whether the format is.png
        xml_files.append(file)
xml_files = sorted(xml_files)
f = open('./imglists/' + path + '.lst', "w+")
for i in range(0, len(xml_files)):
    f.write(
        str(i) + '\tImg8bit/' + xml_files[i] + '\tgtFine/' + xml_files[i][:-4] + '_gtFine_instanceIds.png' + '\n')
f.close()
print('<------------finished----------------->')
