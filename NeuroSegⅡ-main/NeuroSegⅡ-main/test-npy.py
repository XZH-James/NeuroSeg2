# -*- coding: utf-8 -*-

"""
NeuroSegⅡ
NeuroSegⅡ of the test program for mesoscopic two-photon Ca2+ imaging
The coordinate value after reading each image segmentation is stored in npy format

Written by ZheHao Xu

"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import pandas as pd
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import time
from neuroseg2.config import Config
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from neuroseg2 import utils
import neuroseg2.model as modellib
from neuroseg2 import visualize


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "models/NeuroSeg2.h5")
print(ROOT_DIR)
# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)
    print("***********************")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "BV/test/gtFine")  ## gtFine
print(IMAGE_DIR)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shape"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


# import train_tongue

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.NeuroSeg(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', '']
# Load a random image from the images folder

file_names = next(os.walk(IMAGE_DIR))[2]
# print(file_names)
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# image = skimage.io.imread(os.path.abspath(IMAGE_DIR))

#####################Batch image prediction####################
count = os.listdir(IMAGE_DIR)

from collections import OrderedDict

metric_log = OrderedDict([
    ('ids', []),
    ('rois', []),
    ('masks', []),
    ('scores', []),
])

for i in range(0, len(count)):
    path = os.path.join(IMAGE_DIR, count[i])  #  Graph reading path
    file = ''  #  Save data path
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
        Cmax = np.max(image[:])
        Cmin = np.min(image[:])
        image = np.uint8((np.double(image) - Cmin) / (Cmax - Cmin) * 255)

        image = np.array(image)
        img1 = image.reshape(image.shape[0], image.shape[1], 1)
        img2 = image.reshape(image.shape[0], image.shape[1], 1)
        img3 = image.reshape(image.shape[0], image.shape[1], 1)
        image = cv2.merge([img1, img2, img3])

        # Run detection
        results = model.detect([image], verbose=1)
        print(results)
        r = results[0]
        visualize.display_instances_(count[i], image, r['rois'], r['masks'], r['class_ids'],
                                     class_names, r['scores'], ax=False,
                                     show_mask=False, show_bbox=False)
    print(count[i])


    np.set_printoptions(threshold=np.inf)

    rois = r['rois']
    masks = r['masks']
    scores = r['scores']

    log = utils.savelogs1(metric_log, file, count[i], rois, masks, scores)
    print(masks.shape)

    dict_save = {"rois": rois, "masks": masks, "scores": scores}
    np.save(count[i] + '.npy', dict_save)

