#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
NeuroSegⅡ of the test program for hybrid and Neurofinder

Written by YuKun Wu

"""

import os
import sys

import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)


sys.path.append(ROOT_DIR)  # To find local version of the library
from neuroseg2.config import Config
from neuroseg2 import utils
import neuroseg2.model as modellib
from neuroseg2 import visualize
import os
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# print(MODEL_DIR)

# Directory to save logs and trained model
PREDICT_DIR = os.path.join(ROOT_DIR, "predict")
print(PREDICT_DIR)


class ShapeConfig(Config):
    NAME = "shape"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # batch_size
    NUM_CLASSES = 1 + 1

    # Use small images for faster training. Set the limits of the small
    # side, the large side, and tat determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8,16,32,64) # anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # 8, 16, 32, 64, 128  32, 64, 128, 256, 512

    # Reduce training ROIS per image because the images are small and
    # have few objects. Aim to allow ROI sampling to pick 33% positive ROIS.
    TRAIN_ROIS_PER_IMAGE = 192  # 100->128
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class InferenceConfig(ShapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    ####
    NAME = "shape"
    NUM_CLASSES = 1 + 1  # background + your shapes

    DETECTION_MIN_CONFIDENCE = 0


inference_config = InferenceConfig()
inference_config.display()  ###

# create the model in inference mode
model = modellib.NeuroSeg(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights

model_path = './models/NeuroSeg2.h5'


class ShapesDataset(utils.Dataset):
    """
    Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def get_path(self, imageset, root_path, data_path):
        self.image_set = imageset
        self.root_path = root_path
        self.data_path = data_path

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        """
        image_set_index_file = os.path.join(self.data_path, 'imglists', self.image_set + '.lst')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        image_set_index = []
        image_mask_index = []
        with open(image_set_index_file, 'r') as f:
            for line in f:
                if len(line) > 1:
                    label = line.strip().split('\t')
                    image_set_index.append(label[1])
                    image_mask_index.append(label[2].replace('_labelTrainIds.', '_instanceIds.'))
        self.image_set_index = image_set_index
        self.image_mask_index = image_mask_index
        assert len(self.image_set_index) > 1, "Please check the image set path"

    def load_shapes(self):  # add the path for reading images
        """
        Generate the requested number of synthetic images.

        """
        # Add classes
        self.add_class("neuron", 1, "neuron")
        count = len(self.image_set_index)
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            # bg_color, shapes = self.random_image(height, width)
            self.add_image("neuron", image_id=i, path=self.image_set_index[i],
                           mask_path=self.image_mask_index[i])
            self.image_name = self.image_set_index[i]


    def load_image(self, image_id):
        """
        Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]  # image_info ??utils????dataset????????
        image_path = os.path.join(self.data_path, self.image_set, info["path"])
        image = (cv2.imread(image_path))
        return image

    def image_reference(self, image_id):  # ????????
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "neuron":
            return info["neuron"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        seg_gt = os.path.join(self.data_path, self.image_set, info["mask_path"])
        pixel = cv2.imread(seg_gt, -1)
        mask_all = []
        class_ids = []
        for c in range(1, len(self.class_info)):
            px = np.where((pixel >= c * 1000) & (pixel < (c + 1) * 1000))
            if len(px[0]) == 0:
                continue
            ids = np.unique(pixel[px])
            for id in ids:
                px = np.where(pixel == id)
                x_min = np.min(px[1])
                y_min = np.min(px[0])
                x_max = np.max(px[1])
                y_max = np.max(px[0])
                if x_max - x_min <= 1 or y_max - y_min <= 1:
                    continue
                mask_temp = np.zeros([pixel.shape[0], pixel.shape[1]])
                mask_temp[px] = 1
                mask_all.append(mask_temp)
                class_ids.append(c)
        mask_all = np.array(mask_all)
        mask_all = mask_all.swapaxes(0, 1).swapaxes(1, 2)
        mask_all.astype(np.bool_)
        # mask_all.astype(bool)
        class_ids = np.array(class_ids)

        class_ids.astype(np.int32)
        return mask_all, class_ids


data_set = 'Neurofinder'

# test dataset
dataset_test = ShapesDataset()
dataset_test.get_path(root_path='./', data_path=data_set, imageset='test5')
dataset_test.load_image_set_index()
dataset_test.load_shapes()
dataset_test.prepare()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


from collections import OrderedDict

metric_log = OrderedDict([
    ('ids', []),
    ('iou', []),
    ('precison', []),
    ('recall', []),
    ('f1_score', []),
])

for i in range(len(dataset_test.image_ids)):
    image_id = dataset_test.image_ids[i]
    print(image_id)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)
    results = model.detect([original_image], verbose=1)
    r = results[0]


    # visualize difference between ground truth and original image
    visualize.display_differences(
        original_image, data_set,
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'],
        r['scores'],
        r['masks'],
        dataset_test.class_names, title="", ax=False,
        show_box=False, show_mask=False,
        iou_threshold=0.5, score_threshold=0.5)

    # save img
    figname = str(image_id) + '.png'
    path = './logs/evalution/' + str(data_set)
    if not os.path.exists(path):
        os.makedirs(path)
    figpath = path + '/plt/difference/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figpath = os.path.join(figpath, figname)
    plt.savefig(fname=figpath)
    plt.close()
    # get the metrics of result
    mean_iou, precision, recall, f1 = \
        utils.compute_metrics(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'])
    utils.savelogs(metric_log, path, image_id, mean_iou, precision, recall, f1)
