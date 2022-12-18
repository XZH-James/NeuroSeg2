#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
the training program of NeuroSegⅡ
train for hybrid
epoch:500
batch_size:2
"""
import os
import sys
import numpy as np
import cv2

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from neuroseg2.config import Config
from neuroseg2 import utils
import neuroseg2.model as modellib
from neuroseg2 import visualize
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from imgaug import augmenters as iaa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
ImageNet_MODEL_PATH = os.path.join(ROOT_DIR, "models", "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
BRC_MODEL_PATH = os.path.join(ROOT_DIR, "models", "BRC.h5")  # BRC pre-training model

class ShapeConfig(Config):
    NAME = "neurons"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1

    # # Use small images for faster training. Set the limits of the small
    # # side, the large side, and tat determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIS per image because the images are small and
    # have few objects. Aim to allow ROI sampling to pick 33% positive ROIS.
    TRAIN_ROIS_PER_IMAGE = 192
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 16


config = ShapeConfig()
config.display()
print(ShapeConfig)


class ShapesDataset(utils.Dataset):

    def get_path(self, imageset, root_path, data_path):
        self.image_set = imageset
        self.root_path = root_path
        self.data_path = data_path

    def load_image_set_index(self):
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
        # Add classes
        self.add_class("neurons", 1, "neuron")
        count = len(self.image_set_index)
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            self.add_image("neurons", image_id=i, path=self.image_set_index[i],
                           mask_path=self.image_mask_index[i])

    def load_image(self, image_id):
        info = self.image_info[image_id]  # image_info inherits from the dataset in utils
        image_path = os.path.join(self.data_path, self.image_set, info["path"])
        image = (cv2.imread(image_path))
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "neurons":
            return info["neurons"]
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
        mask_all.astype(np.bool)
        class_ids = np.array(class_ids)
        class_ids.astype(np.int32)
        return mask_all, class_ids


# Training dataset

dataset_train = ShapesDataset()

dataset_train.get_path(root_path='./', data_path='hybrid/', imageset='train')

dataset_train.load_image_set_index()

dataset_train.load_shapes()
dataset_train.prepare()

# Val dataset
dataset_val = ShapesDataset()

dataset_val.get_path(root_path='./', data_path='hybrid/', imageset='val')

dataset_val.load_image_set_index()

dataset_val.load_shapes()
dataset_val.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 2)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.NeuroSeg(mode="training", config=config,model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "BRC"  # imagenet, last, or BRC

if init_with == "imagenet":
    model.load_weights(ImageNet_MODEL_PATH, by_name=True)

elif init_with == "BRC":
    model.load_weights(BRC_MODEL_PATH, by_name=True)

elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# augmentation
augmentation = iaa.SomeOf((0, 5), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0)),
])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            augmentation=augmentation,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 5,
            epochs=150,
            augmentation=augmentation,
            layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=200,
            augmentation=augmentation,
            layers="all")
