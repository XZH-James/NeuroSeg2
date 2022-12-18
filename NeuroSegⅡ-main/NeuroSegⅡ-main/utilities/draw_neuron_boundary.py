#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
draw the mask on the image.

Written by YuKun Wu
"""
import cv2
import numpy as np


def draw_boundary(img, FinalMasks, color, enlarge_factor=1, thickness=1):
    """
    Args:
        @param img: Destination image
        @param FinalMasks: An array that is prepared to draw boundaries, [n,h,w]
        @param enlarge_factor: Magnification of the img, int
        @param color: Color of the contours, (0-255,0-255,0-255)
        @param thickness: Thickness of contours lines, int
        If it is negative, the contour interiors are drawn.
    Returns:
        neuron_boundary: np.array, shape = (h,w,3), dtype = uint8
    """
    for index in range(FinalMasks.shape[0]):
        mask = FinalMasks[index]
        mask_png = np.array(mask).astype('uint8')
        h, w = mask_png.shape
        mask_png = cv2.resize(
            mask_png,
            (enlarge_factor * w,
             enlarge_factor * h),
            interpolation=cv2.INTER_NEAREST)
        # Binarize the image
        _, th1 = cv2.threshold(mask_png, 0, 255, cv2.THRESH_BINARY)
        contours, heriachy = cv2.findContours(
            th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        max_contour = []
        # Extract the boundary with the largest length
        for k in range(len(contours)):
            max_contour.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(max_contour))
        neuron_boundary = cv2.drawContours(
            img, contours, max_idx, color, thickness)
    return neuron_boundary
