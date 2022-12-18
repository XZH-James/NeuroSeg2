#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroSegⅡ
Draw the mask on the image
and displayed the number of neuron in the center of it.

Written by YuKun Wu
"""
import cv2
import numpy as np


def draw_neuron_boundary_with_index(img, FinalMasks, font_size=0.4,thickness=1):
    """
    Args:
        @param img: Destination image
        @param FinalMasks: An array that is prepared to draw boundaries, [n,h,w]
        @param font_size: Size of the neuron number to display, float
        @param thickness: Thickness of contours lines, int
        If it is negative, the contour interiors are drawn.
    Returns:
        neuron_border_with_index: np.array, shape = (h,w,3), dtype = uint8
    """
    for index_ in range(FinalMasks.shape[0]):
        mask = FinalMasks[index_]
        mask_png = np.array(mask).astype('uint8')
        ret, th1 = cv2.threshold(mask_png, 0, 255, cv2.THRESH_BINARY)
        contour, heriachy = cv2.findContours(
            th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contour[0])
        # Check whether the boundary is closed
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            h = np.shape(img)[0]
            w = np.shape(img)[1]
            font_face = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            neuron_border_with_index = cv2.drawContours(img, contour, -1, (0, 175, 0))
            number_size = cv2.getTextSize(str(index_ + 1), font_face, font_size, thickness)
            width = number_size[0][0]
            heigth = number_size[0][1]
            x = center_x - round(width / 2)
            y = center_y + round(heigth / 2)
            if center_x + round(width / 2) > w:
                x = w - width
            elif x < 0:
                x = 0
            elif y > h:
                y = h
            elif center_y - round(heigth / 2) < 0:
                y = heigth
            cv2.putText(neuron_border_with_index, str(index_ + 1),
                        (x, y), font_face, font_size, (0, 0, 255), thickness)
        else:
            print(f"The {index_ + 1} outline is open and needs to be removed")
    return neuron_border_with_index
