#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:00:30 2023

@author: louisethomsen
"""

import cv2
import numpy as np
import os
from PIL import Image

path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"
path_to_folder_CNN ="/Users/louisethomsen/Desktop"

# from img_mask import create_centre_mask

def create_centre_mask(dims):
    """ Create an image mask to cover rapid reader's writings. """

    h, w = dims

    # Create a circular mask
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w//2, h//2
    r = min(cx, cy)
    cv2.circle(circle_mask,(cx, cy), r, 255, -1)
    inv_circle_mask = cv2.bitwise_not(circle_mask)

    # thres_gray = cv2.threshold(inv_circle_mask, 0, 255, cv2.THRESH_OTSU)[1]
    # gray_inv = cv2.bitwise_not(thres_gray)
    # out1 = cv2.bitwise_or(img, img, mask=gray_inv)


    # Create a rectangular mask
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box_mask, (32, 32), (w-32, h-32), 255, -1)
    inv_box_mask = cv2.bitwise_not(box_mask)
    # thres_gray2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_OTSU)[1]
    # gray_inv2 = cv2.bitwise_not(thres_gray2)


    mask = cv2.bitwise_or(inv_circle_mask, inv_box_mask)
    mask = cv2.bitwise_not(mask)
    boolean_mask = mask.astype(bool)

    return mask, boolean_mask, cx, cy


counter = 0

raw_image_path = r''+str(path_to_folder)+'/tf/input'
for file in os.listdir(raw_image_path):
    if not file.startswith(".DS"):
        image_path = os.path.join(raw_image_path, file)
        mask_image_path = os.path.join(r''+str(path_to_folder_CNN)+'/CNN/Input/Raw_images', str(file))
        
        # Load the image
        img = cv2.imread(image_path)
        
        mask, boolean_mask, cx, ct = create_centre_mask(img.shape[:2])
        combined = cv2.bitwise_or(img, img, mask=mask)
        
        y=25
        x = 25
        h= 576-y
        w=576-x
        cropped_image = combined[x:w, y:h]
        
        
        # Save the image to the destination folder with a new name
        cv2.imwrite(mask_image_path,cropped_image)
        
        counter += 1    
        print(counter)




