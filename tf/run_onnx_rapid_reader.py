#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:15:49 2023

@author: louisethomsen
"""


"""Compute depth maps for images in the input folder.
    Output is the depth of the whole photo with rapid reader covered
"""

#variables that should be adapted to local
Path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"
print_img_in_console = False #True or false

#%%

Path_to_tf = Path_to_folder + "/tf/"

import os
os.chdir(Path_to_tf)

import glob
import numpy as np
import utils
import cv2
import argparse
import matplotlib.pyplot as plt
from transforms import Resize, NormalizeImage, PrepareForNet
import onnxruntime as rt
import pandas as pd



def create_circle_mask(dims):
    """ Create an image mask to cover everything except for rapid reader's writings. """
    
    #Should images be printed in console
    global print_img_in_console 
    
    # Extract dimentions in image
    h, w = dims
    

    # Create a circular mask
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w//2, h//2
    r = min(cx, cy)-5 #Minus to decrese the radius a little bit
    cv2.circle(circle_mask,(cx, cy), r, 255, -1)
    inv_circle_mask = cv2.bitwise_not(circle_mask)


    # Create a rectangular mask
    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box_mask, (32, 32), (w-32, h-32), 255, -1)
    inv_box_mask = cv2.bitwise_not(box_mask)


    mask = cv2.bitwise_or(inv_circle_mask, inv_box_mask)
    mask = cv2.bitwise_not(mask)
    boolean_mask = mask.astype(bool)
    
    

    return boolean_mask


def run(input_path, output_path, model_path, model_type="large"):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = "CUDA:0"
    #device = "CPU"
    print("device: %s" % device)

    # network resolution
    if model_type == "large":
        net_w, net_h = 384, 384
    elif model_type == "small":
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    # load network
    print("loading model...")
    model = rt.InferenceSession(model_path)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    resize_image = Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    
    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    transform = compose2(resize_image, PrepareForNet())

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)
        if print_img_in_console == True:
            plt.imshow(img)
            plt.show()
        
        boolean_mask = create_circle_mask(img.shape[:2])

        # Information about patient identification, which is going to be used in the data frame      
        img_name_parts = img_name.split('_')
        # Extract the patient ID, polyp number, and label number
        patient_id = img_name_parts[1]
        polyp_number = img_name_parts [2]  
        label_number = img_name_parts[4][:-4]  
        
        
        img_input = transform({"image": img, "mask": boolean_mask})["image"]


        # compute
        output = model.run([output_name], {input_name: img_input.reshape(1, 3, net_h, net_w).astype(np.float32)})[0]
        prediction = np.array(output).reshape(net_h, net_w)
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        """ If you wanna print the prediction without making the mask the mean color"""
        #print("prediction without mask sat to mean)
        #plt.imshow(prediction)
        #plt.show()
        
        #Find the color array for only the pixels that are true in the boolean array only!
        color_array = prediction[boolean_mask]
        

        #Color values
        min_value = np.min(prediction)
        max_value = np.max(prediction)
        mean_value = np.mean(color_array)
        
        
        #Set the mask color to mean value, the depth colors will then automatically reload
        prediction[np.bitwise_not(boolean_mask)] = mean_value
        print("The final prediction")
        if print_img_in_console == True:
            plt.imshow(prediction)
            plt.show()

        
        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction, bits=2)
        
        

    print("finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default='model-f6b98070.onnx',
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='large',
        help='model type: large or small'
    )

    args = parser.parse_args()

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type)

