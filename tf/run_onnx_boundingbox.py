#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 09:36:31 2023

@author: louisethomsen
"""

"""Compute depth maps for images in the input folder.
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
df = pd.DataFrame(columns=['patient_id', 'polyp_number', 'label_number',  "min_value_box", "max_value_box", "mean_value_box", "pixels_depth"])



def create_bound_box_area_mask(dims, name_of_img):
    """ Create an image mask to cover everything except for rapid reader's writings. """
    
    #Should images be printed in console
    global print_img_in_console 
    
    # Extract dimentions in image
    h, w = dims
    
    # Extract the identification for the image we are working with to create a mask from segmentation
    img_name = name_of_img
    img_name_parts = img_name.split('_')

    # Extract the patient ID, polyp number, and label number
    patient_id = img_name_parts[1]
    polyp_number = img_name_parts [2][1:]  
    label_number = img_name_parts[4]  

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
    
    
    """ Using the segmentation image to create a mask """
    
    global  Path_to_tf
    segmentation_folder_path = Path_to_tf + "mask_segmentation"
    
    
    # Create the image path using the given variables
    segmentation_image_path = os.path.join(segmentation_folder_path, f"segmentation_{patient_id}_P{polyp_number}_Label_{label_number}")
    
    
    # Load the image using OpenCV
    img = cv2.imread(segmentation_image_path)
    
    # Check if the image is loaded successfully
    if img is None:
        print("Failed to load image!")
    else:
        # Do something with the image
        pass
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to create a binary mask
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Convert the binary mask to a boolean mask
    boolean_mask_segmentation = thresh.astype(bool)
    
    
    """ Using bounding box image to create a mask """
    bounding_folder_path = Path_to_tf + "mask_boundingbox"
    
    # Create the image path using the given variables
    bounding_image_path = os.path.join(bounding_folder_path, f"Contour_{patient_id}_{polyp_number}_Label_{label_number}")
    
    # Load the image using OpenCV
    img = cv2.imread(bounding_image_path)
    
    # Check if the image is loaded successfully
    if img is None:
        print("Failed to load image!")
    else:
        # Do something with the image
        pass

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of hues to look for. Because we are working with Hsv they will be a dark blue, i could not get this part to work with rgb colors
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([0, 0, 255])

    # Create a mask for the  dark blue pixels
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image
    masked_img = cv2.bitwise_and(img, img, mask=mask_blue)

    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
        # 127 is the threshold value, which is used to separate the pixel values into two groups: those below the threshold and those above it the threshold
        # 255 is the maximum value that can be assigned to a pixel
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle of the largest contour
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    

    # Creates a black mask of the same size as the image that will turn into our box boolean mask
    mask = np.zeros_like(img[:,:,0])  
    # Create a white rectangle on the mask
    mask[y:y+h, x:x+w] = 255
    # Convert the mask to boolean values
    boolean_mask_bounding = mask.astype(bool)
    
    
    # Initialize the surrounding_mask to be all false
    surrounding_mask = np.zeros_like(boolean_mask_segmentation, dtype=bool)
    
    # set the true values where bounding_mask is true and segment_mask is false
    surrounding_mask[np.logical_and(boolean_mask_bounding, ~boolean_mask_segmentation)] = True
    surrounding_mask[boolean_mask == False] = False
    

    return surrounding_mask


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
        
        boolean_mask = create_bound_box_area_mask(img.shape[:2], img_name)

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
        

        #Color mean
        mean_value = np.mean(color_array)
        
        
        #Set the mask color to mean value, the depth colors will then automatically reload
        prediction[np.bitwise_not(boolean_mask)] = mean_value
        print("The final prediction")
        if print_img_in_console == True:
            plt.imshow(prediction)
            plt.show()
        min_value = np.min(prediction)
        max_value = np.max(prediction)
        mean_value_2 = np.mean(prediction)  

        # Find the number of pixels that are visible in the image and not covered by mask
        num_true = np.count_nonzero(boolean_mask)
        

        
        # Add information to df
        global df
        df.loc[len(df)] = [patient_id, polyp_number, label_number, min_value, max_value, mean_value_2, num_true]
        
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
        default='output_bounding',
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

df.to_csv("depth_for_bounding_box_area.csv",sep=';')
