#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 20:28:59 2023

@author: louisethomsen
"""
"""Adjust path to folder"""


Path = "/Users/louisethomsen/Desktop/GitHub/Speciale"

#%%
import os
import pandas as pd
import re
from PIL import Image
import cv2
import numpy as np




os.chdir(Path)

df = pd.read_csv("FinishedAnnotations_data.csv")




def input_images_to_depth_function(path_to_folder, patient_id_folder, polyp_number, label_number):
    counter = 0
    len_img = len(patient_id_folder)
    for patient_id, polyp, label in zip(patient_id_folder, polyp_number, label_number):

### Get the raw image to tf/input
        
        # The raw images are in folders with different names, so these are loop through to find the one that fits each specific patient id folder
        raw_image_folder_names = ['Raw Images', 'Raw material', 'Raw materials']
        for folder_name in raw_image_folder_names:
            raw_image_path = r''+str(path_to_folder)+'/Finished Annotations/'+str(patient_id)+'/P'+str(polyp)+'/'+str(folder_name)
            if os.path.exists(raw_image_path):
                break
        else:
            # If none of the directory exists, an error is printed
            print(f"Error: Raw image directory not found for patient {patient_id}, polyp {polyp}")

            continue
        
        # Finds polyp number of the label, this is done to later take the X number of image in Raw folder that corresponds with specific label
        label_x = int(re.search(r'\d+', label).group())
        
        
        # Counter helps identify the correct image to take
        counter_raw = 0
        
        # Get the raw image into input folder in the tf folder by looping through all files in the folder with jpg format!
        for file in sorted(os.listdir(raw_image_path)):
            #check if the file is a jpg image
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                counter_raw +=1
                
                #Check if the counted image is corresponding with the labeled image
                if counter_raw == label_x:                 
                    #Path to source and destination of image
                    img_raw_path = os.path.join(raw_image_path, file)
                    img_raw_tf_input = os.path.join(r''+str(path_to_folder)+'/tf/input', r'raw_'+str(patient_id)+'_P'+str(polyp)+'_'+str(label) )
                    
                    #Open image and convert to png
                    with Image.open(img_raw_path) as img:
                        img.save(img_raw_tf_input)
                        
                    break
                
### Get the segmentation image to tf/mask_segmentation       
         
        
        # Set path for locating the segmentation image and relocating to tf folder
        segmentation_image_path = r''+str(path_to_folder)+'/Finished Annotations/'+str(patient_id)+'/P'+str(polyp)+'/PixelLabelData/'+str(label)
        mask_segmentation_path = os.path.join(r''+str(path_to_folder)+'/tf/mask_segmentation', r'segmentation_'+str(patient_id)+'_P'+str(polyp)+'_'+str(label))
        
        # Load the image
        #img = Image.open(segmentation_image_path)
        
        img = cv2.imread(segmentation_image_path)
        
        #scaling the image
        img = (img * 255).astype(np.uint8)
        
        #convert the image to grayscale
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        im = Image.fromarray(img1)
        # Save the image to the destination folder with a new name
        im.save(mask_segmentation_path)
        
### Get the image with bounding box to tf/mask_boundingbox
    #    contour_image_path = r''+str(path_to_folder)+'/Contours'
     #   for file in os.listdir(contour_image_path):
      #      if file.endswith(".png"):
        #        each_contour_image_path = os.path.join(contour_image_path , file)
          #      mask_boundingbox_path = os.path.join(r''+str(path_to_folder)+'/tf/mask_boundingbox', r'Contour_'+str(file))
                
                # Load the image
           #     img = Image.open(each_contour_image_path)
                
                # Save the image to the destination folder with a new name
             #   img.save(mask_boundingbox_path)
       # counter += 1        
        #print(f"Image from patient {counter} out of {len_img} is done")
    
### Get the image with bounding box to tf/mask_boundingbox
        contour_image_path = r''+str(path_to_folder)+'/Bounding box/Contours'
        for file in os.listdir(contour_image_path):
            if not file.startswith(".DS"):
                each_contour_image_path = os.path.join(contour_image_path , file)
                mask_boundingbox_path = os.path.join(r''+str(path_to_folder)+'/tf/mask_boundingbox', r'Contour_'+str(file))
                
                # Load the image
                img = Image.open(each_contour_image_path)
                
                # Save the image to the destination folder with a new name
                img.save(mask_boundingbox_path)
        counter += 1        
        print(f"Image from patient {counter} out of {len_img} is done")
        

input_images_to_depth_function(Path, df["Patient ID"], df["Polyp"], df["Label"])
