#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:24:14 2023

@author: louisethomsen
The bounding box and contour creation is inspired from: https://www.tutorialspoint.com/how-to-find-the-bounding-rectangle-of-an-image-contour-in-opencv-python
"""

Path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"



#%%
import cv2
import numpy as np
import os
import re
import pandas as pd


def size_function(Path, image_label, output_path, save_img):
    #Loading the image
    os.chdir(Path)
    id_num = re.findall(r'\d{5}', Path)[0]
    polyp_num = re.findall(r'P\d+', Path)[0]
    polyp_num = polyp_num[1:]
    
    img_label_loaded = image_label
    img = cv2.imread(img_label_loaded)
    
    #scaling the image
    img = (img * 255).astype(np.uint8)
    
    #convert the image to grayscale
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(img1,127,255,0)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    num_of_contours = len(contours)
    
    
    if num_of_contours != 0: 
        #Checks if there are multiple contours on the image, if so choose the largest
        areaArray = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)
            areaLargest = np.argmax(areaArray)
        cnt = contours[areaLargest]
        
        
        ###Contour data 
        
        #Centeroid
        M = cv2.moments(cnt)
        centeroid_x = int(M['m10']/M['m00'])
        centeroid_y = int(M['m01']/M['m00'])
        contour_centeroid = "{},{}".format(centeroid_x, centeroid_y)
        
        #Contour area
        contour_area = cv2.contourArea(cnt)
        
        #Contour circumference
        contour_circumference = cv2.arcLength(cnt,True)
        
    
        ###Bounding box 
        
        #Compute straight bounding rectangle
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.drawContours(img,[cnt],0,(255,255,0),2)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        #Compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect) #box is the four corners of the rectangle
        box = np.int0(box)
        
        # Extract box size and angle from tuple
        box_center, box_size, box_angle = rect
        box_width, box_height = box_size
        
        #Draw rotating bounding box
        img = cv2.drawContours(img,[box],0,(0,255,255),2)
        
        #Define smallest diameter and largest diameter
        if box_width > box_height:
            largest_diameter = box_width
            smallest_diameter = box_height
        else: 
            largest_diameter = box_height
            smallest_diameter = box_width
        
        #Box area
        box_area = box_width * box_height
        
        #Box Circumference
        box_circumference = 2 * (box_width + box_height)
        
        #Box aspect area
        box_aspect_ratio = box_width/box_height
        
        #Extent (object/box area)
        extent_fill = contour_area/box_area

        if save_img == True:
            output_folder = output_path
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, f"{id_num}_{polyp_num}_{img_label_loaded}"), img)
            
        return id_num, polyp_num, img_label_loaded, num_of_contours, contour_centeroid, contour_area, contour_circumference, largest_diameter, smallest_diameter, box_area, box_circumference, box_aspect_ratio, extent_fill

      
    else:
        print(f"Label {img_label_loaded} does not contain a contour")
        



#%%




def list_file(folder_list, polyp_no): 
    global Path_to_folder
    file_info_list = []
    for folder, polyp in zip(folder_list, polyp_no):
        try: 
        
            path = Path_to_folder+'/Finished Annotations/'+str(folder)+'/P'+str(polyp)+'/PixelLabelData'
            path_output = Path_to_folder+'/Bounding box/Contours'
            for labels in os.listdir(path):
                if labels.startswith('Label'):
                    file_info = size_function(path, labels, path_output, True)
                    file_info_list.append(file_info)
                    
        except FileNotFoundError:
                    print(f"{path} does not exist. Skipping.")
                    continue
        
    return file_info_list

file_info_list = []
os.chdir(Path_to_folder+"/Prepare data")
merged_df = pd.read_csv("clinical_FinishedAnnotation_merged.csv")
merged_df = merged_df.drop_duplicates(subset=['SDK-ID', 'CCE_polyp_no'], keep='first')


file_info_list = list_file(merged_df['SDK-ID'].tolist(), merged_df['CCE_polyp_no'].tolist())


df = pd.DataFrame(file_info_list, columns=['Patient ID', 'Polyp', 'Label', 'Number of contours', 'Contour centeroid', 'Contour area', 'Contour circumference', 'Largest diameter', 'Smallest diameter', 'Box area', 'Box circumference', 'Box W/H ratio', 'Fill ratio' ])
df = df.sort_values(by =['Patient ID'])
df = df.sort_values(['Patient ID', 'Polyp', 'Largest diameter'], ascending=[True, True, False])
df_size_function = df.drop_duplicates(subset=['Patient ID', 'Polyp'], keep='first')
df_size_function = df_size_function.dropna(how='any')






#%% Merging the size features together with the provided size information
os.chdir(Path_to_folder+'/Bounding box')

df_size_function.to_csv(Path_to_folder+'/Prepare data/df_size_function.csv', sep=';')
df_size_function['Patient ID'] = df_size_function['Patient ID'].astype('int64')
df_size_function['Polyp'] = df_size_function['Polyp'].astype('int64')


columns_to_open = ['Record ID', 'SDK-ID',  'CCE_polyp_no',
       'OC polyp no', 'Pathology polyp no', 'CCE polyp size', 'CCE Morphology',
       'CCE Location', 'OC polyp size', 'OC Morphology', 'OC Location',
       'Pathology polyp size', 'Histology', 'Dysplasia',
       'Neoplasia/Non-neoplasia']
df_CSV = pd.read_csv(Path_to_folder+'/clinical_data.csv', usecols=columns_to_open)  

df_CSV['SDK-ID'] = df_CSV['SDK-ID'].astype('int64')
df_CSV['CCE_polyp_no'] = df_CSV['CCE_polyp_no'].astype('int64')
df_CSV = df_CSV.dropna(subset=['Pathology polyp size'])

size_merge_df = pd.merge(df_CSV, df_size_function, left_on=['SDK-ID', 'CCE_polyp_no'], right_on=['Patient ID', 'Polyp'], how ='inner')
size_merge_df = size_merge_df.drop_duplicates(subset=['SDK-ID', 'CCE_polyp_no'], keep='first')


def assign_class(size):
    if size <= 5:
        return '0-5 mm'
    elif size >= 6 and size <= 9:
        return '6-9 mm'
    elif size >= 10 and size <= 19:
        return '10-19 mm'
    else:
       return '20+ mm'
    
size_merge_df['class'] = size_merge_df['Pathology polyp size'].apply(assign_class)



size_merge_df.to_csv(Path_to_folder+'/FinishedAnnotations_data.csv')



#%% Keep only the relevant images in contours


# Define the path to the folder containing the contour images
img_folder = Path_to_folder+"/Bounding box/Contours"

# Read in the DataFrame of finished annotations
df = pd.read_csv(Path_to_folder+"/FinishedAnnotations_data.csv")

# Get a list of all image files in the folder
img_files = os.listdir(img_folder)

# Iterate over the image files
for img_file in img_files:
    # Check if the image corresponds to the variables in the DataFrame
    if any(row[['Patient ID', 'Polyp', 'Label']].astype(str).str.cat(sep="_") in img_file for _, row in df.iterrows()):
        # Keep the image file in the folder
        continue
    else:
        # Delete the image file
        os.remove(os.path.join(img_folder, img_file))


#%% See classes
#def assign_class(size):
#    if size <= 5:
#        return '0-5 mm'
#    elif size >= 6 and size <= 9:
#        return '6-9 mm'
#    elif size >= 10 and size <= 19:
#        return '10-19 mm'
#    else:
#       return '20+ mm'
    
#size_merge_df['class'] = size_merge_df['Pathology polyp size'].apply(assign_class)

#class_counts = size_merge_df['class'].value_counts()

#print(class_counts)

