#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:45:58 2023

@author: louisethomsen
"""

""" Here are the variables that needs to be adjusted to your path"""


Path_to_speciale_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"



#%%
import pandas as pd
import os 

Path_to_tf = Path_to_speciale_folder + "/tf"
Path_to_tf 

os.chdir(Path_to_tf)

#open csv
depth_segmentation = pd.read_csv('depth_for_segmentation_area.csv', sep = ';')
depth_segmentation = depth_segmentation.drop(["Unnamed: 0"], axis = 1)
depth_bounding_box = pd.read_csv('depth_for_bounding_box_area.csv', sep = ';')
depth_bounding_box = depth_bounding_box.rename(columns={'Unnamed: 0': 'Index'})


depth_segmentation_dilation = pd.read_csv('depth_for_segmentation_dilated.csv', sep = ';')
depth_segmentation_dilation = depth_segmentation_dilation.drop(["Unnamed: 0"], axis = 1)


depth_df = pd.merge(depth_bounding_box, depth_segmentation, on=['patient_id', 'polyp_number', 'label_number'])
depth_df = pd.merge(depth_df,depth_segmentation_dilation, on=['patient_id', 'polyp_number', 'label_number']  )

filename = Path_to_speciale_folder +"/depth_information.csv"



#ratio min/max
depth_df["ratio_surroundings"] = round((depth_df["min_value_box"] / depth_df["max_value_box"]),2)
depth_df["ratio_segmentation"] = round((depth_df["min_value_contour"] / depth_df["max_value_contour"]),2)
depth_df["ratio_segmentation_surround"] = round((depth_df["min_value_contour_area"] / depth_df["max_value_contour_area"]),2)




depth_df.to_csv(filename, index=False)


# Merged together with Finished annotations data

os.chdir(Path_to_speciale_folder)
df_box_data = pd.read_csv("FinishedAnnotations_data.csv")
df_box_data['Label'] = df_box_data['Label'].str.extract('(\d+)').astype(int)
print(df_box_data.columns)

df_depth_data = pd.read_csv(filename)
df_depth_data['polyp_number'] = df_depth_data['polyp_number'].str.extract('(\d+)').astype(int)



df_depth_data = df_depth_data.rename(columns={'patient_id': 'Patient ID', 'polyp_number': 'Polyp', 'label_number':'Label'})
print(df_depth_data.columns)

df_merged = pd.merge(df_box_data, df_depth_data, on=['Patient ID', 'Polyp', 'Label'])
df_merged = df_merged.drop(columns=['Index'])

filename_merged = Path_to_speciale_folder +"/Finished_annotations_depth_information.csv"
df_merged.to_csv(filename_merged, index=False)



