#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 09:30:41 2023

@author: louisethomsen
"""


Path_to_folder = '/Users/louisethomsen/Desktop/GitHub/Speciale'

images_you_want_printed = ['raw_10032_P1_Label_1', 'raw_21195_P1_Label_4']

from_which_folders = [ 'output', 'output_segmentation', 'output_segmentation_dilation', 'output_bounding']


#%%

from pathlib import Path
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


Path_to_tf= Path_to_folder + '/tf/'

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale



for i in images_you_want_printed: 
    image = i+'.png'
    image_path = Path_to_tf+'input/'+image
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()


for j in from_which_folders: 
    image = j
    path_list = Path_to_tf + j
    os.chdir(path_list)

    for i in images_you_want_printed: 
        image = i+'.pfm'
        
        print(i)
        image = read_pfm(image)
        
        plt.imshow(image)
        plt.show()



#%% All full CCE 
folder_path = Path_to_folder +'/tf/output'

for filename in os.listdir(folder_path):
    if filename.endswith('.pfm'):
        image_path = os.path.join(folder_path, filename)
        image = read_pfm(image_path)
        
        plt.imshow(image)
        plt.show()




