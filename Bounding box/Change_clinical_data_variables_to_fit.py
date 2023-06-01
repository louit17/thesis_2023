#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:28:02 2023

@author: louisethomsen
"""

Path_to_folder = '/Users/louisethomsen/Desktop/GitHub/Speciale'



#%%    
#Importation of packages
import pandas as pd
import os

os.chdir(Path_to_folder)

#%%
df = pd.read_excel('Matched polyps 03-05-2023-2.xls')   
df = df.rename(columns={'CCE polyp no': 'CCE_polyp_no'})
df = df.rename(columns={'SDK ID': 'SDK-ID'})
df = df.dropna(subset=['Pathology polyp size'])
df = df[df["Pathology polyp size"] != 0] 
df.to_csv('clinical_data.csv', sep=',', index=False)
