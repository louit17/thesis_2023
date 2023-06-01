#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 00:30:29 2023

@author: louisethomsen
"""

Path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"
from sklearn.model_selection import train_test_split

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

def assign_class(size):
    if size <= 5:
        return '1-5 mm'
    elif size >= 6 and size <= 9:
        return '6-9 mm'
    elif size >= 10 and size <= 19:
        return '10-19 mm'
    else:
        return '20+ mm'

img_save_path = Path_to_folder+"/Graphs/Graphs_images"

os.chdir(Path_to_folder + "/Graphs")


#%% Confusion matrix between pathology polyp size and CCE polyp size
df = pd.read_csv(Path_to_folder+'/Finished_annotations_depth_information.csv', header = 0, sep = ',')
df = df[df['Pathology polyp size'] > 0]
df = df.dropna(subset=['CCE polyp size'])


df['actual_class'] = df['Pathology polyp size'].apply(assign_class)
df['Predicted_class'] = df['CCE polyp size'].apply(assign_class)

cm = confusion_matrix(df['actual_class'], df['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

### Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for CCE polyp size', #CHANGE title
       ylabel='Pathology polyp size',
       xlabel='CCE polyp size ')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig(img_save_path + '/CM_CCE_polyp_size.png', dpi=300, bbox_inches='tight')  
plt.show()


accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()

class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}



actual_one_hot = [class_mapping_one_hot[label] for label in df['actual_class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in df['Predicted_class']]

# Calculate the F1 score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print('Confusion Matrix:\n', cm)

print('Accuracy:', accuracy)

print('F1 Score:', f1_score)


#%% Confusion matrix between pathology polyp size and OC polyp size
df = pd.read_csv(Path_to_folder+'/Finished_annotations_depth_information.csv', header = 0, sep = ',')
df = df[df['Pathology polyp size'] > 0]
df = df.dropna(subset=['OC polyp size'])


df['actual_class'] = df['Pathology polyp size'].apply(assign_class)
df['Predicted_class'] = df['OC polyp size'].apply(assign_class)

cm = confusion_matrix(df['actual_class'], df['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

### Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for OC polyp size', #CHANGE title
       ylabel='Pathology polyp size',
       xlabel='OC polyp size ')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig(img_save_path + '/CM_OC_polyp_size.png', dpi=300, bbox_inches='tight')  
plt.show()



accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()

class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}



actual_one_hot = [class_mapping_one_hot[label] for label in df['actual_class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in df['Predicted_class']]

# Calculate the F1 score
from sklearn.metrics import f1_score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print('Confusion Matrix:\n', cm)

print('Accuracy:', accuracy)

print('F1 Score:', f1_score)





#######################################################
#ONLY ON SAME VALIDATION DATA AS POLYNOMIAL EVALUATION#
#######################################################

#%% Confusion matrix between pathology polyp size and OC polyp size only on the validation
df = pd.read_csv(Path_to_folder+'/Finished_annotations_depth_information.csv', header = 0, sep = ',')
df = df[df['Pathology polyp size'] > 0]


train_df_split, val_df_split = train_test_split(df, test_size=0.25, random_state=35)
val_df_split = val_df_split.dropna(subset=['OC polyp size'])

val_df_split['actual_class'] = val_df_split['Pathology polyp size'].apply(assign_class)
val_df_split['Predicted_class'] = val_df_split['OC polyp size'].apply(assign_class)

cm = confusion_matrix(val_df_split['actual_class'], val_df_split['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])
### Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for OC polyp size with validation data', #CHANGE title
       ylabel='Pathology polyp size',
       xlabel='OC polyp size ')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig(img_save_path + '/CM_OC_polyp_size_validation_data.png', dpi=300, bbox_inches='tight')  
plt.show()
accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()
class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}
actual_one_hot = [class_mapping_one_hot[label] for label in val_df_split['actual_class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in val_df_split['Predicted_class']]

# Calculate the F1 score
from sklearn.metrics import f1_score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print("Evaluation of only validation data")
print('Confusion Matrix:\n', cm)
print('Accuracy:', accuracy)
print('F1 Score:', f1_score)


#%% Confusion matrix between pathology polyp size and CCE polyp size
df = pd.read_csv(Path_to_folder+'/Finished_annotations_depth_information.csv', header = 0, sep = ',')
df = df[df['Pathology polyp size'] > 0]
df = df.dropna(subset=['CCE polyp size'])

train_df_split, val_df_split = train_test_split(df, test_size=0.25, random_state=35)
val_df_split = val_df_split.dropna(subset=['CCE polyp size'])


val_df_split['actual_class'] = val_df_split['Pathology polyp size'].apply(assign_class)
val_df_split['Predicted_class'] = val_df_split['CCE polyp size'].apply(assign_class)

cm = confusion_matrix(val_df_split['actual_class'], val_df_split['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

### Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for CCE polyp size with validation data', #CHANGE title
       ylabel='Pathology polyp size',
       xlabel='CCE polyp size ')

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig(img_save_path + '/CM_CCE_polyp_size_validation.png', dpi=300, bbox_inches='tight')  
plt.show()


accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()
class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}
actual_one_hot = [class_mapping_one_hot[label] for label in val_df_split['actual_class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in val_df_split['Predicted_class']]

# Calculate the F1 
from sklearn.metrics import f1_score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print("Evaluation of only validation data")
print('Confusion Matrix:\n', cm)
print('Accuracy:', accuracy)
print('F1 Score:', f1_score)
