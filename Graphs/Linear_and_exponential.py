#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:04:25 2023

@author: louisethomsen
"""
Path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"

print_plots = False

#%% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from tabulate import tabulate
import matplotlib.patches as mpatches
from sklearn.metrics import f1_score




#%%
# Load in dataset
os.chdir(Path_to_folder)

# Read in dataset
data = pd.read_csv("FinishedAnnotations_data.csv", sep = ',')
data = data[data['Pathology polyp size'] > 0]


#%% Linear regression - Pathology polyp size vs polyp size in pixels

img_save_path = Path_to_folder+"/Graphs/Graphs_images"


# Create the plot
plt.scatter(data['Largest diameter'], data['Pathology polyp size'], color='black', marker='o', facecolors='none')
#plt.title('Pathology polyp size vs pixels')
plt.xlabel('Polyp length in pixels')
plt.ylabel('Pathology polyp size in mm')


x = data['Largest diameter']
y = data['Pathology polyp size']

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
r_squared = r_value**2
line_eq = 'y = ' + str(round(slope,2)) + ' * x + ' + str(round(intercept,2))
print(line_eq)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x+b)


plt.text(0.02, 0.95, line_eq, ha='left', va='top', transform=plt.gca().transAxes, fontsize=12) # Add line equation on image
plt.text(0.02, 0.85, 'R-squared: {:.4f}'.format(r_squared) , ha='left', va='top', transform=plt.gca().transAxes, fontsize=12) # Add R squared on image
#plt.text(50, 60, 'R-squared: {:.4f}'.format(r_squared), fontsize=12)
plt.savefig(img_save_path + '/Pathology polyp size vs pixels.png', dpi=300, bbox_inches='tight')

plt.show()

#%% Making the plottet true in categories plot

df_category1 = data[data['Pathology polyp size'] < 6]
df_category2 = data[(data['Pathology polyp size'] > 5) & (data['Pathology polyp size'] < 10)]
df_category3 = data[(data['Pathology polyp size'] > 9) & (data['Pathology polyp size'] < 20)]
df_category4 = data[data['Pathology polyp size'] > 19]
len_df = len(df_category1)+len(df_category2)+len(df_category3)+len(df_category4)
if len_df != len(data):
    print("There is a mistake in the dividing")


#Category 1-5 mm
x_category_1 = (5-intercept)/slope
df_category1_true = df_category1[df_category1['Largest diameter'] <= x_category_1]
df_category1_false = df_category1[df_category1['Largest diameter'] > x_category_1]
category1_overestimated = len(df_category1_false)

#Category 6-9 mm
x_category_2 = (9-intercept)/slope
df_category2_true = df_category2[(df_category2['Largest diameter'] > x_category_1) & (df_category2['Largest diameter'] <=x_category_2)]
df_category2_false = pd.concat([df_category2,df_category2_true]).drop_duplicates(keep=False)
category2_underestimated = len(df_category2_false[df_category2_false['Largest diameter'] < x_category_1])
category2_overestimated = len(df_category2_false[df_category2_false['Largest diameter'] > x_category_2])


#Category 10-19 mm
x_category_3 = (19-intercept)/slope
df_category3_true = df_category3[(df_category3['Largest diameter'] > x_category_2) & (df_category3['Largest diameter'] <=x_category_3)]
df_category3_false = pd.concat([df_category3,df_category3_true]).drop_duplicates(keep=False)
category3_underestimated = len(df_category3_false[df_category3_false['Largest diameter'] < x_category_2])
category3_overestimated= len(df_category3_false[df_category3_false['Largest diameter'] > x_category_3])


#Category 20+ mm
df_category4_true = df_category4[df_category4['Largest diameter'] > x_category_3]
df_category4_false = df_category4[df_category4['Largest diameter'] <= x_category_3]

category4_underestimated = len(df_category4_false)

total_true = len(df_category1_true+df_category2_true+df_category3_true+df_category4_true)
total_false = len(df_category1_false+df_category2_false+df_category3_false+df_category4_false)



#Information table

table_data = [["Number of patients", len(df_category1), len(df_category2), len(df_category3), len(df_category4), len_df],
              ["Correctly classifed", len(df_category1_true), len(df_category2_true), len(df_category3_true), len(df_category4_true), total_true],
              ["Misclassifed", len(df_category1_false), len(df_category2_false), len(df_category3_false), len(df_category4_false), total_false],
              ["Underestimiated", 0, category2_underestimated , category3_underestimated , category4_underestimated , (category2_underestimated +category3_underestimated + category4_underestimated)],
              ["Overestimated", category1_overestimated, category2_overestimated, category3_overestimated, 0, (category1_overestimated+category2_overestimated+category3_overestimated) ],
              ["Error rate",(len(df_category1_false )/len(df_category1)) ,(len(df_category2_false )/len(df_category2)) ,(len(df_category3_false )/len(df_category3)) ,(len(df_category4_false )/len(df_category4)) , (total_false/(total_false+total_true))]]

col_table = ["", "1-5 mm", "6-9 mm", "10-19 mm", "20+ mm", "Total"]
data_table = tabulate(table_data, headers = col_table)

print(data_table)

#Plot
df_total_false = pd.concat([df_category1_false,df_category2_false,df_category3_false,df_category4_false], axis = 0)

plt.figure(figsize=(10,6))

# scatterplot for category 1 true
x1  = df_category1_true['Largest diameter']
y1 = df_category1_true['Pathology polyp size']
plt.scatter(x1, y1, color='green', marker='o', facecolors='none')

# x and y plot for category 2 true
x2  = df_category2_true['Largest diameter']
y2 = df_category2_true['Pathology polyp size']
plt.scatter(x2, y2, color='green', marker='o', facecolors='none')

# scatterplot for category 3 true
x3  = df_category3_true['Largest diameter']
y3 = df_category3_true['Pathology polyp size']
plt.scatter(x3, y3, color='green', marker='o', facecolors='none')

# x and y plot for category 4 true
x4  = df_category4_true['Largest diameter']
y4 = df_category4_true['Pathology polyp size']
plt.scatter(x4, y4, color='green', marker='o', facecolors='none')

# scatterplot for all wrongly classifed polyps
x5  = df_total_false['Largest diameter']
y5 = df_total_false['Pathology polyp size']
plt.scatter(x5, y5, color='black', marker='o', facecolors='none')


plt.xlabel('Polyp length in pixels')
plt.ylabel('Pathology polyp size in mm')

plt.axhline(y=5, color='black', linestyle='--', lw =0.5)
plt.axhline(y=9, color='black', linestyle='--', lw =0.5)
plt.axhline(y=19, color='black', linestyle='--', lw =0.5)
plt.vlines(x=[x_category_1, x_category_2, x_category_3], ymin=[0, 0, 0], ymax=[19, 19, 19], colors='black', ls='--', lw=0.5)

plt.savefig(img_save_path + '/Correctly classified.png', dpi=300, bbox_inches='tight')


plt.show()



#%% Over and under estimation plot

plt.figure(figsize=(10,6))
plt.scatter(data['Largest diameter'], data['Pathology polyp size'], color='black', marker='o', facecolors='none')
plt.xlabel('Polyp length in pixels')
plt.ylabel('Pathology polyp size in mm')
plt.xlim(0)

y_max = data['Pathology polyp size'].max()
x_max = data['Largest diameter'].max()
plt.ylim([0, (y_max+5)])

plt.axhline(y=5, color='black', linestyle='--', lw =0.5)
plt.axhline(y=9, color='black', linestyle='--', lw =0.5)
plt.axhline(y=19, color='black', linestyle='--', lw =0.5)
plt.vlines(x=[x_category_1, x_category_2, x_category_3], ymin=[0, 0, 0], ymax=[19, 19, 19], colors='black', ls='--', lw=0.5)

# add underestimated area
left, bottom, width, height = (0, 19, x_category_3, (y_max-19+5))
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="purple")
plt.gca().add_patch(rect)

left, bottom, width, height = (0, 9, x_category_2, (19-9))
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="purple")
plt.gca().add_patch(rect)
 
left, bottom, width, height = (0, 5, x_category_1, (9-5))
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="purple")
plt.gca().add_patch(rect)


# add overestimated area
left, bottom, width, height = (x_category_1, 0, (x_category_2-x_category_1), 5)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="green")
plt.gca().add_patch(rect)
left, bottom, width, height = (x_category_2, 0, (x_category_3-x_category_2), 9)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="green")
plt.gca().add_patch(rect)
left, bottom, width, height = (x_category_3, 0, (x_max), 19)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        #fill=False,
                        alpha=0.2,
                        #color="purple",
                       #linewidth=2,
                       facecolor="green")
plt.gca().add_patch(rect)

plt.text((x_category_1/3), (y_max/2),'Underestimated polyps',fontsize=12, color="purple")
plt.text((x_max/3*2), 6,'Overestimated polyps',fontsize=12, color="green")
plt.savefig(img_save_path + '/Over-under-classified.png', dpi=300, bbox_inches='tight')       
plt.show()



#%% Confusion matrix for linear regression
def assign_class_pixels(size):
    global x_category_1
    global x_category_2
    global x_category_3
    
    if size <= x_category_1:
        return '1-5 mm'
    elif size > x_category_1 and size <= x_category_2:
        return '6-9 mm'
    elif size > x_category_2 and size <= x_category_3:
        return '10-19 mm'
    else:
        return '20+ mm'
    

data['Predicted_class'] = data['Largest diameter'].apply(assign_class_pixels)
data['class'] = data['class'].replace('0-5 mm', '1-5 mm') #Rename because i used the wrong label in csv.
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])


# Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for linear regression',
       ylabel='True label',
       xlabel='Predicted label')

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
plt.savefig(img_save_path + '/CM_linear.png', dpi=300, bbox_inches='tight')   
plt.show()



# Evaluation
from sklearn.metrics import confusion_matrix, classification_report

# Compute confusion matrix
cm = confusion_matrix(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

# Compute classification report
cr = classification_report(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

# Calculate accuracy
accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()

class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}

actual_one_hot = [class_mapping_one_hot[label] for label in data['class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in data['Predicted_class']]

# Calculate the F1 score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print('Confusion Matrix:\n', cm)
print('Classification Report:\n', cr)
print('Accuracy:', accuracy)
print('F1 Score:', f1_score)


# Check if consusion matrix is correct with the labels
value_counts = data['class'].value_counts()
print(value_counts)
value_counts = data['Predicted_class'].value_counts()
print(value_counts)




#%% Exponential regression
"""Code is inspired by https://rowannicholls.github.io/python/curve_fitting/exponential.html """

# Fit data to an exponential regression
p = np.polyfit(x, np.log(y), 1)
a = np.exp(p[1])
b = p[0]

# R-squared value for the exponential regression
ss_residuals = np.sum((y - a * np.exp(b * x))**2)
ss_total = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_residuals / ss_total)

# Generate small the small data points long the curve, that is added to display the line
xp = np.linspace(min(x), max(x), 100)
yp = a * np.exp(b * xp)

# Plot the scatterplot and the regression
plt.scatter(x, y, color='black', marker='o', facecolors='none')
plt.plot(xp, yp, color='blue')

# Add the regression equaton on the plot
equation = 'y = {:.2f} * exp({:.2f} * x)'.format(a, b)
plt.text(0.02, 0.95, equation, ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.02, 0.85, 'R-squared: {:.4f}'.format(r_squared), ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)


plt.xlabel('Polyp length in pixels')
plt.ylabel('Pathology polyp size in mm')
#plt.title('Exponential regression')

plt.savefig(img_save_path + '/exponential_regression.png', dpi=300, bbox_inches='tight')   


plt.show()

import math
#Category 1-5 mm
x_category_1 = 100*math.log(5)/math.log(2.29)


#Category 6-9 mm
x_category_2 = 100*math.log(9)/math.log(2.29)

#Category 10-19 mm
x_category_3 =100*math.log(19)/math.log(2.29)




def assign_class_pixels(size):
    global x_category_1
    global x_category_2
    global x_category_3
    
    if size <= x_category_1:
        return '1-5 mm'
    elif size > x_category_1 and size <= x_category_2:
        return '6-9 mm'
    elif size > x_category_2 and size <= x_category_3:
        return '10-19 mm'
    else:
        return '20+ mm'
    

data['Predicted_class'] = data['Largest diameter'].apply(assign_class_pixels)
data['class'] = data['class'].replace('0-5 mm', '1-5 mm')

from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])


# Plot the confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
       title='Confusion matrix for exponential regression',
       ylabel='True label',
       xlabel='Predicted label')

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
plt.savefig(img_save_path + '/CM_exponential.png', dpi=300, bbox_inches='tight')   
plt.show()




# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

# Compute confusion matrix
cm = confusion_matrix(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])

# Compute classification report
cr = classification_report(data['class'], data['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])
accuracy = (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / cm.sum()

class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}
data['class'] = data['class'].replace('0-5 mm', '1-5 mm')
actual_one_hot = [class_mapping_one_hot[label] for label in data['class']]
predicted_one_hot = [class_mapping_one_hot[label] for label in data['Predicted_class']]

# Calculate the F1 score
f1_score= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

print('Confusion Matrix:\n', cm)
print('Classification Report:\n', cr)
print('Accuracy:', accuracy)

print('F1 Score:', f1_score)


# Check if consusion matrix is correct with the labels
value_counts = data['class'].value_counts()
print(value_counts)
value_counts = data['Predicted_class'].value_counts()
print(value_counts)










