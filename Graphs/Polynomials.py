
""" Define path to speciale folder and select if data and confusion matrix should be printet"""

Path_to_folder = "/Users/louisethomsen/Desktop/GitHub/Speciale"
print_CM = False
save_cm_name = 'name'

"""Polynomial regression on only largest diameter can be found in the bottem of this script"""
#%% Multivariable polynomial 

import pandas as pd
from scipy import stats
#import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import os
#from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

img_save_path = Path_to_folder+"/Graphs/Graphs_images/"

os.chdir(Path_to_folder + "/Graphs")
df = pd.read_csv(Path_to_folder+'/Finished_annotations_depth_information.csv', header = 0, sep = ',')
df = df[df['Pathology polyp size'] > 0]

label_mapping_morphology = {
    'Broad-based/sessile': 0,
    'Flat': 1,
    'Non-polypoid': 2,
    'Not reported': 3,
    'Pedunculated': 4,
    'Tumor': 5
}

df['morphology'] = df['OC Morphology'].map(label_mapping_morphology)

#Polynomial regression function

def polynomial_regression(*args, y, degree=1, x_names=None):
    print_CM = False
    # Combine all input data points into a single 2D array
    X = np.column_stack(args)

    # Create a polynomial feature transformer with given degree
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Transform the input data into polynomial features
    X_poly = poly.fit_transform(X)

    # Fit a linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)
    #print("Intercept: ", model.intercept_)
    #print("Coefficients: ", model.coef_)

    # Predict the values of y for the original data points
    y_pred = model.predict(X_poly)

    # Calculate R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y)
    r_squared = r_value**2
    
    if print_CM == True:
        # Plot the actual vs. predicted values
        plt.scatter(y_pred, y, color='black', marker='o', facecolors='none')
        plt.xlabel('Predicted values')
        plt.ylabel('Actual Values')
        
        plt.title('Polynomial Regression with {} variables'.format(len(X[0])))
    
        
        # Add line equation with R squared on the image
        line_eq = 'y = {:.2f} * x + {:.2f}'.format(slope, intercept)
        combined_variables = 'Parameters = {:.0f}'.format(len(model.coef_))
        plt.text(0.02, 0.95, line_eq, ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.02, 0.75, combined_variables, ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.02, 0.85, 'R-squared: {:.4f}'.format(r_squared), ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.02, 0.65, 'Degree: {:.0f}'.format(degree), ha='left', va='top', transform=plt.gca().transAxes, fontsize=12)
        
    
    
        # Show the names of the input variables on the plot
        if x_names:
            #legend_text = '\n'.join([f'{name}' for name in x_names])
            #plt.legend(title='Input variables', labels=[legend_text], bbox_to_anchor=(1, 1),  loc='center left')
            legend_text = '\n'.join([f'{name}' for name in x_names])
            plt.text(1.01, 0.50, legend_text, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Show the plot
        #plt.show()
    
    return model.intercept_, model.coef_, r_squared, poly, model




def assign_class(size):
    if size <= 5:
        return '1-5 mm'
    elif size >= 6 and size <= 9:
        return '6-9 mm'
    elif size >= 10 and size <= 19:
        return '10-19 mm'
    else:
        return '20+ mm'

def calculate_accuracy_and_f1_score(model_args, args, variable_names, degree):
    global print_CM
    intercept, values, r_squared, poly, model  = polynomial_regression(*model_args, y=train_df_split['Pathology polyp size'], degree=degree, x_names=[*variable_names])
    X_new = np.column_stack(args)
    X_new_poly = poly.transform(X_new)
    y_new_pred = model.predict(X_new_poly)
    val_df_split['Predicted polyp size'] = y_new_pred
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_new_pred, val_df_split['Pathology polyp size'])


    #assign classes
    val_df_split['actual_class'] = val_df_split['Pathology polyp size'].apply(assign_class)
    val_df_split['Predicted_class'] = val_df_split['Predicted polyp size'].apply(assign_class)

    cm = confusion_matrix(val_df_split['actual_class'], val_df_split['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])
    
    class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}
    
    if print_CM == True:
        ### Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
               title='Confusion matrix', #CHANGE title
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
        if print_CM == True:
            plt.savefig(img_save_path + save_cm_name+'.png', dpi=300, bbox_inches='tight')  
            #plt.show()
    
    accuracy = np.diag(cm).sum() / cm.sum()
    actual_one_hot = [class_mapping_one_hot[label] for label in val_df_split['actual_class']]
    predicted_one_hot = [class_mapping_one_hot[label] for label in val_df_split['Predicted_class']]

    # Calculate the F1 score
    f1_score_one_hot= f1_score(actual_one_hot, predicted_one_hot, average='weighted')

    accuracy = accuracy*100
    f1_score_one_hot = f1_score_one_hot*100
    
    return variable_names, degree, accuracy, f1_score_one_hot






# Split dataset
df_split = df
train_df_split, val_df_split = train_test_split(df_split, test_size=0.25, random_state=35)



model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter']
variable_names = ('Smallest diameter', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) #One of the highest scores
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)




model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Box area'], train_df_split['Box circumference']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Box area'], val_df_split['Box circumference']
variable_names = ('Largest diameter', 'Smallest diameter', 'Box area', 'Box circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Fill ratio'], train_df_split['Box circumference']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Fill ratio'], val_df_split['Box circumference']
variable_names = ('Largest diameter', 'Smallest diameter', 'Fill ratio', 'Box circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Fill ratio'], train_df_split['Box W/H ratio']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Fill ratio'], val_df_split['Box W/H ratio']
variable_names = ('Largest diameter', 'Smallest diameter', 'Fill ratio', 'Box W/H ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Contour area'], train_df_split['Contour circumference']
args_val = val_df_split['Contour area'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter']
variable_names = ('Contour area', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Fill ratio']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Fill ratio']
variable_names = ('Contour area', 'Largest diameter', 'Fill ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Smallest diameter'], train_df_split['Contour circumference']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Smallest diameter'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Largest diameter', 'Smallest diameter', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Fill ratio'], train_df_split['Contour circumference'] # one of the highest scores
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Fill ratio'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Largest diameter', 'Fill ratio', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



model_args_train  = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Smallest diameter'], train_df_split['Contour circumference'], train_df_split['Box area'], train_df_split['Box circumference'], train_df_split['Fill ratio'], train_df_split['Box W/H ratio']
args_val  = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Smallest diameter'], val_df_split['Contour circumference'], val_df_split['Box area'], val_df_split['Box circumference'], val_df_split['Fill ratio'], val_df_split['Box W/H ratio']
variable_names = ('Contour area', 'Largest diameter', 'Smallest diameter', 'Contour circumference', 'Box area', 'Box circumference', 'Fill ratio', 'Box W/H ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
#calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3) # to many parameters
#calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)# to many parameters





model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter']
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter']
variable_names = ('Smallest diameter', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Box area'], train_df_split['Box circumference']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Box area'], val_df_split['Box circumference']
variable_names = ('Largest diameter', 'Smallest diameter', 'Box area', 'Box circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Fill ratio'], train_df_split['Box circumference']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Fill ratio'], val_df_split['Box circumference']
variable_names = ('Largest diameter', 'Smallest diameter', 'Fill ratio', 'Box circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Fill ratio'], train_df_split['Box W/H ratio']
args_val = val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Fill ratio'], val_df_split['Box W/H ratio']
variable_names = ('Largest diameter', 'Smallest diameter', 'Fill ratio', 'Box W/H ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Contour area'], train_df_split['Contour circumference']
args_val = val_df_split['Contour area'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter']
variable_names = ('Contour area', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Fill ratio']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Fill ratio']
variable_names = ('Contour area', 'Largest diameter', 'Fill ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Smallest diameter'], train_df_split['Contour circumference']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Smallest diameter'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Largest diameter', 'Smallest diameter', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Fill ratio'], train_df_split['Contour circumference']
args_val = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Fill ratio'], val_df_split['Contour circumference']
variable_names = ('Contour area', 'Largest diameter', 'Fill ratio', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



model_args_train  = train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Smallest diameter'], train_df_split['Contour circumference'], train_df_split['Box area'], train_df_split['Box circumference'], train_df_split['Fill ratio'], train_df_split['Box W/H ratio']
args_val  = val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Smallest diameter'], val_df_split['Contour circumference'], val_df_split['Box area'], val_df_split['Box circumference'], val_df_split['Fill ratio'], val_df_split['Box W/H ratio']
variable_names = ('Contour area', 'Largest diameter', 'Smallest diameter', 'Contour circumference', 'Box area', 'Box circumference', 'Fill ratio', 'Box W/H ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)





#%% Morphology

model_args_train = train_df_split['morphology'], train_df_split['Smallest diameter'], train_df_split['Largest diameter'] #highest morphology
args_val =val_df_split['morphology'], val_df_split['Smallest diameter'], val_df_split['Largest diameter']
variable_names = ('morphology','Smallest diameter', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


model_args_train = train_df_split['morphology'], train_df_split['Largest diameter']
args_val =val_df_split['morphology'], val_df_split['Largest diameter']
variable_names = ('morphology', 'Largest diameter')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)




model_args_train = train_df_split['morphology'], train_df_split['Largest diameter'], train_df_split['Smallest diameter'],  train_df_split['Fill ratio'], train_df_split['Box W/H ratio']
args_val = val_df_split['morphology'], val_df_split['Largest diameter'], val_df_split['Smallest diameter'],  val_df_split['Fill ratio'], val_df_split['Box W/H ratio']
variable_names = ('morphology','Largest diameter', 'Smallest diameter', 'Fill ratio', 'Box W/H ratio')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
#calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4) # to many parameters




model_args_train =train_df_split['morphology'], train_df_split['Contour area'], train_df_split['Largest diameter'], train_df_split['Fill ratio'], train_df_split['Contour circumference'] #highest morphology
args_val = val_df_split['morphology'], val_df_split['Contour area'], val_df_split['Largest diameter'], val_df_split['Fill ratio'], val_df_split['Contour circumference']
variable_names = ('morphology','Contour area', 'Largest diameter', 'Fill ratio', 'Contour circumference')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)






#%% DEPTH
#variables available
#segmentation variables 'mean_value_contour', 'ratio_segmentation', 'contour_pixels_depth'
#Bounding box variables 'mean_value_box', 'ratio_surroundings', 'pixels_depth'
#Outline  'mean_value_contour_area', ' ratio_segmentation_surround', 'contour_area_pixels_depth'

#segmentation
model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['mean_value_contour'], train_df_split['ratio_segmentation'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['mean_value_contour'], val_df_split['ratio_segmentation'] 
variable_names = ('Smallest diameter', 'Largest diameter', 'mean_value_contour', 'ratio_segmentation')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

#Bounding box
model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['mean_value_box'], train_df_split['ratio_surroundings'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['mean_value_box'], val_df_split['ratio_surroundings'] 
variable_names = ('Smallest diameter', 'Largest diameter', 'mean_value_box', 'ratio_surroundings')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)

model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['ratio_surroundings'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['ratio_surroundings'] 
variable_names = ('Smallest diameter', 'Largest diameter', 'ratio_surroundings')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)
    # Doest no affect to remove mean_value

#outline
model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['mean_value_contour_area'], train_df_split['ratio_segmentation_surround'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['mean_value_contour_area'], val_df_split['ratio_segmentation_surround'] 
variable_names = ('Smallest diameter', 'Largest diameter', 'mean_value_contour_area', 'ratio_segmentation_surround')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


# bounding and outline ratio
model_args_train = train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['ratio_surroundings'], train_df_split['ratio_segmentation_surround'] 
args_val = val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['ratio_surroundings'], val_df_split['ratio_segmentation_surround'] 
variable_names = ('Smallest diameter', 'Largest diameter', 'ratio_surroundings', 'ratio_segmentation_surround')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3) 
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)


#morhology and bounding surrounding
model_args_train = train_df_split['morphology'], train_df_split['Smallest diameter'], train_df_split['Largest diameter'], train_df_split['ratio_surroundings'] #highest morphology
args_val =val_df_split['morphology'], val_df_split['Smallest diameter'], val_df_split['Largest diameter'], val_df_split['ratio_surroundings']
variable_names = ('morphology','Smallest diameter', 'Largest diameter','ratio_surroundings')
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 1)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 2)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 3)
calculate_accuracy_and_f1_score(model_args_train, args_val, variable_names, degree = 4)



#%% Polynomial regression on only largest diameter

x = train_df_split['Largest diameter']
y = train_df_split['Pathology polyp size']


for degree in range(1, 5):
    mymodel = np.poly1d(np.polyfit(x, y, degree))
    
    
    x_pred = val_df_split['Largest diameter']
    
    y_new_pred = mymodel(x_pred)
    val_df_split['Predicted polyp size'] = y_new_pred
    
    val_df_split['actual_class'] = val_df_split['Pathology polyp size'].apply(assign_class)
    val_df_split['Predicted_class'] = val_df_split['Predicted polyp size'].apply(assign_class)
    
    
    #assign classes
    val_df_split['actual_class'] =val_df_split['Pathology polyp size'].apply(assign_class)
    val_df_split['Predicted_class'] = val_df_split['Predicted polyp size'].apply(assign_class)
    
    cm = confusion_matrix(val_df_split['actual_class'], val_df_split['Predicted_class'], labels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'])
    
    class_mapping_one_hot = {'1-5 mm': 0, '6-9 mm': 1, '10-19 mm': 2, '20+ mm': 3}
    
    print_CM = True
    
    if print_CM == False:
        ### Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'], yticklabels=['1-5 mm', '6-9 mm', '10-19 mm', '20+ mm'],
               title='Confusion matrix', #CHANGE title
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
        
        #plt.show()
    
    accuracy = np.diag(cm).sum() / cm.sum()
    actual_one_hot = [class_mapping_one_hot[label] for label in val_df_split['actual_class']]
    predicted_one_hot = [class_mapping_one_hot[label] for label in val_df_split['Predicted_class']]
    
    # Calculate the F1 score
    f1_score_one_hot= f1_score(actual_one_hot, predicted_one_hot, average='weighted')
    
    accuracy = accuracy*100
    f1_score_one_hot = f1_score_one_hot*100
    
    print(degree)
    print(accuracy)
    print(f1_score_one_hot)



