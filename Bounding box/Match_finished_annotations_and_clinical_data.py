
""" The file is used to match patient ID and polyp number from the csv containing the clinical data with 
    the finished annotations folder that consists of maps with the Patient ID number, inside these there 
    a map with a polyp number, that identifies each polyps a patient can have. Inside these there can multiple
    images of the specific polyp from different angles as well as the clinically drawn polyp segmentation """


Path_to_folder = '/Users/louisethomsen/Desktop/GitHub/Speciale'



#%%    
#Importation of packages
import cv2
import pandas as pd
import os

os.chdir(Path_to_folder)
Path_to_folder = '/Users/louisethomsen/Desktop/GitHub/Speciale'



#%%
df = pd.read_csv('clinical_data.csv', sep = ",")   

#%%

# Define path to finished annotations folder
path_to_FA = Path_to_folder+'/Finished Annotations'

os.chdir(path_to_FA)
FA_df = pd.DataFrame(columns=['Patient_Finished_annotations', 'Polyp_number_Finished_annotations'])

# Iterate through folders in X
for folder_z in os.listdir(path_to_FA):
    if os.path.isdir(os.path.join(path_to_FA, folder_z)):
        # Iterate through folders in Z
        for folder_y in os.listdir(os.path.join(path_to_FA, folder_z)):
            if os.path.isdir(os.path.join(path_to_FA, folder_z, folder_y)):
                if len(folder_y) < 5:
                # Append to DataFrame
                    FA_df = FA_df.append({'Patient_Finished_annotations': folder_z, 'Polyp_number_Finished_annotations': folder_y}, ignore_index=True)

#Remove P and operations in polyp number to make compareable to clincal data
Polyp_number_FA_df =[y.replace('P', '') for y in FA_df['Polyp_number_Finished_annotations']]
Polyp_number_FA_df =[y.replace('+', '0') for y in Polyp_number_FA_df] #This is done because we have in have a value that is P3+4
Polyp_number_FA_df =[y.replace('-', '0') for y in Polyp_number_FA_df]
Polyp_number_FA_df = [int(i) for i in Polyp_number_FA_df]
FA_df['Polyp_number'] = Polyp_number_FA_df
FA_df = FA_df.rename(columns={'Patient_Finished_annotations': 'SDK-ID', 'Polyp_number': 'CCE_polyp_no'})
FA_df['SDK-ID'] = FA_df['SDK-ID'].astype(int)
FA_df['CCE_polyp_no'] = FA_df['CCE_polyp_no'].astype(int)

filename = Path_to_folder + '/Prepare data/Finished_annotations_identifiers.csv'


FA_df.to_csv(filename)


#COMPARE clinical data csv and finished annotation 

os.chdir(Path_to_folder)


# Make dictionary from CSV file with size measurements
df = pd.read_csv('clinical_data.csv', sep = ",", usecols= ['SDK-ID','CCE_polyp_no'])   



merged_df = pd.merge(df, FA_df, on=['SDK-ID','CCE_polyp_no'])


print('Between the clincial data csv and finished annotation images, there are ' + str(len(merged_df )) + ' overlaps of same polyp number for the same SDK-ID.') 


filename = Path_to_folder + '/Prepare data/clinical_FinishedAnnotation_merged.csv'
merged_df


merged_df.to_csv(filename)

