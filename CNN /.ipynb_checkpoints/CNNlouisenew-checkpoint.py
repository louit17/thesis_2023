# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

#only use GPU 0 
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Force CPU use
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#import tensorrt
import tensorflow as tf

#If memory growth is enabled for a PhysicalDevice, the runtime initialization will not allocate all memory on the device. 
#https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
#gpu = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)


import sys
from pathlib import Path
import os
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2

import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
import sklearn.metrics
from sklearn.metrics import confusion_matrix

#import tensorrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu


HOME = Path('/home/peregrin/Desktop')

data_base_dir = HOME / 'CNN-main/Input'
Path_to_folder = HOME / 'CNN-main'


IMG_CHANNELS = 3
IMG_WIDTH = 526
IMG_HEIGHT = 526
epochs = 100                    
batchSIZE = 3
Patience = 5                   
learning_rate = 1e-5
augment_training_data = True    # Performs flipping and rotations by 90°

data_generator_args = dict(rotation_range=0.5,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')

def data_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(IMG_CHANNELS, IMG_WIDTH),
                    seed=1,
                    augment:bool=False):

    image_datagen = ImageDataGenerator(**aug_dict)


    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="filename",
        y_col = "label",
        class_mode="sparse",
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)




    for (img, label) in image_generator:
        #print('Got label', label, ' -- ', img[0, 125, 125,0])
        for _i, _m in adjust_data(img, label, augment):
            yield (_i, _m)


def rescale_data(img):
    """ Rescale images to have float channels \in [0, 1], boolean masks.

    """
    img = img / 255


    return (img)


def augment_data(img, label):
    """ Data augmentation by rotation and flipping.
    """
    images = [img]

    flipped_img = tf.image.flip_up_down(img)
    images.append(flipped_img)

    for i in range(1, 4):
        images.append(tf.image.rot90(img, i))
        images.append(tf.image.rot90(flipped_img, i))

    return zip(images, 8 * [label])


def adjust_data(img, label, augment:bool):
    """ Perform image value scaling and augmentation """
    img = rescale_data(img)

    if not augment:
        return zip([img], [label])

    return augment_data(img, label)

def assign_class(size):
    if size == '0-5 mm':
        return 0
    elif size == '6-9 mm':
        return 1
    elif size == '10-19 mm':
        return 2
    else:
        return 3
    
#Image folder
folder_images = 'Raw_images'

#image
print('Collecting data from {}:'.format(str(data_base_dir)))
train_files = list((data_base_dir / folder_images).glob('*.png'))

#MASK not used yet
mask_files = list()
for _f in train_files:
    _mf = data_base_dir / str(_f).replace("raw", "segmentation").replace(folder_images, "segmentation_images")
    
    mask_files.append(_mf)
    

print('  … collected {} samples.'.format(len(train_files)))


#Label
os.chdir(Path_to_folder)
df_labels = pd.read_csv('Finished_annotations_depth_information.csv')
label_files = list()

for _f in train_files: 
    filename = os.path.basename(_f)
    parts = filename.split("_")
    patient_id = int(parts[1])
    P_x = parts[2]
    polyp_num = int(P_x[1:])
    
    filter_condition = (df_labels['Patient ID'] == patient_id) & (df_labels['Polyp'] == polyp_num)
    #label_OC = df_labels.loc[filter_condition, 'OC Morphology'].values[0]
    label_value = df_labels.loc[filter_condition, 'class'].values[0]
    label_value = assign_class(label_value)
    
    label_files.append(label_value)
    
print(len(label_files), len(train_files), len(mask_files))


# Splitting the data set
df = pd.DataFrame(data={"filename": [str(_f) for _f in train_files], 'mask': [str(_f) for _f in mask_files], 'label': [str(_f) for _f in label_files]})
df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)

print('Data set statistics:')
print('\tFull data set:  {:8d} samples'.format(len(df)))
print('\tTraining set:   {:8d} samples, {:2.0f} %, to be augmented x8: {}'.format(len(df_train), np.floor(len(df_train)/len(df)*100), augment_training_data))
print('\tValidation set: {:8d} samples, {:2.0f} %'.format(len(df_val), np.floor(len(df_val)/len(df)*100)))
print('\tTest set:       {:8d} samples, {:2.0f} %'.format(len(df_test), np.floor(len(df_test)/len(df)*100)))


#Generate training and validation data from dfs
train_gen = data_generator(df_train, batchSIZE, data_generator_args,
                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                            augment=augment_training_data)

#x_img, x_mask= next(train_gen)

valid_gen = data_generator(df_val, batchSIZE,
                            dict(),
                            target_size=(IMG_HEIGHT, IMG_WIDTH))
test_gen = data_generator(df_test, batchSIZE,
                            dict(),
                            target_size=(IMG_HEIGHT, IMG_WIDTH))

Path_to_save= HOME / 'CNN_model_weights'
os.chdir(Path_to_save)

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 30, min_delta = 0.001, restore_best_weights = True)

#Set activation function and optimizers
act = 'relu'
opt = 'adam'

#%%dropout model 1
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=act, 
                           input_shape=(526, 526, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=act),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=act),
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(64, activation=act),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(4, activation='softmax')
    ])


model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )


model.summary()

hist= model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs, callbacks=[early_stop],
                        validation_data=valid_gen,
                        validation_steps=len(df_val) / batchSIZE, verbose=1)

model.save('model_1_dropout')


fig = plt.figure(figsize = plt.figaspect(0.3))
fig.suptitle("Loss and accuracy for model")
        
ax = fig.add_subplot(1,2,1)
ax.plot(hist.history['loss'], label='Train loss opt: '+opt+ ' and act: '+act)
ax.plot(hist.history['val_loss'], label = 'Validation loss opt: '+opt+ ' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


ax = fig.add_subplot(1,2,2)
ax.plot(hist.history['accuracy'], label='Train accuracy opt: '+opt+' and act: '+act)
ax.plot(hist.history['val_accuracy'], label = 'Validation accuracy opt: '+opt+' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.show()


#newmodel1 = tf.keras.models.load_model('/home/peregrin/Desktop/CNN_model_weights/model_1_dropout')
#newmodel1.evaluate(test_gen, steps=len(df_test)/batchSIZE, verbose=1)

model.evaluate(test_gen, steps=len(df_test)/batchSIZE, verbose=1)

#%% model 2

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=act,
                               input_shape=(526, 526, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=act),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=act),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), activation = act),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=act),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=act),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation=act),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )

model.summary()

hist= model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs, callbacks=[early_stop],
                        validation_data=valid_gen,
                        validation_steps=len(df_val) / batchSIZE, verbose=1)

model.save('model_2')


fig = plt.figure(figsize = plt.figaspect(0.3))
fig.suptitle("Loss and accuracy for model 2")
        
ax = fig.add_subplot(1,2,1)
ax.plot(hist.history['loss'], label='Train loss opt: '+opt+ ' and act: '+act)
ax.plot(hist.history['val_loss'], label = 'Validation loss opt: '+opt+ ' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


ax = fig.add_subplot(1,2,2)
ax.plot(hist.history['accuracy'], label='Train accuracy opt: '+opt+' and act: '+act)
ax.plot(hist.history['val_accuracy'], label = 'Validation accuracy opt: '+opt+' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.evaluate(test_gen, steps=len(df_test)/batchSIZE, verbose=1)

#%% Model 3

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=act, 
                           input_shape=(526, 526, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=act),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=act),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=act),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=act),
    tf.keras.layers.Flatten(), # flatten before fully connected part
    tf.keras.layers.Dense(256, activation=act),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation=act),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation=act),
    tf.keras.layers.Dense(4, activation='softmax')
    ])


model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )

model.summary()

hist= model.fit(train_gen,
                        steps_per_epoch=len(df_train) / batchSIZE,
                        epochs=epochs, callbacks=[early_stop],
                        validation_data=valid_gen,
                        validation_steps=len(df_val) / batchSIZE, verbose=1)

model.save('model_3')


fig = plt.figure(figsize = plt.figaspect(0.3))
fig.suptitle("Loss and accuracy for model 3")
        
ax = fig.add_subplot(1,2,1)
ax.plot(hist.history['loss'], label='Train loss opt: '+opt+ ' and act: '+act)
ax.plot(hist.history['val_loss'], label = 'Validation loss opt: '+opt+ ' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


ax = fig.add_subplot(1,2,2)
ax.plot(hist.history['accuracy'], label='Train accuracy opt: '+opt+' and act: '+act)
ax.plot(hist.history['val_accuracy'], label = 'Validation accuracy opt: '+opt+' and act: '+act)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.show()

model.evaluate(test_gen, steps=len(df_test)/batchSIZE, verbose=1)