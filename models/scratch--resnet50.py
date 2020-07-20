import numpy as np
import pandas as pd
import glob
from pathlib import Path
import os
import shutil 
from time import time
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(42)
datasets_dir = Path('dogs-vs-cats')

print('Loaded!')

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

X_train = np.load('X_train_MANY-dog_vs_cat_features.npy')
X_test = np.load('X_test_MANY-dog_vs_cat_features.npy')
X_val = np.load('X_val_MANY-dog_vs_cat_features.npy')
y_test = np.load('y_test_MANY-dog_vs_cat_truth.npy')
y_val = np.load('y_val_MANY-dog_vs_cat_truth.npy')
y_train = np.load('y_train_MANY-dog_vs_cat_truth.npy')

# hyperparameters
img_width = img_height = 224
img_dim = (img_width, img_height)
n_epochs = 100
learning_rate = 1e-5

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam

n_epochs = 500
for learning_rate in [1e-2]:
# for learning_rate in [1e-4]:
    model = Sequential()
    model.add(Dense(2, activation='softmax', input_shape=(2048,)))

    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(learning_rate=learning_rate),
                 metrics=['accuracy'])

    start = time()
    run = model.fit(x=X_train,
                    y=y_train,
                    epochs=n_epochs,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    workers=-1)
    print(f'\n\n learning rate: {learning_rate}, max: {np.max(run.history["val_accuracy"])}\n{(time() - start) / 60:.2f} min')
    
    fig, ax = plt.subplots()
    ax.plot(range(len(run.history['val_accuracy'])), run.history['val_accuracy'])
    ax.set(title=f'lr: {learning_rate}, Val accuracy over epochs', ylabel='Val acc', xlabel='Epochs')
    plt.show()