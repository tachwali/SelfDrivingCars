import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
import pickle
import data_util as ut

## read in the training data and split into train and validation  

csv_file_path = './data/driving_log.csv'
train_observations, validation_observations = ut.get_train_validate_lines(csv_file_path)

## create data generators
train_generator = ut.generate_data(train_observations)
validation_generator = ut.generate_data(validation_observations)

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(80,320,3)))

## Use NVIDIA self driving car network architecture 
## https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

## convolutional layers
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))

## dropout
model.add(Dropout(0.5))

## fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

## fit the model
model_path = "model.h5"

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_observations), validation_data=validation_generator,
                    nb_val_samples=len(validation_observations), nb_epoch=5, verbose = 1)

## save model and its history
model.save('model.h5')

pickle_file = 'model_history.p'
with open(pickle_file, 'wb') as file_pi:
        pickle.dump(history_object.history, file_pi)

print('Model history is cached in pickle file.')
