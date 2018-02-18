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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

## Use NVIDIA self driving car network architecture 
## https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

## convolutional layers
## Note: 'Leaky RELU' could be another good candidate to experiment with.
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))  
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))

## dropout
## Note: Another interesting possible technique is spatial dropout:
## https://faroit.github.io/keras-docs/1.1.1/layers/core/#spatialdropout2d
model.add(Dropout(0.5))

## fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

## fit the model
model_path = "model.h5"

"""
Longer or shorter training might cause overfitting/underfitting of the model to the training set. 
Therefore, it is important to find the right balance Instead. This can be done by identifying 
the number of the epoch that provides the maximum performance (or convergence) on the validation set. 
It will be useful to implement a condition, that when met the training iteration stops. That can save 
processing time and will promise optimal network for the following analysis. 
One possible method is to use function ModelCheckpoint that will save the model after every epoch and 
in turn the best model from the training. Those models can be tried on track and a further comparison 
about the error and model performance on track can be made.
Ref: https://keras.io/callbacks/#modelcheckpoint
"""
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_observations), validation_data=validation_generator,
                    nb_val_samples=len(validation_observations), nb_epoch=5, verbose = 1)

## save model and its history
model.save('model.h5')

pickle_file = 'model_history.p'
with open(pickle_file, 'wb') as file_pi:
        pickle.dump(history_object.history, file_pi)

print('Model history is cached in pickle file.')
