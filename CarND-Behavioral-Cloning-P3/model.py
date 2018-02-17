import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

import data_util as ut

## read in the training data and split into train and validation  

csv_file_path = './data/driving_log.csv'
train_observations, validation_observations = ut.get_train_validate_lines(csv_file_path)

## create data generators
train_generator = ut.generate_data(train_observations)
validation_generator = ut.generate_data(validation_observations)



model = Sequential()
# model.add(BatchNormalization())

model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(80,320,3)))
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

# what is the differece between relu and elu?
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

"""
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""



#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
## fit the model
model_path = "model.h5"
#checkpoint = ModelCheckpoint(model_path, verbose=0, save_best_only=True)
#callbacks_list = [checkpoint]

#model.fit_generator(train_generator, samples_per_epoch=len(train_observations), validation_data=validation_generator,
#                    nb_val_samples=len(validation_observations), nb_epoch=5, callbacks=callbacks_list)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_observations), validation_data=validation_generator,
                    nb_val_samples=len(validation_observations), nb_epoch=5, verbose = 1)

model.save('model.h5')

#To run: python drive.py model.h5

"""
model.add(Convolution2D(6,6,6,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,6,6,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
"""


