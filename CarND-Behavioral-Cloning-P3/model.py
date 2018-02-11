import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, Dropout, BatchNormalization

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	# skip first line
	first_line = next(reader)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	#print(source_path)
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)



model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

## convolutional layers
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))

## dropout
model.add(Dropout(0.5))

## fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

"""
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

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


