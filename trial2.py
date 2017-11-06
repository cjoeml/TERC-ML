from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split
from get_data import get_data # Should probably make this in our repo

import numpy as np

X,Y = get_data()
# X is the set of our inputs (images) with a size of (tentatively) 100 x 100 --> (n, width, height)
## ^ We would probably load images into an array
# Y is our class label data

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size =0.2)
 # We don't need to do this if our test and training data is separate

img_channels = 3	 # RGB
img_rows     = 100
img_cols     = 100
nb_classes   = 5     # Should change this to how many classes we have

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax')) # Should probably use this because we're doing multiclass stuff


# let's train the model using SGD + momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # ty adam
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train,y_train,nb_epoch=30,batch_size=32,show_accuracy=True,validation_data=(x_test,y_test),shuffle=True, callbacks=[early_stopping])
score = model.evaluate(x_test, y_test, verbose=0)
out = model.predict(x_test)
