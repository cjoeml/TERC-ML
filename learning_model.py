import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split

# imports from our repo
import get_data as io

# TODO(Tony): These are duplicated over in get_data.py. Please fix
# Useful Constants
images_folder = "./data"
resized_images_folder = images_folder + "/resized"
raw_images_folder = images_folder + "/raw"
transformed_images_folder = images_folder + "/transformed"

resize_dimensions = (150,150)

def image_preprocessing():
	""" Preprocesses images for eventually submission to the learning model"""

	# generate tags for images, if they do not exist
	if not os.path.isfile(images_folder + "/tags.csv"):
		print("Extracting tags...")
		io.extract_tags(raw_images_folder,images_folder)
		print("Done!\n")

	# resize the images to a particular size, if not already done
	# Let's assume its already done if a particular directory 
	# has images in it...
	if len(os.listdir(resized_images_folder)) ==  0:
		print("Resizing images...this could take awhile!\n")
		io.resize_images(raw_images_folder,resized_images_folder,resize_dimensions)


	# generate X and Y, in the format keras needs for the model
	# and return
	return io.partition_image_data(resized_images_folder,(150,150),3)

def main():
	X_train,X_test,Y_train,Y_test,input_shape = image_preprocessing()

	nb_classes = len(Y_test[0])
	# TODO(Tony): Bring over code from trial3.py
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=input_shape))
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
	model.fit(X_train,Y_train,nb_epoch=10,batch_size=32,validation_data=(X_test,Y_test),shuffle=True, callbacks=[early_stopping])
	score = model.evaluate(X_test, Y_test, verbose=0)
	out = model.predict(X_test)



if __name__ == "__main__":
    main()