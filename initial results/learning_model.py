import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split
from keras import metrics
# imports from our repo
import get_data as io
import datagenerator as gen
# TODO(Tony): These are duplicated over in get_data.py. Please fix
# Useful Constants
images_folder = "./data"
resized_images_folder = images_folder + "/resized"
raw_images_folder = images_folder + "/raw"
transformed_images_folder = images_folder + "/transformed"
resize_dimensions = (224,224)

def image_preprocessing():
	""" Preprocesses images for eventually submission to the learning model"""
	# generate tags for images, if they do not exis	t
	if not os.path.isfile(images_folder + "/train_tags.csv"):
		print("Extracting tags...")
		io.extract_tags(raw_images_folder + "/train",images_folder,"train_")
		print("Done!\n")
	if not os.path.isfile(images_folder + "/val_tags.csv"):
		print("Extracting tags...")
		io.extract_tags(raw_images_folder + "/validation", images_folder,"val_")
		print("Done!\n")

	# resize the images to a particular size, if not already done
	# Let's assume its already done if a particular directory 
	# has images in it...
	'''if len(os.listdir(resized_images_folder)) ==  0:
		print("Resizing images...this could take awhile!\n")
		io.resize_images(raw_images_folder,resized_images_folder,resize_dimensions)
	'''
	if(len(os.listdir(transformed_images_folder + "/train")) + len(os.listdir(transformed_images_folder + "/validation")) > 0):	
		print("clearing out previously transformed images")	
		os.system("rm ./data/transformed/train/* ./data/transformed/validation/*")
		
	# generate X and Y, in the format keras needs for the model
	# and return
	#return io.partition_image_data(resized_images_folder,(150,150),3)
	print("obtaining model inputs")
	return io.format_data(resize_dimensions, 3)

def imgnet_model():
	pass

def main():
	X_train,X_test,Y_train,Y_test,input_shape = image_preprocessing()

	nb_classes = len(Y_test[0])
	# TODO(Tony): Bring over code from trial3.py
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding="same",input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding = "same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), padding="same"))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3), padding = "same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid')) # Should probably use this because we're doing multiclass stuff

	# let's train the model using SGD + momentum
	sgd = SGD(lr=.001, decay=1e-9, momentum=0.7, nesterov=True)
	inTop3 = gen.partial(metrics.top_k_categorical_accuracy , k=3)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, inTop3]) # ty adam
	early_stopping = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(X_train,Y_train, epochs=3,batch_size=64, validation_data=(X_test,Y_test),shuffle=True, callbacks=[early_stopping])
	score = model.evaluate(X_test, Y_test, verbose=1)
	out = model.predict(X_test)
	print(out, Y_test)
	#model.save_weights('first_try.h5')  # always save your weights after training or during training
	#TODO: Save weights


if __name__ == "__main__":
	main()
