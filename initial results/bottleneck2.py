import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras import metrics

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras import backend as K

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

def main():
	X_train,X_test,Y_train,Y_test,input_shape = image_preprocessing()
	#def top_k_binary_accuracy(y_true, y_pred, k=3): return K.mean(K.in_top_k(y_pred)  K.in_top_k(y_true, axis=-1), k), axis=-1)
	def inTop3(x,y): return metrics.top_k_categorical_accuracy(x,y,k=3) #metric function

	nb_classes = len(Y_test[0])

	base_model = ResNet50(weights='imagenet', include_top=False)
	
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(nb_classes, activation='sigmoid')(x)
	print("obtaining base model")
	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
		layer.trainable = False
	print("compiling model")
	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, inTop3]) # ty adam
	
	# train the model on the new data for a few epochs
	model.fit(X_train,Y_train,epochs=3,batch_size=64, validation_data=(X_test,Y_test),shuffle=True,verbose=2)
			
	# at this point, the top layers are well trained and we can start fine-tuning
	# convolutional layers from inception V3. We will freeze the bottom N layers
	# and train the remaining top layers.

	# let's visualize layer names and layer indices to see how many layers
	# we should freeze:
	#for i, layer in enumerate(base_model.layers):
	#   print(i, layer.name)
	
	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 249 layers and unfreeze the rest:
	for layer in model.layers[:142]:
	   layer.trainable = False
	for layer in model.layers[142:]:
	   layer.trainable = True
	print("recompiling model to train top 2 inception blocks")
	# we need to recompile the model for these modifications to take effect
	# we use SGD with a low learning rate




	from keras.optimizers import SGD
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=[metrics.binary_accuracy, metrics.categorical_accuracy, inTop3])

	early_stopping = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(X_train,Y_train, epochs=10,batch_size=64, validation_data=(X_test,Y_test),shuffle=True, callbacks=[early_stopping])
	#score = model.evaluate(X_test, Y_test, verbose=1)
	#print(score)
	#out = model.predict(X_test)
	#print(out, Y_test)
	print("saving weights")
	model.save_weights('bottleneck_weights.h5')
	from keras.utils import plot_model
	plot_model(model, to_file='model.png')
	
	'''
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid')) # Should probably use this because we're doing multiclass stuff

	# let's train the model using SGD + momentum
	sgd = SGD(lr=1, decay=1e-9, momentum=0.7, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) # ty adam
	early_stopping = EarlyStopping(monitor='val_loss', patience=3)
	model.fit(X_train,Y_train, epochs=10,batch_size=64, validation_data=(X_test,Y_test),shuffle=True, callbacks=[early_stopping])
	score = model.evaluate(X_test, Y_test, verbose=1)
	out = model.predict(X_test)
	#model.save_weights('first_try.h5')  # always save your weights after training or during training
	#TODO: Save weights
	'''

if __name__ == "__main__":
	main()
