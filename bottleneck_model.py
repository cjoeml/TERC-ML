# system imports
import os
# graphing imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
# keras imports
from keras.callbacks import EarlyStopping
from keras import metrics
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import backend as K
# numpy imports
import numpy as np
# imports from our repo
import get_data as io
# ignore deprecation warnings
import warnings
warnings.filterwarnings("ignore")


# Global Constants
IMAGES_FOLDER = "./data"
RAW_IMAGES_LOCATION = IMAGES_FOLDER + "/raw"
TRANSFORMED_IMAGES_LOCATION = IMAGES_FOLDER + "/transformed"
RESIZE_DIMENSIONS = (224, 224)
RESULT_PREFIX = "bestmodel_5transform_"

# Tweak these
OUTPUT_LAYER_EPOCHS = 10
COMPLETE_MODEL_EPOCHS = 10


def image_preprocessing():
    """ Preprocesses images for eventually submission to the learning model"""
    # generate tags for images, if they do not exis t
    if not os.path.isfile(IMAGES_FOLDER + "/train_tags.csv"):
        print("Extracting training tags...")
        io.extract_tags(RAW_IMAGES_LOCATION + "/train",
                        IMAGES_FOLDER, "train_")
        print("Done!\n")
    else:
        print("Training tags exist...moving along...\n")

    if not os.path.isfile(IMAGES_FOLDER + "/val_tags.csv"):
        print("Extracting validation tags...")
        io.extract_tags(RAW_IMAGES_LOCATION + "/validation",
                        IMAGES_FOLDER, "val_")
        print("Done!\n")
    else:
        print("Validation tags exist...moving along...\n")

    if(len(os.listdir(TRANSFORMED_IMAGES_LOCATION + "/train")) +
            len(os.listdir(TRANSFORMED_IMAGES_LOCATION + "/validation")) > 0):
        print("Clearing out previously transformed images...")
        os.system("rm ./data/transformed/train/* ./data/transformed/validation/*")
        print("Done!\n")

    # generate X and Y, in the format keras needs for the model
    # and return
    # return io.partition_image_data(resized_IMAGES_FOLDER,(150,150),3)
    print("Obtaining model inputs...")
    X_train, X_test, Y_train, Y_test, input_shape = io.format_data(
        RESIZE_DIMENSIONS, 3)
    print("Done!\n")

    return (X_train, X_test, Y_train, Y_test, input_shape)


def individual_accuracy(y_true, y_pred):
    '''Returns array containing the accuracy for each tag'''
    return np.equal(K.get_value(y_true), K.get_value(K.round(y_pred)), axis=1)


def binary_accuracy_with_threshold(y_true, y_pred, threshold=.6):
    y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
    return K.mean(K.equal(y_true, y_pred))


def create_graphs(epoch_history, prefix, model):

    plot_model(model, to_file=prefix + 'model.png')

    # plot history for accuracy
    plt.plot(epoch_history.history['binary_accuracy'])
    plt.plot(epoch_history.history['val_binary_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='bottom right')
    plt.savefig(prefix + 'accuracy.png')
    
    plt.clf()

    # summarize history for loss
    plt.plot(epoch_history.history['loss'])
    plt.plot(epoch_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='top right')
    plt.savefig(prefix + 'loss.png')


def main():
    print("Beginning Model Training...\n")
    X_train, X_test, Y_train, Y_test, input_shape = image_preprocessing()

    nb_classes = len(Y_test[0])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    base_model = ResNet50(weights='imagenet', include_top=False)
    
    #for i, layer in enumerate(base_model.layers):
    #    print(i, layer.name)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='sigmoid')(x)

    print("Obtaining base model...")
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print("Done!\n")

    # freeze all resnet layers
    for layer in base_model.layers:
        layer.trainable = False

    print("Compiling model...")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
                  metrics.binary_accuracy, binary_accuracy_with_threshold])
    print("Done!\n")

    print("Training output model layers...")
    # train our layers
    model.fit(X_train, Y_train, epochs=OUTPUT_LAYER_EPOCHS, batch_size=64,
              validation_data=(X_test, Y_test), shuffle=True, verbose=2)
    print("Done!\n")
    # freeze first 4 layers from the Resnet Model (142 -> stage 5)
    for layer in model.layers[:80]:
        layer.trainable = False

    # Just train the last layer with our output layers
    for layer in model.layers[80:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    print("Recompiling model...")
    # SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=[
                  metrics.binary_accuracy, binary_accuracy_with_threshold])
    print("Done!\n")

    print("Training all model layers...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    epoch_history = model.fit(X_train, Y_train, epochs=COMPLETE_MODEL_EPOCHS,
                            batch_size=64, validation_data=(
                                  X_test, Y_test), shuffle=True,
                              callbacks=[early_stopping])
    print("Done!\n")
    print("Saving weights...")
    model.save_weights(RESULT_PREFIX + 'bottleneck_weights.h5')
    print("Done!\n")

    print("Creating graphs...")
    create_graphs(epoch_history, RESULT_PREFIX, model)
    print("Done!\n")
    

if __name__ == "__main__":
    main()
