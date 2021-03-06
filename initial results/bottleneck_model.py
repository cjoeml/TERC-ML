# system imports
import os

# keras imports
from keras.callbacks import EarlyStopping
from keras import metrics
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import backend as K

# graphing imports
import matplotlib.pyplot as plt

# numpy imports
import numpy as np

# imports from our repo
import get_data as io

# Global Constants
IMAGES_FOLDER = "./data"
RAW_IMAGES_LOCATION = IMAGES_FOLDER + "/raw"
TRANSFORMED_IMAGES_LOCATION = IMAGES_FOLDER + "/transformed"
RESIZE_DIMENSIONS = (224, 224)


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
        os.system("""rm ./data/transformed/train/*
            ./data/transformed/validation/*""")
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

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn

def binary_accuracy_with_threshold(y_true, y_pred, threshold=.6):
    y_pred = K.cast(K.greater(y_pred, threshold), K.floatx())
    return K.mean(K.equal(y_true, y_pred))


def create_graphs(epoch_history, prefix):
    # plot history for accuracy
    plt.plot(epoch_history.history['binary_accuracy'])
    plt.plot(epoch_history.history['val_binary_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper left')
    plt.savefig(prefix + 'volcanoaccuracy.png')
    plt.plot(epoch_history.history['volcano_accuracy'])
    plt.plot(epoch_history.history['val_volcano_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper left')
    plt.savefig(prefix + 'accuracy.png')

    # summarize history for loss
    plt.plot(epoch_history.history['loss'])
    plt.plot(epoch_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper left')
    plt.savefig(prefix + 'loss.png')


def main():
    X_train, X_test, Y_train, Y_test, input_shape = image_preprocessing()

    nb_classes = len(Y_test[0])

    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
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

    # train our layers
    model.fit(X_train, Y_train, epochs=3, batch_size=64,
              validation_data=(X_test, Y_test), shuffle=True, verbose=2)

    # freeze first 4 layers from the Resnet Model
    for layer in model.layers[:142]:
        layer.trainable = False

    # Just train the last layer with our output layers
    for layer in model.layers[142:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    print("Recompiling model...")
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='binary_crossentropy', metrics=[
                  metrics.binary_accuracy, binary_accuracy_with_threshold])
    print("Done!\n")

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    epoch_history = model.fit(X_train, Y_train, epochs=10,
                              batch_size=64, validation_data=(
                                  X_test, Y_test), shuffle=True,
                              callbacks=[early_stopping])

    print("Saving weights...")
    model.save_weights('baseline_bottleneck_weights.h5')
    print("Done!\n")

    print("Creating graphs...")
    create_graphs(epoch_history, "")
    print("Done!\n")


if __name__ == "__main__":
    main()
