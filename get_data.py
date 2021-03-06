# TODO(Tony): Clean this up
import os
import numpy as np
import skimage.io as skimi
import skimage.transform as skimt
from sklearn.cross_validation import train_test_split
from keras import backend as keras_backend
import re as re
from PIL import Image, IptcImagePlugin
import os
import pandas as pd
import csv
from datagenerator import get_training_set, get_validation_set
from get_img_dict import get_img_dict
# TODO(Tony): These are duplicated over in learning_module.py. Please fix
# Useful Constants
images_folder = "./data"
resized_images_folder = images_folder + "/resized"
#raw_images_folder = images_folder + "/raw"
raw_images_folder = "../../terc-sail/multi_label_images"
transformed_images_folder = images_folder + "/transformed"

resize_dimensions = (150,150)

# valid_labels = ["Aurora","Cupola","Day","Dock UnDock","Moon",
            # "Movie","Night","Stars","ISS Structure","Sunrise Sunset","Volcano","Inside ISS"]
valid_labels = ["Day", "ISS Structure", "Moon", "Sunrise Sunset", "Harvey", "Dock Undock", "Deployer", "Cupola",
            "Night", "Clouds", "Solar Panels", "Hurricane", "Aurora", "Solar Eclipse", "Inside ISS", "Sun", "Volcano",
            "Deployed satellite", "San Francisco"]

def get_valid_labels():
    return valid_labels

def extract_tags(read_directory, write_directory, prefix=""):
    """ Extracts image tags embedded in said images
    Arguments:
    read_directory -- directory to read the images from
    write_directory -- directory to write the tags.csv file to
    """
    filenames_and_tags = [["Filename","Tags"]]
    for subdir, dirs, files in os.walk(read_directory):
        for f in files:
                if re.search(".jpg",f):
                    img = Image.open(subdir + "/" + f)
                    keywords = IptcImagePlugin.getiptcinfo(img)[(2, 25)]

                    tags = ""
                    # if the image has just a single tag
                    if type(keywords) == type(bytes()):
                        # tags += str(keywords, encoding="utf-8")
                        tag = str(keywords, encoding="utf-8")
                        if tag == "Harvey":
                            tags += "Hurricane"
                        else:
                            tags += tag
                    else:
						# if the image has many tags
                        for k in keywords:
                            # tags += str(k, encoding="utf-8")
                            tag = str(k, encoding="utf-8")
                            if tag == "Harvey":
                                tags += "Hurricane"
                            else:
                                tags += tag
                            # don't put a comma on the end
                            if k != keywords[len(keywords)-1]:
                                tags += ","
                    filenames_and_tags.append([f,tags])

    pd.DataFrame(filenames_and_tags).to_csv(write_directory + "/" + prefix + "tags.csv")



def resize_images(read_directory,write_directory,dimensions):
    """ Resizes images to the dimensions and saves them to disk

    Arguments:
    read_directory -- directory to read the images from
    write_directory -- directory to write the resized images to
    dimensions -- a tuple with the dimensions to resize to i.e (100,100)
    """
    if os.path.isdir(read_directory):
        i = 0   
        for subdir, _, files in os.walk(read_directory):
            for f in files:
                original_img = skimi.imread(subdir + "/" + f)
                # NOTE(Tony): the API suggested using anti_aliasing for downsizing, 
                # but the keyword parameter wasn't working...
                resized_img = skimt.resize(original_img,dimensions)
                skimi.imsave(write_directory + "/" + f,resized_img)
                
                print(i)
                i += 1



def format_data(img_dimensions, img_depth):
    '''Generates the training set and test set inputs for the Keras Model
       Uses the dictionaries generated for each set to create one hot vectors for predictions
    '''
    width = img_dimensions[0]
    height = img_dimensions[1]
    
    vim_dict = get_img_dict(images_folder + "/", "val_tags.csv")
    trim_dict = get_img_dict(images_folder + "/", "train_tags.csv")
    
    tr = get_training_set(trim_dict)
    val = get_validation_set(vim_dict)
    
    n_train = len(tr)
    n_val = len(val)
    
    # different keras backends expect different orders
    if keras_backend.image_data_format() == "channels_first":
        input_shape = (img_depth,width,height)
        X_train = np.zeros((n_train,img_depth,width,height))
        X_val = np.zeros((n_val,img_depth,width,height))
    else:
        input_shape = (width,height,img_depth)
        X_train = np.zeros((n_train, width,height,img_depth))
        X_val = np.zeros((n_val, width, height, img_depth))
    
    # labels
    Y_train = np.zeros((n_train, len(valid_labels)))
    Y_val = np.zeros((n_val, len(valid_labels)))
    
    sample_index = 0
    for filename, tags in tr.items():
        X_train[sample_index] = skimi.imread(transformed_images_folder + "/train/" + filename)
        for t in tags:
           label_index = 0
           for valid_tag in valid_labels:
               # if the image has what we consider valid tags
               # mark the hot vector appropriately
               if t.lower() == valid_tag.lower():
                   Y_train[sample_index][label_index] = 1
               label_index += 1
        sample_index += 1

    sample_index = 0
    for filename, tags in val.items():
        X_val[sample_index] = skimi.imread(transformed_images_folder + "/validation/" + filename)
        for t in tags:
           label_index = 0
           for valid_tag in valid_labels:
               # if the image has what we consider valid tags
               # mark the hot vector appropriately
               if t.lower() == valid_tag.lower():
                   Y_val[sample_index][label_index] = 1
               label_index += 1
        sample_index += 1
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    #X_train /= 255
    #X_val /= 255
    return (X_train,X_val,Y_train,Y_val,input_shape) 

def partition_image_data(directory,img_dimensions,img_depth, opt=""):
    """ Takes images and puts them in the format keras
        needs for its cnn model
    Arguments:
    directory -- directory containing images
    img_dimensions - dimensions of the images
    img_depth - color depth i.e 3 for RGB
    """



    if os.path.isdir(directory):

        width = img_dimensions[0]
        height = img_dimensions[1]
        
        n = len(os.listdir(directory))
        img_to_tags = get_img_dict(images_folder+"/")
        if(opt == "trans"):
            img_to_tags = get_trans_dict(img_to_tags)
        # different keras backends expect different orders
        if keras_backend.image_data_format() == "channels_first":
            input_shape = (img_depth,width,height)
            X = np.zeros((n,img_depth,width,height))
        else:
            input_shape = (width,height,img_depth)
            X = np.zeros((n,width,height,img_depth))
        # labels
        Y = np.zeros((n, len(valid_labels)))
        
        for subdir, _, files in os.walk(directory):
            sample_index = 0
            for f in files:
                if re.search(".jpg",f):
                    # read the image and store it as the nth sample
                    X[sample_index] = skimi.imread(subdir + "/" + f)
                    
                    tags = img_to_tags[f]
                    for t in tags:
                        label_index = 0
                        for valid_tag in valid_labels:
                            # if the image has what we consider valid tags
                            # mark the hot vector appropriately
                            if t.lower() == valid_tag.lower():
                                Y[sample_index][label_index] = 1
                            label_index += 1

                    sample_index += 1
                    
            # TODO(Tony): Replace with Matt's stuff later
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.2)

        # Converg to float32 and normalize (per multiclass tutorial)
        # TODO(Tony): Remind yourself why we normalize
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        return (X_train,X_test,Y_train,Y_test,input_shape)

def main():
    os.system("rm ./data/*.csv")
    extract_tags("./data/raw/validation","./data/raw","val_")
    extract_tags("./data/raw/train","./data/raw","train_")


if __name__ == "__main__":
    main()
