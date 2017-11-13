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

# TODO(Tony): These are duplicated over in learning_module.py. Please fix
# Useful Constants
images_folder = "./data"
resized_images_folder = images_folder + "/resized"
raw_images_folder = images_folder + "/raw"
transformed_images_folder = images_folder + "/transformed"

resize_dimensions = (150,150)

valid_labels = ["Aurora","Cupola","Day","Dock UnDock","Moon",
            "Movie","Night","Stars","ISS Structure","Sunrise Sunset","Volcano","Inside ISS"]


def extract_tags(read_directory, write_directory):
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
                            tags += str(keywords, encoding="utf-8")
                    else:
                        for k in keywords:
                            tags += str(k, encoding="utf-8")
                            # don't put a comma on the end
                            if k != keywords[len(keywords)-1]:
                                tags += ","

                    filenames_and_tags.append([f,tags])

    pd.DataFrame(filenames_and_tags).to_csv(write_directory + "/tags.csv")


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


def get_img_dict(directory):
    """ Parses the tags.csv file and builds a dictionary mapping
        filenames to their tags

    Arguments:
    directory -- directory containing the 'tags.csv' file
    """
    with open(directory + 'tags.csv', 'r') as tags:
        reader = csv.reader(tags, delimiter=',')
        img_dict = {}

        for row in reader:
            if ( "0" in row):
                continue
            try:
                # print(row)
                img_dict[row[1]] = row[2].split(',')
            except(Exception):
                pass

        # We should probably double check which tags are relevant or not on Slack
        removeIrrelevantTags = ['1', 'Tags', 'Windows on Earth', '#WinEarthFavs', 'Astronaut Favs', 'Private', 'BUSpark', 'Christ Hadfield tweet',
                                'Scott Kelly tweet', 'Anousheh Ansari', 'Anousheh_All_Sun_Moon', 'Anousheh_Earth', 'Anousheh_Iran', 'Reid_Wiseman',
                                '#MovieSession - Reviewed', 'ReidWiseman movie tweet', 'WinEarthTweet', 'Maynard Show', 'BarnesAndNoble', 'ReidFavs', 'Leroy Chiao']

        # Removing unneeded tags that we can't measure
        for k in removeIrrelevantTags:
            img_dict.pop(k, None)

        return img_dict

               
def partition_image_data(directory,img_dimensions,img_depth):
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

        # different keras backends expect different orders
        if keras_backend.image_data_format() == "channels_first":
            input_shape = (img_depth,width,height)
            X = np.zeros((n,img_depth,width,height))
        else:
            input_shape = (width,height,img_depth)
            X = np.zeros((n,width,height,img_depth))

        # labels
        Y = np.zeros((n,len(valid_labels)))
        
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
    pass


if __name__ == "__main__":
    main()
