from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from keras import backend as K
import numpy as np
import os
from get_img_dict import get_img_dict
from multiprocessing import Pool
from functools import partial

'''File whose purpose is to preprocess images and produce an even larger dictionary for use as training and test data.'''

images_folder = "./data"
resized_images_folder = images_folder + "/resized"
raw_images_folder = images_folder + "/raw"
train_folder = raw_images_folder + "/train"
test_folder = raw_images_folder + "/validation"
transformed_images_folder = images_folder + "/transformed"

NUM_TRANSFORMED = 5
TARGET_SIZE = (224,224)

'''This is the number of transformed images for each in the original set'''

traingen = ImageDataGenerator(
  rotation_range = 360,
  width_shift_range = .3,
  height_shift_range = .3,
  rescale = 1/255,
  shear_range = .05,
  vertical_flip = True,
  zoom_range = .3,
  horizontal_flip = True,
  fill_mode = 'nearest',
  #featurewise_center = True,
  #featurewise_std_normalization = True
)
testgen = ImageDataGenerator(
  rescale = 1/255,
  #featurewise_std_normalization = True,
  #featurewise_center = True
)



def transform(arr, dr, datagen, threshold, save):
    key, tags = arr
    img = load_img(dr.format(key), target_size = TARGET_SIZE)
    x = img_to_array(img)
    x = x.reshape((1, *x.shape))
    
    #Generate random transformations of image
    i=0
    #datagen.fit(x)
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save,
        save_prefix=key.replace(".jpg", ""), save_format='jpg'):
        i += 1
        if i >= threshold: #gen threshold images for this shiz
            break  # otherwise the generator would loop indefinitely

    return 1


def transform_train_images(image_dict):
    ''' transform training set with its generator '''
    part = partial(transform, dr = "./data/raw/train/{}", datagen = traingen, threshold = NUM_TRANSFORMED, save="./data/transformed/train")
    pool = Pool()
    pool.map(part, image_dict.items())
    pool.close()
    pool.join()

def transform_val_images(image_dict):
    ''' transform test set with its generator '''
    part = partial(transform, dr = "./data/raw/validation/{}", datagen = testgen, threshold = 1, save ='./data/transformed/validation')
    pool = Pool()
    pool.map(part, image_dict.items())
    pool.close()
    pool.join()

def get_trans_dict(image_dict, directory):
    '''Get a dictionary representing the tags for a filename key on the transformed images'''
    trans_dict = {}
    translist = os.listdir(directory)
    for filename, tags in image_dict.items():
        for transname in translist:
            if(filename[:-4] in transname[:-4]): #ignoring the .jpg check to see transformation is child of each parent
                trans_dict[transname] = tags
                translist.remove(transname)
    return trans_dict


def get_training_set(img_dict):
    '''returns the dictionary representing the partitioned training set'''
    print("generating training set")
    transform_train_images(img_dict)
    return get_trans_dict(img_dict, transformed_images_folder + "/train")
def get_validation_set(img_dict):
    '''returns the dictionary representing the partitioned validation set'''
    print("generating validation set")
    transform_val_images(img_dict)
    return get_trans_dict(img_dict, transformed_images_folder + "/validation")

if(__name__ == "__main__"):
    
    os.system("rm ./data/transformed/train/* ./data/transformed/validation/*")
    tr_img_data = get_img_dict("./data/", "train_tags.csv")
    v_img_data = get_img_dict("./data/", "val_tags.csv")
    #print(get_trans_dict(img_data))
    print(get_training_set(tr_data))
    print(get_validation_set(v_data))
    
    # get_img_dict("./data/")
