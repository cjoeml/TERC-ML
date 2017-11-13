from get_data import get_img_dict
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import numpy as np
import os

'''File whose purpose is to preprocess images and produce an even larger dictionary for use as training and test data.'''

NUM_TRANSFORMED = 5
'''This is the number of transformed images for each in the original set'''

datagen = ImageDataGenerator(
  rotation_range = 40,
  width_shift_range = .2,
  height_shift_range = .2,
  rescale = 1/255,
  shear_range = .2,
  zoom_range = .2,
  horizontal_flip = True,
  fill_mode = 'nearest'
)

def transform_images(image_dict):
    '''Produce NUM_TRANSFORMED images that are transformations of the resized data set'''
    for key, tags  in image_dict.items():
        img = load_img('./data/resized/{}'.format(key))
        x = img_to_array(img)
        x = x.reshape((1, *x.shape))

        #Generate random transformations of image
        i=0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='./data/transformed',
        save_prefix=key.replace(".jpg", ""), save_format='jpg'):
            i += 1
            if i > NUM_TRANSFORMED: #gen 50 images for this shiz
                break  # otherwise the generator would loop indefinitely
        img_width, img_height = 150,150
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

def get_trans_dict(image_dict):
    '''Get a dictionary representing the tags for a filename key on the transformed images'''
    trans_dict = {}
    translist = os.listdir("./data/transformed")
    for filename, tags in image_dict.items():
        for transname in translist:
            if(filename[:-4] in transname[:-4]): #ignoring the .jpg check to see transformation is child of each parent
                trans_dict[transname] = tags
                translist.remove(transname)
    return trans_dict

def shuffle_data():
    '''returns the indices for the 80% training and 20% validation sets'''
    np.random.seed(55)
    xy  = get_img_dict("./data")
    xyl = list(xy)
    K = int(.8 * len(xyl))
    np.random.shuffle(xyl)
    train = xyl[:K]
    test = xyl[K:]
    return train, test, xy

def get_training_set():
    '''returns the dictionary representing the partitioned training set'''
    train, test, imgdict = shuffle_data()
    train = {x : imgdict[x] for x in train}
    transform_images(train)
    transtrain = get_trans_dict(train)
    return transtrain

def get_validation_set():
    '''returns the dictionary representing the partitioned validation set'''
    train, test, imgdict = shuffle_data()
    return {x : imgdict[x] for x in test}

if(__name__ == "__main__"):
    
    os.system("rm ./data/transformed/*")
    img_data = get_img_dict("./data")
    #transform_images(img_data)
    #print(get_trans_dict(img_data))
    print(get_training_set())
    print(get_validation_set())
    
    # get_img_dict("./data/")
