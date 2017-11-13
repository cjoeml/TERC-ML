import os

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

	# TODO(Tony): Bring over code from trial3.py

if __name__ == "__main__":
    main()