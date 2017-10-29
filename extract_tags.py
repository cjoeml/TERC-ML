import re as re
from PIL import Image, IptcImagePlugin
import os
import pandas as pd

# Constants, configure as appropriate
EARTH_PHOTOS_DIRECTORY = "./WinEarthPhotosByKeyword/"
INTERIOR_ISS_DIRECTORY = "./ISS/"

directories = [EARTH_PHOTOS_DIRECTORY,INTERIOR_ISS_DIRECTORY]

for d in directories:
	filenames_and_tags = []
	filenames_and_tags.append(["Filename","Tags"])

	for subdir, dirs, files in os.walk(d):
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

	pd.DataFrame(filenames_and_tags).to_csv(d + "tags.csv")