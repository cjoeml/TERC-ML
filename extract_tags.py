import re as re
from PIL import Image, IptcImagePlugin
import os
import pandas as pd

def extract_tags(directories):
	for d in directories:
		filenames_and_tags = ["Filename","Tags"]
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

if __name__ == "__main__":
    extract_tags(["../data/"])
