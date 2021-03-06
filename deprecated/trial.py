import numpy as np
import csv
from get_img_dict import get_img_dict
# from skimage import color, exposure, transform

# Constants, configure as appropriate
EARTH_PHOTOS_DIRECTORY = "./WinEarthPhotosByKeyword/"
INTERIOR_ISS_DIRECTORY = "./ISS/"
directories = [EARTH_PHOTOS_DIRECTORY,INTERIOR_ISS_DIRECTORY]

# Opening our csv file with our tags generated by extract_tags.py
with open(EARTH_PHOTOS_DIRECTORY + 'tags.csv', 'r') as tags:
	reader = csv.reader(tags, delimiter=',')
	tag_dict = {}
	for row in reader:
		for image in row[2:]:
			tags = image.split(',')
			for tag in tags:
				try:
					tag_dict[tag] = tag_dict[tag] + 1
				except KeyError:
					tag_dict[tag] = 1

	# We should probably double check which tags are relevant or not on Slack
	removeIrrelevantTags = ['1', 'Tags', 'Windows on Earth', '#WinEarthFavs', 'Astronaut Favs', 'Private', 'BUSpark', 'Christ Hadfield tweet', 
	'Scott Kelly tweet', 'Anousheh Ansari', 'Anousheh_All_Sun_Moon', 'Anousheh_Earth', 'Anousheh_Iran', 'Reid_Wiseman', 
	'#MovieSession - Reviewed', 'ReidWiseman movie tweet', 'WinEarthTweet', 'Maynard Show', 'BarnesAndNoble', 'ReidFavs', 'Leroy Chiao']

	# Removing unneeded tags that we can't measure
	for k in removeIrrelevantTags:
	    tag_dict.pop(k, None)

	# Sort and print by descending order of frequency
	sorted_tags = sorted(tag_dict, key=tag_dict.__getitem__, reverse=True)
	ttl = 0
	for k in sorted_tags:
	    # print("{} : {}".format(k, tag_dict[k]))
	    ttl += tag_dict[k]
	# print("======================\nTotal tags: " + str(ttl))
	TOTAL_TAGS = len(sorted_tags)
	# print("Total unique tags: " + str(TOTAL_TAGS))

	# A tuple in case we need it
	data_tup = (ttl, TOTAL_TAGS, tag_dict, sorted_tags)

	img_dict = get_img_dict(EARTH_PHOTOS_DIRECTORY)
	for key in img_dict.keys():
		print("[" + key + "]" + ": " + str(img_dict[key]))