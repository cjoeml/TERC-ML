import csv

def get_img_dict(directory):

	with open(directory + '/tags.csv', 'r') as tags:
		reader = csv.reader(tags, delimiter=',')
		img_dict = {}
		for row in reader:
			try:
				if(row[1] in ["Filename", "Tags"]):
					continue
				filename, tags = eval(row[1])
				tags = tags.split(",")
				img_dict[filename] = tags
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
if __name__ == "__main__":
	d = get_img_dict("../data")
	print(d)
