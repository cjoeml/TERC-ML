import csv

def get_img_dict(directory):

	with open(directory + 'tags.csv', 'r') as tags:
		reader = csv.reader(tags, delimiter=',')
		img_dict = {}
		for row in reader:
			img_dict[row[1]] = row[2].split(',')
			print(row[1] + ": " + str(img_dict[row[1]]))

		# We should probably double check which tags are relevant or not on Slack
		removeIrrelevantTags = ['1', 'Tags', 'Windows on Earth', '#WinEarthFavs', 'Astronaut Favs', 'Private', 'BUSpark', 'Christ Hadfield tweet', 
		'Scott Kelly tweet', 'Anousheh Ansari', 'Anousheh_All_Sun_Moon', 'Anousheh_Earth', 'Anousheh_Iran', 'Reid_Wiseman', 
		'#MovieSession - Reviewed', 'ReidWiseman movie tweet', 'WinEarthTweet', 'Maynard Show', 'BarnesAndNoble', 'ReidFavs', 'Leroy Chiao']

		# Removing unneeded tags that we can't measure
		for k in removeIrrelevantTags:
		    img_dict.pop(k, None)

		return img_dict