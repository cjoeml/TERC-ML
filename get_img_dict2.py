from PIL import Image, IptcImagePlugin
import os
import pandas as pd
import csv
import pprint

def get_img_dict(directory, filename="tags.csv", cutoff=20):
    """ Parses the tags.csv file and builds a dictionary mapping
        filenames to their tags

    Arguments:
    directory -- directory containing the 'tags.csv' file
    """
    with open(directory + filename, 'r') as tags:
        reader = csv.reader(tags, delimiter=',')
        img_dict = {}
        unique_tags = []

        for row in reader:
            try:
                # print(row)
                tag_list = row[1:]
                # print(tag_list)
                for tag in tag_list:
                    img_dict[tag[1:]] = img_dict.get(tag[1:], 0) + 1
                    if tag not in unique_tags:
                        unique_tags.append(tag)
            except(Exception):
                pass

        # We should probably double check which tags are relevant or not on Slack
        removeIrrelevantTags = ['1', 'Tags', 'Windows on Earth', '#WinEarthFavs', 'Astronaut Favs', 'Private', 'BUSpark', 'Christ Hadfield tweet',
                                'Scott Kelly tweet', 'Anousheh Ansari', 'Anousheh_All_Sun_Moon', 'Anousheh_Earth', 'Anousheh_Iran', 'Reid_Wiseman',
                                '#MovieSession - Reviewed', 'ReidWiseman movie tweet', 'WinEarthTweet', 'Maynard Show', 'BarnesAndNoble', 'ReidFavs', 'Leroy Chiao', 'Movie']

        # Removing unneeded tags that we can't measure
        for k in removeIrrelevantTags:
            print("Popped {}".format(k))
            img_dict.pop(k, None)

        sorted_tags = sorted(img_dict, key=img_dict.__getitem__, reverse=True)
        ttl = 0
        for k in sorted_tags:
            if (img_dict[k] >= cutoff):
                print("{} : {}".format(k, img_dict[k]))
            ttl += img_dict[k]
        print("======================\nTotal tags: " + str(ttl))
        TOTAL_TAGS = len(sorted_tags)
        print("Total unique tags: " + str(TOTAL_TAGS))

        return

if(__name__ == "__main__"):
    get_img_dict("./", "BU10000Set.csv", cutoff=20)
