# TERC-ML - Team 4

Image Sets and Tags:

https://spark-xlab.slack.com/archives/C70DFLJ7M/p1510588534000126
https://spark-xlab.slack.com/archives/C70DFLJ7M/p1510588582000116
https://spark-xlab.slack.com/archives/C70DFLJ7M/p1511291170000494


Testing was done using the submit.sh bash script on scc.bu.edu. Place the raw images under ./data/raw and either use the script or run
bottleneck_model.py. The weights from our model are located in the final results folder.


File descriptions:

bottleneck_model.py - where the model is
datagenerator.py - code to transform images
get_data.py - code for extracting and formating tags
get_img_dict - file containing a function for mapping image names to their tags
BU10000Set.csv - file containing tags for all images

Folder Descriptions:

logs -  output from running the model on the SCC
tests - collecton of runs using different parameters and model architectures
final results - contains the output from our best run
initial results - contains the output of our initial model
deprecated - older, deprecated code

