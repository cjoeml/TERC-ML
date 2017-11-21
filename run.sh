#!/bin/bash -l




#Specify project
# -P cs542

#Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
# -l h_rt=1:00:00

#Send an email when the job is done or aborted (by default no email is sent)
# -m e

# Give job a name
# -N Training

#$ Join output and error streams into one file
# -j y

qrsh -P "cs542" -N "Training" 
#-l gpus=2 -l gpu_c=3.5  

#load appropriate environment
module load python/3.6.1
module load python/3.6.2
module load cuda/8.0
module load cudnn/6.0
module load tensorflow/r1.3_cpu
module load keras
module load h5utils/1.12.1
#execute the program
python learning_model.py

