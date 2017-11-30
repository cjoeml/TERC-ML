#!/bin/bash -l

#Specify project
#$ -P cs542

#Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=12:00:00

#Send an email when the job is done or aborted (by default no email is sent)
#$ -m e

# Give job a name

#$ -N Training

#request kits if memory
#$ -l mem_per_core=8G

# request 18 cpus because why not
#$ -pe omp 16

#request 2 gpus because why not
#$ -l gpus=.125 -l gpu_c=3.5  

# Join output and error streams into one file
#$ -j y

#name output
#$ -o results.txt

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

