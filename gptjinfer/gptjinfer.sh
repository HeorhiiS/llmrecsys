#!/bin/bash
# Set number of tasks to run
#SBATCH -n 28
#SBATCH -p nvidia
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
# Walltime format hh:mm:ss
#SBATCH --time=8:30:00
# Output and error files
#SBATCH -o gptjin.out
#SBATCH -e gptjin.err

# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
source ~/.bashrc
conda activate llamaenv
# You may need to load gcc here .. This is application specific
# module load gcc
# Replace this with your actual command. 'serial-hello-world' for example
# Set MP, set TARGET_FOLDER to the folder containing the model and tokenizer
#TARGETFOLDER = llamadownloads, 1 -> 7B, 2 -> 13B, 4 -> 30B, 8 -> 65B
python gptj_run.py