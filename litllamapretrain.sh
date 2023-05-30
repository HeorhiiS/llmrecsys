#!/bin/bash
# Set number of tasks to run
##SBATCH -n 4
#SBATCH -c 40
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:8
#SBATCH --mem=480G
# Walltime format hh:mm:ss
#SBATCH --time=24:30:00
# Output and error files
#SBATCH -o finetune.out
#SBATCH -e finetune.err

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
python lit-llama/finetune/lora.py