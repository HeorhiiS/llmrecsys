#!/bin/bash
# Set number of tasks to run
#SBATCH --nodes=1
# SBATCH -c 10
#BATCH --ntasks-per-node=3
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:3
# SBATCH -C a100
# SBATCH --gpus=6
#SBATCH --mem=200G
# SBATCH -w cn004,cn005
# Walltime format hh:mm:ss
#SBATCH --time=24:30:00



# Output and error files
#SBATCH -o finetune7B.out
#SBATCH -e finetune7B.err

# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
# module load gcc
source ~/.bashrc
conda activate llamaenv
# numqueued.sh
# how-my-limits
# You may need to load gcc here .. This is application specific
# module load gcc
# Replace this with your actual command. 'serial-hello-world' for example
# Set MP, set TARGET_FOLDER to the folder containing the model and tokenizer
# TARGETFOLDER = llamadownloads, 1 -> 7B, 2 -> 13B, 4 -> 30B, 8 -> 65B
srun python lit-llama/finetune/adapter_v2.py
