#!/bin/bash
# Set number of tasks to run
#SBATCH --nodes=1
#SBATCH -c 40
# SBATCH -n 10
# BATCH --ntasks-per-node=8
#SBATCH -p nvidia
# BATCH --gres=gpu:8
# SBATCH -C a100
#SBATCH --gpus=8
#SBATCH --mem=400G
#SBATCH -w dn002
# Walltime format hh:mm:ss
#SBATCH --time=48:30:00



# Output and error files
#SBATCH -o finetune65B.out
#SBATCH -e finetune65B.err

# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
# module load gcc
source ~/.bashrc
conda activate llamaenv
# export WANDB_MODE="run"
# numqueued.sh
# how-my-limits
# You may need to load gcc here .. This is application specific
# module load gcc
# Replace this with your actual command. 'serial-hello-world' for example
# Set MP, set TARGET_FOLDER to the folder containing the model and tokenizer
# TARGETFOLDER = llamadownloads, 1 -> 7B, 2 -> 13B, 4 -> 30B, 8 -> 65B
# torchrun --nproc_per_node 1 customllamatrain.py
python customllamatrain.py

