#!/bin/bash
# Set number of tasks to run
#SBATCH --nodes=1
# SBATCH -c 10
#BATCH --ntasks-per-node=10
#SBATCH -p nvidia
# SBATCH --gres=gpu:8
#SBATCH -C a100
#SBATCH --gpus=3
#SBATCH --mem=200G
# SBATCH -w cn007
# Walltime format hh:mm:ss
#SBATCH --time=24:30:00



# Output and error files
#SBATCH -o finetuneGPTJ.out
#SBATCH -e finetuneGPTJ.err

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
# torchrun --nproc_per_node=3 python customllamatrain.py
python gptjfinetune.py