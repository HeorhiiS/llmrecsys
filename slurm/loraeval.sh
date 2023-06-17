#!/bin/bash
# Set number of tasks to run
#SBATCH --nodes=1
#SBATCH -c 40
# SBATCH -n 10
# BATCH --ntasks-per-node=8
#SBATCH -p nvidia
# SBATCH --gres=gpu:3
#SBATCH -C a100
#SBATCH --gpus=3
#SBATCH --mem=300G
# SBATCH -w dn001
# Walltime format hh:mm:ss
#SBATCH --time=10:30:00



# Output and error files
#SBATCH -o eval65B.out
#SBATCH -e eval65B.err
# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
# module load gcc
source ~/.bashrc
conda activate llamaenv # --> replace with your conda environment name
# You may need to load gcc here .. This is application specific
# module load gcc
# Replace this with your actual command. 'serial-hello-world' for example
# Set MP, set TARGET_FOLDER to the folder containing the model and tokenizer
# torchrun --nproc_per_node 1 customllamatrain.py --> use for distributed training, may need to use srun
python generate_responses.py