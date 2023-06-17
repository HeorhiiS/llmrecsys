#!/bin/bash
# Set number of tasks to run
#SBATCH -n 8
#SBATCH -c 5
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:8
# Walltime format hh:mm:ss
#SBATCH --time=5:30:00
# Output and error files
#SBATCH -o llamain.out
#SBATCH -e llamain.err

# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

# **** Actual commands start here ****
# Load modules here (safety measure)
module purge
source ~/.bashrc
conda activate llamaenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# You may need to load gcc here .. This is application specific
# module load gcc
# Replace this with your actual command. 'serial-hello-world' for example
# Set MP, set TARGET_FOLDER to the folder containing the model and tokenizer
#TARGETFOLDER = llamadownloads, 1 -> 7B, 2 -> 13B, 4 -> 30B, 8 -> 65B
torchrun --nproc_per_node 4 example.py --ckpt_dir llamadownloads/30B --tokenizer_path llamadownloads/tokenizer.model


