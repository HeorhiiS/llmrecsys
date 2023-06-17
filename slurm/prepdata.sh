#!/bin/sh
# Set number of tasks to run
##SBATCH -n 4
#SBATCH -c 40
#SBATCH --nodes=1
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
# Walltime format hh:mm:ss
#SBATCH --time=5:30:00
# Output and error files
#SBATCH -o prep65.out
#SBATCH -e prep65.err

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
python ../convert_llama_weights_to_hf.py \
    --input_dir ../weights_dir/ --model_size 65B --output_dir ../converted_checkpoints/65B