# Turning LLMs into recommendation systems 🐝

## What this repository has: 

 - **Experiments with LLMs for recommendation systems for MLOSS on movielens dataset**
 - **Finetuning scripts for LLMs**
 - **SLURM batch files for running finetuning on a cluster**
 - **LoRA weights to run this yourself**


### How do I run this myself?

#### 1. Install the requirements

Recommended: create a virtual environment (I prefer conda)

```bash
conda create -n llmrec python=3.9
conda activate llmrec
pip install -r requirements.txt
```

#### 2. Download the official LLAMA weights from Meta

Paste the presign URL into the bash script `slurm/download.sh` and run it. This will download the weights and tokenizer to `llamadownloads/`.

For GPT-J the weight will be loaded and cached automatically.

#### 3. Apply converstion script to convert the weights to the right format. SLURM version in `slurm/prepdata.sh`

```bash
python ../convert_llama_weights_to_hf.py \
    --input_dir ../weights_dir/ --model_size 65B --output_dir ../converted_checkpoints/65B
```

#### 4. Run the finetuning script

```bash
python lora_llm/customllamatrain.py
```

#### 5. Run the evaluation script

```bash
python lora_llm/customllamaeval.py
```

For GPT-J, the procedure is the same, but you can skip step 2 and 3. Scrits use wandb for logging, so you need to modifty the scripts if you don't want to use it or provide your own API key and apply `wandb.login()`.

You can also play around with prompt and the dataset in the notebook makemydata.ipynb. You will need to download the dataset from [here](https://grouplens.org/datasets/movielens/). The notebook will prepare the data for you for LLAMA. For GPT-J the script will combine instruction and input fields into one field and add endoftext tokens.
