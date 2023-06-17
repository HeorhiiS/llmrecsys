# Turning LLMs into recommendation systems
### What this repository has: 
 - **Experiments with LLMs for recommendation systems for MLOSS.**
 - **Finetuning scripts for LLMs**
 - **SLURM batch files for running finetuning on a cluster**
 - **LoRA weights to run this yourself**


### How do I run this myself?

#### 1. Install the requirements
Recommended: create a virtual environment (I prefer conda)
```
conda create -n llmrec python=3.9
conda activate llmrec
pip install -r requirements.txt
```

#### 2. Download the official LLAMA weights from Meta
