import os
import sys
import fire
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
import torch
from datasets import load_dataset
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

def train(

    base_model: str = "models--EleutherAI--gpt-j-6B",
    output_dir: str = "finetuned_models/30B/",
    train_data_path: str = "../litllamadata/finetune_dataset/llama_train_red.json",
    val_data_path: str = "../litllamadata/finetune_dataset/llama_eval_red.json",
    use_wandb: bool = True,
    add_eos_token: bool = True,
):

    batch_size = 128
    micro_batch_size = 4
    num_epochs = 10
    learning_rate = 3e-4
    seq_len = 256
    # Load model and tokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # LoRA hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
    device_map = "auto"
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < seq_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    # add LoRA config
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)





if __name__ == "__main__":
    fire.Fire(train)