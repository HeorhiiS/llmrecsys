import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import wandb
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

def train(
    base_model: str = "hfcheckpoints/7B/",  # the only required argument
    train_data_path: str = "yahma/alpaca-cleaned",
    val_data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "/alpaca_lora/finetuned_models/",










):
    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    batch_size = 128
    micro_batch_size = 4
    num_epochs = 3
    learning_rate = 3e-4
    seq_len = 256
    param_space_size = "7B"

    # LoRA hyperparams
    lora_r = 8,
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt


    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(

            project="llmrecsys",

            name=f"experiment-adapterv2-{param_space_size}",

            config={
                "learning_rate": learning_rate,
                "architecture": f"lit-llama{param_space_size}",
                "dataset": "custom-movielens",
                "max iterations": max_iters,
            })

    device_map = "auto"  # because we want to use sharding via fsdp or deepspeed

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
        train_data = load_dataset("json", data_files=train_data_path)
        eval_data = load_dataset("json", data_files=val_data_path)
    else:
        train_data = load_dataset(data_files=train_data_path)
        eval_data = load_dataset(data_files=val_data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    BASE_MODEL = checkpoint_path

    model.save_pretrained(output_dir)

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)