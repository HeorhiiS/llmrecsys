import os
import sys
from typing import List
import torch
import transformers
import fire
from datasets import load_dataset
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
) 
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer
from utils.prompter import Prompter


def train(
    base_model: str = "../hfcheckpoints/65B/",  # the only required argument
    train_data_path: str = "../finetune_dataset/llama_train_red.json",
    val_data_path: str = "../finetune_dataset/llama_eval_red.json",
    output_dir: str = "finetuned_models/65B/",
    dataset_whole_path: str = None,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,
    use_wandb: bool = True,
    add_eos_token: bool = True,

):
    devices = 4
    # training hyperparams
    batch_size = 128
    micro_batch_size = 2
    num_epochs = 10
    learning_rate = 3e-4
    seq_len = 256
    param_space_size = "65B" # change this variable to the size of the model
    val_set_size = 2000
    epoch_size =16000
    prompt_template_name = "alpaca"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # LoRA hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]

    # Load model and tokenizer
    device_map = "auto"  # because we want to use sharding via fsdp or deepspeed
    prompter = Prompter()

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = 'right'

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

    # add wandb logging
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(

            project="llmrecsys",
            name=f"experiment-lora-{param_space_size}",
        )

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

    # Load the dataset
    if dataset_whole_path is not None:
        data = load_dataset("json", data_files=dataset_whole_path)
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            eval_data = (
                train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_data = None
    else:
        if train_data_path.endswith(".json") or train_data_path.endswith(".jsonl"):
            train_data = load_dataset("json", data_files=train_data_path)
            eval_data = load_dataset("json", data_files=val_data_path)
        else:
            train_data = load_dataset(data_files=train_data_path)
            eval_data = load_dataset(data_files=val_data_path)

        if eval_data is not None:
            train_data = (
                train_data["train"].shuffle().map(generate_and_tokenize_prompt) # apply tokenization to the dataset
            )
            eval_data = (
                eval_data["train"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_data = None

    # Train the model using trainer utility, can be modified to accept ddp and deepspeed
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=epoch_size * 2 // micro_batch_size-2 // devices,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            # ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,

        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )


    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32": # speed up training using JIT
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir) # save the final results

if __name__ == "__main__":
    fire.Fire(train)
