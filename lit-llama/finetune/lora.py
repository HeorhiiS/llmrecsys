"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time
import wandb
import configparser

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig, Block
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from jsonargparse.cli import CLI

# Extra optimization functions
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lightning.fabric.strategies import FSDPStrategy, DeepSpeedStrategy

eval_interval = 100
save_interval = 100
eval_iters = 100
log_interval = 1
devices = 6
# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 2
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 20000  # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // devices
weight_decay = 0.0
max_seq_length = 256  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100
param_space_size ="7B"


def main(
    data_dir: str = "litllamadata/finetune_dataset/",
    pretrained_path: str = f"checkpoints/lit-llama/{param_space_size}/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = f"litllamadata/finetuned_models/{param_space_size}/",
):
    config = configparser.ConfigParser()
    config.read('config.ini')
    api_key = config.get('API', 'key')
    # wandb.login(key=api_key)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)
    # For slurm based cluster must use ddp

    fabric = L.Fabric(accelerator="gpu", devices=devices, num_nodes=2, precision="16-mixed", strategy='fsdp')
    fabric.seed_everything(1337 + fabric.global_rank)
    fabric.launch()



    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        wandb.init(

            project="llmrecsys",

            name=f"experiment-lora-{param_space_size}",

            config={
                "learning_rate": learning_rate,
                "architecture": f"lit-llama{param_space_size}",
                "dataset": "custom-movielens",
                "max iterations": max_iters,
            })

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name(param_space_size)
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)
    print(f"Loaded pretrained weights from {pretrained_path}")
    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    print("Starting training...")
    train(fabric, model, optimizer, train_data, val_data, tokenizer_path, out_dir)
    wandb.finish()
    # Save the final LoRA checkpoint at the end of training
    print(f"Saving final LoRA weights to {out_dir}")
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, f"lit-llama{param_space_size}-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        fabric.backward(loss)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, tokenizer_path)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                wandb.log({"validation loss": val_loss, "iteration": iter_num})
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-lora-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            print(f"iter {iter_num}/{max_iters}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms")
            wandb.log({"loss": loss.item(), "iteration": iter_num})


def generate_response(model, instruction, input, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()
    instruction = "Given a list of liked movies, recommend 4 more movies the user would like.",
    input_ = "Truman Show, The (1998), Good Will Hunting (1997), Backdraft (1991), Contact (1997), Mission: Impossible 2 (2000), Stargate (1994) =>"
    # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, input_, tokenizer_path)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    CLI(main)
