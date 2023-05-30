import lightning as L

from lightning.fabric.strategies import FSDPStrategy

import torch

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from lit_llama.model import Block, LLaMA, LLaMAConfig

def main():

    # ⚡️⚡️⚡️⚡️⚡️ Initialize FSDP strategy ⚡️⚡️⚡️⚡️⚡️

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)

    # ⚡️⚡️⚡️⚡️⚡️ Initialize Fabric ⚡️⚡️⚡️⚡️⚡️

    # setting for 4 GPUs with bf16 mixed precision and FSDP distributed training strategy

    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed", strategy=strategy)

    fabric.launch()

    # Load data

    train_data, val_data = load_datasets()

    # Load model configs

    config = LLaMAConfig.from_name("7B")

    config.block_size = block_size

    config.vocab_size = 100  # from prepare_shakespeare.py

    # ⚡️⚡️⚡️⚡️⚡️ initialize model ⚡️⚡️⚡️⚡️⚡️

    with fabric.device:

        model = LLaMA(config)

    # ⚡️⚡️⚡️⚡️⚡️ Setup model and optimizer for distributed training ⚡️⚡️⚡️⚡️⚡️

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_data, val_data)