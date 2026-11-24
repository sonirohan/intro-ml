"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig

def solver(model_name):
    # Initialize the model
    if model_name == "bigram":
        config = BigramConfig
        model = BigramLanguageModel(config)
    elif model_name == "minigpt":
        config = MiniGPTConfig
        model = MiniGPT(config)
    else:
        raise ValueError("Invalid model name")
    
    # Load the dataset
    train_dataset = TinyStoriesDataset(
        config.path_to_data,
        mode="train",
        context_length=config.context_length,
    )
    eval_dataset = TinyStoriesDataset(
        config.path_to_data, mode="test", context_length=config.context_length
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, pin_memory=True
    )

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print number of parameters in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

    # Initialize wandb if you want to use it
    if config.to_log:
        wandb.init(project="dl2_proj3")

    # Create the save path if it does not exist
    if not Path.exists(config.save_path):
        Path.mkdir(config.save_path, parents=True, exist_ok=True)

    # Define the loss function
    loss = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2
        )

    model.train()
    model.to(device)

    for i, (context, target) in enumerate(train_dataloader):

        # Do the forward pass, compute the loss, do the backward pass, and update the weights with the optimizer.
        optimizer.zero_grad()
        context = context.to(device)
        target = target.to(device)
        output = model(context)
        output = rearrange(output, "batch seq vocab -> (batch seq) vocab")
        target = rearrange(target, "batch seq -> (batch seq)")
        loss_val = loss(output, target)
        loss_val.backward()
        train_loss = loss_val.item()
        if config.to_clip_grad:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )  # Gradient Clipping.
        optimizer.step()

        if config.scheduler:
            scheduler.step()

        del context, target # Clear memory

        if i % config.log_interval == 0:

            model.eval()
            # The eval dataloader never runs out, so compute the loss on a single batch.
            with torch.no_grad():
                for _, (context, target) in enumerate(eval_dataloader):
                    context = context.to(device)
                    target = target.to(device)
                    output = model(context)
                    output = rearrange(output, "batch seq vocab -> (batch seq) vocab")
                    target = rearrange(target, "batch seq -> (batch seq)")
                    eval_loss = loss(output, target).item()
                    break

            print(
                f"Iteration {i}, Train Loss: {train_loss}",
                f"Eval Loss: {eval_loss}",
            )

            if config.to_log:
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Eval Loss": eval_loss,
                    }
                )

            model.train()

        # Save the model every config.save_iterations
        if i % config.save_iterations == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_checkpoint_{i}.pt",
            )

        if i > config.max_iter:
            break