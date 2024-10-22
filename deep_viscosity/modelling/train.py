import os
from math import sqrt

import torch
import torch.utils.data
import wandb
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from modelling.utils.equalizedmseloss import EqualizedMSELoss
from modelling.utils.early_stopping import EarlyStopping
from modelling.utils.functions import create_run_folder


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
) -> None:
    """Train the model using the given data loader and hyperparameters.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader to use for training.
        learning_rate (float): The learning rate to use for training.
        num_epochs (int): The number of epochs to train for.
    """
    wandb.init(
        project="DeppViscosity",
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "DeepViscosity",
            "epochs": num_epochs,
        },
    )
    run_name = wandb.run.name
    create_run_folder(run_name)

    artifact = wandb.Artifact(name=f"{run_name}", type="python_files")
    artifact.add_file(
        local_path=os.path.join(
            os.getcwd(), "deep_viscosity", "modelling", "model.py"),
        name=f"{run_name}_model.py",
    )
    artifact.add_file(
        local_path=os.path.join(
            os.getcwd(), "deep_viscosity", "modelling", "train.py"),
        name=f"{run_name}_train.py",
    )
    artifact.add_file(
        local_path=os.path.join(os.getcwd(), "train_model.sh"),
        name=f"{run_name}_train_model.sh"
    )
    wandb.log_artifact(artifact)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping = EarlyStopping(patience=10, path=os.path.join(
        "models", f"{run_name}", f"{run_name}.pth"), verbose=True)
    criterion = EqualizedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    train_loss_values = []
    val_loss_values = []

    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss_train_sum = 0
        epoch_loss_val_sum = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            epoch_loss_train_sum += train_loss

            train_loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_loss_train_sum = epoch_loss_train_sum / len(train_loader)
        epoch_loss_train_sum = sqrt(epoch_loss_train_sum + 1e-6)
        train_loss_values.append(epoch_loss_train_sum)

        val_loss = 0.0
        with torch.no_grad():

            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets.float()).item()
                epoch_loss_val_sum += val_loss

        epoch_loss_val_sum = epoch_loss_val_sum / len(val_loader)
        epoch_loss_val_sum = sqrt(epoch_loss_val_sum + 1e-6)
        val_loss_values.append(epoch_loss_val_sum)

        wandb.log({"train_loss": epoch_loss_train_sum,
                  "val_loss": epoch_loss_val_sum})

        early_stopping(epoch_loss_val_sum, model)
        if early_stopping.early_stop:
            print("Early stopping")
            print(
                f"Best model train loss: {train_loss_values[val_loss_values.index(min(val_loss_values))]}")
            print(f"Best model validation loss: {min(val_loss_values)}")
            plt.plot(range(epoch+1), train_loss_values, label="Training Loss")
            plt.plot(range(epoch+1), val_loss_values, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss over Epochs")
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(
                "models", f"{run_name}", f"{run_name}_loss_plot.png"))
            return

    print(f"Final train loss: {train_loss_values[-1]}")
    print(f"Final validation loss: {val_loss_values[-1]}")
    plt.plot(range(num_epochs), train_loss_values, label="Training Loss")
    plt.plot(range(num_epochs), val_loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(
        "models", f"{run_name}", f"{run_name}_loss_plot.png"))
