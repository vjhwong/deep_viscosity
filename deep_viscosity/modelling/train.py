import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from math import sqrt

from modelling.modified_loss import WeightedMSELoss


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_loss_values = []
    val_loss_values = []

    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss_train_sum = 0
        epoch_loss_val_sum = 0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            epoch_loss_train_sum += train_loss

            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_loss_train_sum = epoch_loss_train_sum / len(train_loader)
        epoch_loss_train_sum = sqrt(epoch_loss_train_sum + 1e-6)
        train_loss_values.append(epoch_loss_train_sum)

        # here starts the code for the validation
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

        wandb.log({"train_loss": epoch_loss_train_sum, "val_loss": epoch_loss_val_sum})

        if epoch % 5 == 0:
            print()
            print(f"Validation targets:\n{val_targets}")
            print()
            print(f"Validation output:\n{val_outputs}")
            print(f"Validation loss: {epoch_loss_val_sum}")
            print("\n\n")
    # print(
    #     f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
    # )
    print(f"Final train loss: {train_loss_values[-1]}")
    print(f"Final validation loss: {val_loss_values[-1]}")
    plt.plot(range(num_epochs), train_loss_values, label="Training Loss")
    plt.plot(range(num_epochs), val_loss_values, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.grid()
    plt.legend()
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
