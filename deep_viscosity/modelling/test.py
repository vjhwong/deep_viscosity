from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from modelling.utils.equalizedmseloss import EqualizedMSELoss


def test(model: torch.nn.Module, test_loader: DataLoader, train_loader: DataLoader, val_loader: DataLoader, plot_all: bool) -> None:
    """Test the model on the test set and plot the scatter plot of targets vs outputs.

    Args:
        model (torch.nn.Module): Model to test.
        test_loader (DataLoader): DataLoader for the test set.
        train_loader (DataLoader): DataLoader for the test set.
        val_loader (DataLoader): DataLoader for the test set
        plot_all: (bool): Determines if the training and validation data should be included in the prediction plot.
    """
    criterion = EqualizedMSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_targets = []
    test_outputs = []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss = sqrt(loss + 1e-6)
            test_loss += loss
            test_targets.append(targets.cpu())
            test_outputs.append(outputs.cpu())
    print(f"This is the target:{targets} and this is the true: {outputs}")
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss (RMSE): {avg_test_loss:.4f}")
    test_targets = torch.cat(test_targets).numpy()
    test_outputs = torch.cat(test_outputs).numpy()

    if plot_all:
        train_targets = []
        train_outputs = []
        with torch.no_grad():
            for inputs, targets in tqdm(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                train_targets.append(targets.cpu())
                train_outputs.append(outputs.cpu())
        train_targets = torch.cat(train_targets).numpy()
        train_outputs = torch.cat(train_outputs).numpy()

        val_targets = []
        val_outputs = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_targets.append(targets.cpu())
                val_outputs.append(outputs.cpu())
        val_targets = torch.cat(val_targets).numpy()
        val_outputs = torch.cat(val_outputs).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(test_targets, test_outputs, alpha=0.5, color='blue', label='Test predictions')
    if plot_all:
        ax.scatter(train_targets, train_outputs, alpha=0.5, color='green', label='Train predictions')
        ax.scatter(val_targets, val_outputs, alpha=0.5, color='red', label='Validation predictions')
        x_vals = np.linspace(min(train_targets), max(train_targets), 100)
        y_vals = x_vals
        ax.plot(x_vals, y_vals, color="black", linestyle="--", label="Targets = Predictions")
    else: 
        x_vals = np.linspace(min(test_targets), max(test_targets), 100)
        y_vals = x_vals
        ax.plot(x_vals, y_vals, color="black", linestyle="--", label="Targets = Predictions")
    ax.set_xlabel("Targets")
    ax.set_ylabel("Predictions")
    ax.set_title("Scatter Plot of Targets vs Outputs")
    ax.grid(True)
    ax.legend()


    return (fig, test_targets, test_outputs)
