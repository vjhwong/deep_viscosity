import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from modelling.model import CNN3DVisco
import matplotlib.pyplot as plt


def test(model: torch.nn.Module, test_loader: DataLoader) -> None:
    """Test the model on the test set and plot the scatter plot of targets vs outputs.

    Args:
        model (torch.nn.Module): Model to test.
        test_loader (DataLoader): DataLoader for the test set.
    """
    all_targets = []
    all_outputs = []
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets.float())
            test_loss += loss.item()
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())

    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss (MSE): {avg_test_loss:.4f}")

    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs).numpy()

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_outputs, alpha=0.5)
    plt.xlabel("Targets")
    plt.ylabel("Outputs")
    plt.title("Scatter Plot of Targets vs Outputs")
    plt.grid(True)
    plt.show()