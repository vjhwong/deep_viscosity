import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from modelling.model import CNN3DVisco
from tqdm import tqdm


def train(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int,
) -> None:
    """Train the model using the given data loader and hyperparameters.

    Args:
        model (nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The data loader to use for training.
        learning_rate (float): The learning rate to use for training.
        num_epochs (int): The number of epochs to train for.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 batches
            # if (batch_idx + 1) % 10 == 0:
        # print(f"Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
