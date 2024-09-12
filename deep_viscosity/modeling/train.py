import torch
from modeling.deep_viscosity_model import CNN3DVisco
import torch.nn as nn
import torch.optim as optim


def train(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
    # Set device, standard is to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set hyperparameters, godtyckliga värden
    learning_rate = 0.001
    num_epochs = 10

    # Define loss function and optimizer
    # change criterion to least squares loss

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Set the model to training mode
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}"
                )

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")
