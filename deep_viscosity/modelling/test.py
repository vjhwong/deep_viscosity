import torch
import torch.nn as nn
import torch.optim as optim
from modelling.model import CNN3DVisco
import matplotlib.pyplot as plt
import numpy as np

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader)-> None:
        # Collect all targets and outputs for plotting
    all_targets = []
    all_outputs = []
    criterion= nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets.float())
            test_loss += loss.item() ## denna fattar jag ej
            all_targets.append(targets.cpu())  
            all_outputs.append(outputs.cpu())  
        
        
    avg_test_loss = test_loss/ len(test_loader)
    
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    
    all_targets = torch.cat(all_targets).numpy()  # Convert to a NumPy array
    all_outputs = torch.cat(all_outputs).numpy()  # Convert to a NumPy array
    

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_outputs, alpha=0.5)
    plt.xlabel('Targets')
    plt.ylabel('Outputs')
    plt.title('Scatter Plot of Targets vs Outputs')
    plt.grid(True)
    plt.show()
    
    
    
            
    