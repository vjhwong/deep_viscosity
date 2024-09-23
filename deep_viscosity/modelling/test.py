import torch
import torch.nn as nn
import torch.optim as optim
from modelling.model import CNN3DVisco
import matplotlib.pyplot as plt

def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    
    criterion= nn.MSELoss()
    
    model.eval()
    
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets.float())
            test_loss += loss.items() ## denna fattar jag ej
        
        
    avg_test_loss = test_loss/ len(test_loader)
    
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    
    return avg_test_loss  
            
    