import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        """Initialize the custom MSELoss with class weights.

        """
        super(WeightedMSELoss, self).__init__()
        self.class_weights = torch.Tensor(1/3,1/3,1/3)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """Compute the weighted MSE loss.
        
        Args:
            predictions (torch.Tensor): The model predictions.
            targets (torch.Tensor): The ground truth values.
            class_labels (torch.Tensor): The class labels (used for weighting the loss).
        
        Returns:
            torch.Tensor: The weighted MSE loss.
        """
        # Compute the standard MSE loss
        
        mse_loss = (predictions - targets) ** 2
        
        # Apply class weights: each element in the batch is multiplied by the corresponding class weight
        weights = self.class_weights[class_labels]
        weighted_loss = weights * mse_loss
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()