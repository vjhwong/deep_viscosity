import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        """Initialize the custom MSELoss with class weights.

        """
        super(WeightedMSELoss, self).__init__()
        self.class_weights = torch.Tensor([1/3,1/3,1/3])
        self.low_threshold = 312.7
        self.high_threshold = 625.3
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the RMSE loss class-wise and averages it.
        
        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Custom RMSE loss.
        """
        low_mask = targets <= self.low_threshold
        medium_mask = (targets > self.low_threshold) & (targets < self.high_threshold)
        high_mask = targets >= self.high_threshold

        loss = 0
        class_count = 0
        
        if low_mask.sum() > 0:
            loss_low = torch.mean((predictions[low_mask] - targets[low_mask]) ** 2)
            loss += loss_low
            class_count += 1
        
        if medium_mask.sum() > 0:
            loss_medium = torch.mean((predictions[medium_mask] - targets[medium_mask]) ** 2)
            loss += loss_medium
            class_count += 1

        if high_mask.sum() > 0:
            loss_high = torch.mean((predictions[high_mask] - targets[high_mask]) ** 2)
            loss += loss_high
            class_count += 1

        return loss / class_count if class_count > 0 else torch.tensor(0.0, requires_grad=True)


