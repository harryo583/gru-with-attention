import torch
import torch.nn as nn 

# Define a custom loss function that calculates correlation-based loss.
class CorrelationLoss(nn.Module):
    def __init__(self):
        """
        Initialize the CorrelationLoss class.
        
        This loss measures the negative correlation between predicted and true values.
        """
        super(CorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Compute the correlation loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs from the model.
            y_true (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: 1 minus the mean correlation value across the batch.
        """
        y_pred_mean = y_pred - torch.mean(y_pred, dim=0, keepdim=True)
        y_true_mean = y_true - torch.mean(y_true, dim=0, keepdim=True)
        cov = torch.sum(y_pred_mean * y_true_mean, dim=0)  # Numerator: covariance
        pred_std = torch.sqrt(torch.sum(y_pred_mean ** 2, dim=0) + 1e-8)
        true_std = torch.sqrt(torch.sum(y_true_mean ** 2, dim=0) + 1e-8)
        corr = cov / (pred_std * true_std + 1e-8)  # Small epsilon to avoid division by zero
        return 1 - torch.mean(corr)
