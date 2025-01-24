import torch 
import torch.nn as nn 
from .correlation_loss import CorrelationLoss

# Define a custom hybrid loss function that combines MSE loss and correlation loss.
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Initialize the HybridLoss class.

        Args:
            alpha (float): Weighting factor for the MSE loss component. 
                           The correlation loss component will be weighted as (1 - alpha).
                           Default value is 0.5, which gives equal importance to both losses.
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # Weighting parameter for MSE loss
        self.mse_loss = nn.MSELoss()  # Mean Squared Error loss
        self.correlation_loss = CorrelationLoss()  # Custom correlation-based loss

    def forward(self, y_pred, y_true):
        """
        Compute the hybrid loss.

        Args:
            y_pred (torch.Tensor): Predicted outputs from the model.
            y_true (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: Weighted combination of MSE loss and correlation loss.
        """
        # Compute the Mean Squared Error (MSE) loss
        mse = self.mse_loss(y_pred, y_true)

        # Compute the correlation loss
        correlation = self.correlation_loss(y_pred, y_true)

        # Return the weighted sum of MSE loss and correlation loss
        return self.alpha * mse + (1 - self.alpha) * correlation
