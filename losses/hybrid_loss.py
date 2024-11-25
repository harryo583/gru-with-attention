import torch 
import torch.nn as nn 
from .correlation_loss import CorrelationLoss

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.correlation_loss = CorrelationLoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        correlation = self.correlation_loss(y_pred, y_true)
        return self.alpha * mse + (1 - self.alpha) * correlation
