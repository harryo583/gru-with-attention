import torch
import torch.nn as nn 

class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_mean = y_pred - torch.mean(y_pred, dim=0, keepdim=True)
        y_true_mean = y_true - torch.mean(y_true, dim=0, keepdim=True)
        cov = torch.sum(y_pred_mean * y_true_mean, dim=0) # numerator: covariance
        pred_std = torch.sqrt(torch.sum(y_pred_mean ** 2, dim=0) + 1e-8)
        true_std = torch.sqrt(torch.sum(y_true_mean ** 2, dim=0) + 1e-8)
        corr = cov / (pred_std * true_std + 1e-8)  # adjust to prevent division by zero
        return 1 - torch.mean(corr)