import torch
import torch.nn as nn

class SMAPELoss(torch.nn.Module):
    def __init__(self, device):
        super(SMAPELoss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        sequence_length = predictions.shape[1]
        predictions = predictions.float()
        actuals = actuals.float()
        sumf = torch.sum(torch.abs(predictions - actuals) / (torch.abs(predictions) + torch.abs(actuals)), dim=1)

        return torch.mean((2 * sumf) / sequence_length)