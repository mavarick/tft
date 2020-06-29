import torch
import torch.nn as nn

class RMSSELoss(nn.Module):
    def __init__(self, device):
        super(RMSSELoss, self).__init__()
        self.device = device

    def forward(self, predictions, actuals):
        sequence_length = predictions.shape[1]
        numerator = torch.sum(torch.pow(predictions - actuals, 2), dim=1)
        loss = torch.div(numerator, sequence_length)
        loss = torch.sqrt(loss)
        loss = torch.mean(loss)
        return loss