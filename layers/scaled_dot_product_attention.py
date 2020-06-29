from torch import nn
import torch

class ScaledDotProductAttention(nn.Module):
    """Defines scaled dot product attention layer.

      Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
          softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
          q: Queries
          k: Keys
          v: Values
          mask: Masking if required -- sets softmax to very large value

        Returns:
          Tuple of (layer outputs, attention weights)
        """
        temper = torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float))
        attn = torch.bmm(q,k.permute(0,2,1)) # shape=(batch, q, k)
        if mask is not None:
            mmask = (-1e+9) * (1. - torch.tensor(mask, dtype=torch.float)) # setting to infinity
            attn = torch.add(attn, mmask)
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn,v)
        return output, attn
