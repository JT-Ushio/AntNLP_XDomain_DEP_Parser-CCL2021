import torch
import torch.nn as nn

from .dropout import SharedDropout


class MLP(nn.Module):

    def __init__(self, d_in: int, d_hid: int, p_drop:float=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(d_in, d_hid)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x