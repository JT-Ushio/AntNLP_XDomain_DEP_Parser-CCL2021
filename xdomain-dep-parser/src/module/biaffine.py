import torch
import torch.nn as nn


class Biaffine(nn.Module):

    def __init__(
        self, 
        d_in: int, 
        d_out: int = 1, 
        has_xbias: bool = True, 
        has_ybias: bool = True):
        super(Biaffine, self).__init__()
        self.has_xbias, self.has_ybias = has_xbias, has_ybias
        self.W = nn.Parameter(torch.Tensor(d_out, d_in+has_xbias, d_in+has_ybias))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.W)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.has_xbias:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.has_ybias:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, d_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.W, y)
        # remove dim 1 if d_out == 1
        s = s.squeeze(1)

        return s