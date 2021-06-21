from typing import Tuple, List

import torch
import torch.nn as nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence

from .dropout import SharedDropout


class BiLSTM(nn.Module):

    def __init__(
        self, 
        d_in: int, 
        d_hid: int, 
        n_layer: int = 1, 
        p_drop: float = 0):
        super(BiLSTM, self).__init__()

        self.d_hid, self.n_layer, self.p_drop = d_hid, n_layer, p_drop
        self.fwd, self.bwd = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.n_layer):
            self.fwd.append(nn.LSTMCell(d_in, d_hid))
            self.bwd.append(nn.LSTMCell(d_in, d_hid))
            d_in = d_hid * 2

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if len(param.shape) > 1:    # is weight
                nn.init.orthogonal_(param)
            else:   # is bias
                nn.init.zeros_(param)

    def permute_hidden(
        self, 
        hx: Tuple[torch.Tensor, torch.Tensor], 
        permutation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if permutation is None: return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)
        return h, c

    def layer_forward(
        self, 
        x: Tuple[torch.Tensor], 
        hx: Tuple[torch.Tensor, torch.Tensor], 
        cell: nn.ModuleList, 
        batch_sizes: List[int], 
        reverse:bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.p_drop)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
                        for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)
        return output, hx_n

    def forward(
        self, 
        seq: PackedSequence, 
        hx: torch.Tensor=None) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        x, batch_sizes = seq.data, seq.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.n_layer * 2, batch_size, self.d_hid)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, seq.sorted_indices)
        h = h.view(self.n_layer, 2, batch_size, self.d_hid)
        c = c.view(self.n_layer, 2, batch_size, self.d_hid)

        for i in range(self.n_layer):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.p_drop)
                x = [i * mask[:len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(
                x, (h[i,0], c[i,0]), self.fwd[i], batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(
                x, (h[i,1], c[i,1]), self.bwd[i], batch_sizes, reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(
            x, seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, seq.unsorted_indices)
        return x, hx