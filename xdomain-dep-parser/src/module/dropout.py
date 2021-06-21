from typing import Sequence, List

import torch
import torch.nn as nn


class IndependentDropout(nn.Module):

    def __init__(self, p: float=0.5):
        super(IndependentDropout, self).__init__()
        self.p = p

    def forward(self, *items: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]
        return items


class SharedDropout(nn.Module):

    def __init__(self, p: float=0.5, batch_first: bool=True):
        super(SharedDropout, self).__init__()
        self.p, self.batch_first = p, batch_first

    @staticmethod
    def get_mask(x: torch.Tensor, p: float) -> torch.Tensor:
        mask = x.new_empty(x.shape).bernoulli_(1-p)
        mask = mask / (1-p)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    