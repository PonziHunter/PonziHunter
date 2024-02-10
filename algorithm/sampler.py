from typing import Optional
from torch.utils.data import WeightedRandomSampler
from torch import Tensor
import torch


class ImbalancedSampler(WeightedRandomSampler):
    def __init__(self, labels: Tensor, num_samples: Optional[int]=None):

        labels = labels.view(-1)
        if labels.type != torch.long:  y = labels.long()
        else:  y = labels

        num_samples = y.numel()
        class_weight = 1. / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)
