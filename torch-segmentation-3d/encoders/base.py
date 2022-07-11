import torch
from torch import nn
from abc import abstractmethod


class BaseEncoder(nn.Module):
    @abstractmethod
    def get_stages(self):
        raise NotImplementedError
