import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from functools import wraps


def lora2dora(model):
    '''replace lora with dora'''

    def print_args(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(
                f"Calling {func.__name__} with args: {[type(x) for x in args]}, kwargs: {kwargs}"
            )
            return func(*args, **kwargs)

        return wrapper

    class LoRALayer(nn.Module):

        def __init__(self, in_dim, out_dim, rank, alpha):
            super().__init__()
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.B = nn.Parameter(torch.zeros(rank, out_dim))
            self.alpha = alpha

            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

        def forward(self, x):
            x = self.alpha * (x @ self.A @ self.B)
            return x

    # print(res)

    class LinearWithDoRAMerged(nn.Module):

        def __init__(self, linear, rank, alpha):
            super().__init__()
            self.linear = nn.Linear(
                in_features=linear.in_features,
                out_features=linear.out_features)
            self.linear.weight = nn.Parameter(linear.weight)
            self.linear.bias = nn.Parameter(linear.bias)
            self.lora = LoRALayer(linear.in_features, linear.out_features, rank,
                                  alpha)
            self.m = nn.Parameter(
                self.linear.weight.norm(p=2, dim=0, keepdim=True))

            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False

        # Code loosely inspired by
        # https://github.com/catid/dora/blob/main/dora.py

        # @print_args
        def forward(self, x):
            lora = self.lora.A @ self.lora.B
            numerator = self.linear.weight + self.lora.alpha * lora.T
            denominator = numerator.norm(p=2, dim=0, keepdim=True)
            directional_component = numerator / denominator
            new_weight = self.m * directional_component
            return F.linear(x, new_weight, self.linear.bias)

    for name, module in model.named_modules():
        if name.endswith('self'):
            if module.query.loras:
                new_query = LinearWithDoRAMerged(module.query, 32, 1.0)
                setattr(module, 'query', new_query)
            if module.value.loras:
                new_value = LinearWithDoRAMerged(module.value, 32, 1.0)
                setattr(module, 'value', new_value)
