import torch
import math
import torch.nn as nn
def swish(x, beta):
    return x * torch.sigmoid(x * beta)

class LoRALinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
    ):
        super(LoRALinear, self).__init__()

        self.r = r
        assert r > 0

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        out = x @ self.lora_A.T @ self.lora_B.T
        return out