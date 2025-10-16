import torch
import torch.nn as nn

class RMSnorm(nn.Module):
    def __init__(self,hidden_dim,eps=1e-6):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(hidden_dim))

    def forward(self,x):
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)

        output = output * self.gamma

        return output

x = torch.rand(4,6,8)
model = RMSnorm(8)
o1 = model(x)
print(o1.shape)

print(torch.norm(o1[0, 0, :]))
import numpy as np
print(np.sqrt(8))
