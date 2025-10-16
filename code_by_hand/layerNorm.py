import torch
import torch.nn as nn

class LN(nn.Module):
    def __init__(self,hidden_dim,eps=1e-6):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self,x):
        b,s,h = x.size()

        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)

        output = (x-mean)/(std+self.eps)

        output = output * self.gamma + self.bias

        return output

x = torch.rand(4,6,8)
model = LN(8)
o1 = model(x)
print(o1.shape)
print(o1[0,0,:].mean(dim=0))
print(o1[0,0,:].std(dim=0, unbiased=False))

model2 = nn.LayerNorm(8)
o2 = model2(x)
print(o2[0,0,:].mean(dim=0))
print(o2[0,0,:].std(dim=0, unbiased=False))
