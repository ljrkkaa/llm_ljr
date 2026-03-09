import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self, hidden_dim, num_head, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_head = num_head
        assert hidden_dim % num_head ==0

        self.head_dim = hidden_dim // num_head
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim,hidden_dim)
        
        self.Dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(hidden_dim,hidden_dim)
    
    def forward(self, x, mask=None):
        
        b,s = x.shape[0], x.shape[1]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b,s,self.num_head,self.head_dim).transpose(1,2)
        k = k.view(b,s,self.num_head,self.head_dim).transpose(1,2)
        v = v.view(b,s,self.num_head,self.head_dim).transpose(1,2)

        atten = q @ k.transpose(-1,-2) /math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.tril()
            atten = atten.masked_fill(mask==0,-1e9)
        else:
            mask = torch.tril(torch.ones(s,s))
            atten = atten.masked_fill(mask==0,-1e9)

        atten = self.Dropout(F.softmax(atten,dim=-1))

        out = atten @ v
        out = out.transpose(1,2).contiguous().view(b,s,-1)
        out = self.o_proj(out)

        return out
    
class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.up_proj = nn.Linear(hidden_dim, 4*hidden_dim)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim)

        self.act = nn.ReLU()

        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.Dropout(self.down_proj(self.act(self.up_proj(x))))

class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_head, dropout=0.1):
        super().__init__()

        self.ffn = FFN(hidden_dim)
        self.mha = MHA(hidden_dim, num_head)
        self.layernorm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, mask):
        atten_outpuut = self.mha(x,mask)
        atten_outpuut = self.layernorm1(x + atten_outpuut)

        output = self.layernorm2(atten_outpuut + self.ffn(atten_outpuut))

        return output
    
x = torch.rand(3, 4, 64)
net = Decoder(64, 8)
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)
    .unsqueeze(2)
    .repeat(1, 8, 4, 1)
)

print(net(x, mask).shape)


