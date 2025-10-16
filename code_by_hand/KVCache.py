import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA_kv_Cache(nn.Module):
    def __init__(self, hidden_dim, num_head, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # KV 缓存初始化
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, mask=None, use_cache=False):
        # x: (b, seq_len, hidden_dim)
        b, s, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (b, num_head, seq_len, head_dim)
        Q = Q.view(b, s, self.num_head, self.head_dim).transpose(1, 2)
        K = K.view(b, s, self.num_head, self.head_dim).transpose(1, 2)
        V = V.view(b, s, self.num_head, self.head_dim).transpose(1, 2)

        # ⚡ 使用 KV 缓存
        if use_cache and self.cache_k is not None:
            K = torch.cat([self.cache_k, K], dim=-2)
            V = torch.cat([self.cache_v, V], dim=-2)

        if use_cache:
            # 更新缓存
            self.cache_k = K
            self.cache_v = V

        # 注意力计算
        atten = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            atten = atten.masked_fill(mask == 0, float("-inf"))

        atten = self.dropout(F.softmax(atten, dim=-1))
        output = atten @ V

        # 还原形状
        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.o_proj(output)
        return output


# 🔧 测试
x1 = torch.rand(1, 1, 128)
x2 = torch.rand(1, 1, 128)
net = MHA_kv_Cache(128, 8)

# 第一次前向传播（建立缓存）
out1 = net(x1, use_cache=True)
print("out1:", out1.shape, "| cache_k:", net.cache_k.shape)

# 第二次前向传播（复用缓存）
out2 = net(x2, use_cache=True)
print("out2:", out2.shape, "| cache_k:", net.cache_k.shape)



class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_head):
        super().__init__()
        self.attn = MHA_kv_Cache(hidden_dim, num_head)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.norm1(x), use_cache=use_cache)
        x = x + self.ff(self.norm2(x))
        return x


# 🧩 模拟一个简单的推理过程
torch.manual_seed(42)
hidden_dim = 32
num_heads = 4
block = TransformerBlock(hidden_dim, num_heads)

# 模拟序列逐步输入（比如一个句子逐token生成）
for step in range(5):
    new_token = torch.rand(1, 1, hidden_dim)  # 当前步的新token向量
    out = block(new_token, use_cache=True)     # 增量推理
    print(f"Step {step+1}: 输出形状 {out.shape}, KV缓存长度 {block.attn.cache_k.shape[-2]}")