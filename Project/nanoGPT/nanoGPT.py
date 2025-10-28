# ruff: noqa: E731 F401 F841

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16
block_size = 32  # 推理最大上下文长度
max_iters = 5000
eval_iterval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200  # 推理时的迭代次数
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(42)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"字符数:{len(text)}")

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"文本单词：{chars}")
print(f"文本单词数目：{vocab_size}")

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda L: "".join([itos[c] for c in L])

print(f"encode: {encode('hello world')}")
print(f"encode: {decode(encode('hello world'))}")


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    # [0,len(data)-block_size-1]中抽取
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


# 模型评估
@torch.no_grad()
def eval_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logist, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class SingleHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x (b,t,h)
        b, t, _ = x.shape

        # (b,t,head_size)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # (b,t,t)
        atten = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)

        atten = atten.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        atten = self.dropout(F.softmax(atten, dim=-1))

        output = atten @ v

        return output


class MHA(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            SingleHeadAttention(head_size) for _ in range(num_head)
        )

        self.o_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.o_proj(out))
        return out


# model = MHA(n_head, n_embd // n_head)
# x = torch.randn(batch_size, block_size, n_embd)
# out = model(x)

# print("输入 x 的形状:", x.shape)
# print("输出 out 的形状:", out.shape)


class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.mha = MHA(n_head, head_size)
        self.ffn = FFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 这里是pre_norm
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


# model = Block(n_embd, n_embd // n_head)
# x = torch.randn(batch_size, block_size, n_embd)
# out = model(x)

# print("输入 x 的形状:", x.shape)
# print("输出 out 的形状:", out.shape)


class BingramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, target=None):
        b, t = x.shape

        # (b,t,c)
        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(torch.arange(t, device=device))

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln(x)

        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            b, t, _ = logits.shape
            logits = logits.view(b * t, -1)
            target = target.view(b * t)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 生成新的文本，从当前的索引idx开始，每次生成一个新的token，直到生成max_new_tokens个token为止
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 小于 block_size，会自动取到整个序列，
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

print(f"Before CUDA operation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
torch.zeros(1, device='cuda')  # 触发 CUDA 初始化
print(f"After CUDA operation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

model = BingramModel()
m = model.to(device=device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loss = []
val_loss = []

for iter in range(max_iters):
    if iter % eval_iterval == 0 or iter == max_iters - 1:
        losses = eval_loss()
        print(
            f"step {iter}: train loss is {losses['train']:.4f}, val loss is {losses['val']:.4f}"
        )
        train_loss.append(losses["train"])
        val_loss.append(losses["val"])

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()


# 绘制损失曲线图
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
# 保存图片
plt.savefig("loss_plot.png")


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
