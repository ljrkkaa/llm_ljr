import torch
import torch.nn as nn
import torch.optim as optim

# 模拟一个小网络
model = nn.Linear(2, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 假设我们要累积4步再更新一次
grad_accum_steps = 4

# 打印初始参数
print("Initial weight:", model.weight.data.clone())

for step in range(8):  # 模拟8个batch
    # 构造输入和目标
    x = torch.randn(2)
    y = torch.tensor([1.0])

    # 前向 + 计算loss
    pred = model(x)
    loss = (pred - y).pow(2).mean()

    # 缩放loss
    loss = loss / grad_accum_steps

    # 反向传播（此处不会清梯度）
    loss.backward()

    # 打印当前累计的梯度范数
    grad_norm = model.weight.grad.norm().item()
    print(f"Step {step+1:02d} | Grad norm = {grad_norm:.6f}")

    # 每累积4步更新一次
    if (step + 1) % grad_accum_steps == 0:
        print("→ Updating parameters!")
        optimizer.step()
        optimizer.zero_grad()
        print("  Weight after update:", model.weight.data.clone())
        print("  Grad after zero_grad:", model.weight.grad)
        print("-" * 60)
