来自VERL core_algos.py  700行

```
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```



好的，我来详细分解这段 `compute_policy_loss` 代码，尤其解释 **PPO 双重剪裁 (dual-clip)** 和输出指标的计算。我们按流程逐行分析。

---

### 1️⃣ 函数目的

这个函数计算 **策略网络的损失（Policy Loss）** ，用于 **PPO训练** 。它支持：

* 标准 PPO 的**clip ratio**
* 双重剪裁**dual-clip PPO**
* 对 token 级序列进行 mask 处理
* 输出一些统计指标（clip比例、近似 KL）

输入主要是：

* `old_log_prob`：旧策略对动作的 log 概率
* `log_prob`：新策略对动作的 log 概率
* `advantages`：优势函数估计
* `response_mask`：序列中哪些 token 参与计算
* `cliprange` /`cliprange_low` /`cliprange_high`：PPO 的裁剪参数
* `clip_ratio_c`：dual-clip 下界系数

---

### 2️⃣ Ratio 计算

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
```

* `negative_approx_kl = log_prob - old_log_prob` 是`log(pi_new/pi_old)`
* `ratio = exp(log(pi_new/pi_old)) = pi_new / pi_old`
* **为什么 clamp** ？防止概率差异过大导致梯度爆炸

**作用** ：这是 PPO 核心的 “策略比率”，用于裁剪和计算损失。

---

### 3️⃣ PPO KL 统计

```python
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

* 计算一个**近似 KL** ：`KL(pi_old || pi_new)`
* `masked_mean` → 只对有效 token 统计
* 输出用于监控训练，判断策略更新幅度是否过大

---

### 4️⃣ 标准 PPO 损失

```python
pg_losses1 = -advantages * ratio
```

* 这是未裁剪的策略梯度损失：
  [
  L^{PG} = - A_t \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
  ]
* **优势为正** → 惩罚比例大于1时增加损失，推动策略增强
* **优势为负** → 惩罚比例小于1时减少损失，抑制策略

---

### 5️⃣ 双重裁剪 (dual-clip PPO)

```python
pg_losses2 = -advantages * torch.clamp(
    ratio, 1 - cliprange_low, 1 + cliprange_high
)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

* `torch.clamp(ratio, 1-ε, 1+ε)` →**标准 PPO 裁剪**
* `clip_pg_losses1 = max(pg_losses1, pg_losses2)` →**dual-clip PPO** ，对正优势做裁剪
* 输出`pg_clipfrac` → 被裁剪的比例：

```python
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
```

---

### 6️⃣ lower clip（dual-clip 下界）

```python
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_clipfrac_lower = verl_F.masked_mean(
    torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
)
```

* 对**负优势** 做 dual-clip 下界处理
* `clip_ratio_c` 控制最大的梯度放大
* `torch.min(pg_losses3, clip_pg_losses1)` → 防止负优势的 loss 过大

---

### 7️⃣ 根据优势正负选择最终 loss

```python
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

* **优势为正** → 使用上界裁剪后的 loss
* **优势为负** → 使用下界裁剪后的 loss

---

### 8️⃣ 聚合成标量

```python
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

* 将 token 级别的损失聚合成一个标量，用于反向传播
* 支持不同聚合模式，比如`token-mean`、`seq-mean-token-sum` 等

---

### 9️⃣ 输出指标

```python
return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

* `pg_loss` → 最终策略损失，用于`optimizer.step()`
* `pg_clipfrac` → 被上界裁剪 token 的比例
* `ppo_kl` → 近似 KL，监控训练稳定性
* `pg_clipfrac_lower` → 被下界裁剪 token 的比例

---

### 🔹 总结

1. **ratio** ：策略更新比例，核心 PPO 指标
2. **pg_losses1 / pg_losses2** ：未裁剪 vs 裁剪
3. **dual-clip** ：同时考虑优势正负的裁剪，避免梯度过大或过小
4. **pg_losses3 / clip_pg_losses2** ：下界裁剪，控制负优势损失
5. **聚合 loss** ：mask + aggregation → 可对 token 或序列聚合
6. **输出额外指标** → 监控训练行为

---
