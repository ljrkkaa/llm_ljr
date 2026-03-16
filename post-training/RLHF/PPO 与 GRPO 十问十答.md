https://zhuanlan.zhihu.com/p/2010833746958509207



> 随着 [DeepSeek R1](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=DeepSeek+R1&zhida_source=entity) 的火爆，GRPO 算法走入了大众视野。作为 RLHF 的两大支柱，PPO（近端策略优化）与 GRPO（组相对策略优化）在架构和逻辑上有何异同？

### 〇、PPO和GRPO的loss公式分别是什么

在 `[verl](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=verl&zhida_source=entity)` 框架及大模型强化学习的标准实现中，PPO 和 GRPO 的 Loss 公式既有重合之处，也有核心差异。以下是它们的详细公式拆解：

### 1\. PPO (Proximal Policy Optimization) Loss

PPO 的总 Loss 通常由三部分组成：**策略损失（Policy Loss）**、**价值损失（Value Loss）**和**熵正则项（Entropy Regularization）**。

LPPO\=LCLIP+c1LVF+c2LENTL\_{PPO} = L\_{CLIP} + c\_1 L\_{VF} + c\_2 L\_{ENT}L\_{PPO} = L\_{CLIP} + c\_1 L\_{VF} + c\_2 L\_{ENT}

*   **策略损失 (LCLIPL\_{CLIP}L\_{CLIP})**：采用剪切（Clip）机制的重要性采样目标。
*   LCLIP\=−Et\[min(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)\]L\_{CLIP} = - \\mathbb{E}\_t \\left\[ \\min \\left( r\_t(\\theta) \\hat{A}\_t, \\text{clip}(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}\_t \\right) \\right\]L\_{CLIP} = - \\mathbb{E}\_t \\left\[ \\min \\left( r\_t(\\theta) \\hat{A}\_t, \\text{clip}(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}\_t \\right) \\right\]  
    其中 rt(θ)\=πθ(at|st)πθold(at|st)r\_t(\\theta) = \\frac{\\pi\_{\\theta}(a\_t|s\_t)}{\\pi\_{\\theta\_{old}}(a\_t|s\_t)}r\_t(\\theta) = \\frac{\\pi\_{\\theta}(a\_t|s\_t)}{\\pi\_{\\theta\_{old}}(a\_t|s\_t)} 是新旧策略的概率比，A^t\\hat{A}\_t\\hat{A}\_t 是通过 [GAE](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=GAE&zhida_source=entity) 计算的优势函数。 _注：`verl` 还支持 **Dual-clip PPO**，在优势为负且下降过快时增加了一个下界保护。_
*   **价值损失 (LVFL\_{VF}L\_{VF})**：[Critic 网络](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=Critic+%E7%BD%91%E7%BB%9C&zhida_source=entity)的均方误差损失，通常也带 Clip。
*   LVF\=Et\[max((Vθ(st)−Rt)2,(clip(Vθ(st),Vold(st)−ϵ,Vold(st)+ϵ)−Rt)2)\]L\_{VF} = \\mathbb{E}\_t \\left\[ \\max \\left( (V\_{\\theta}(s\_t) - R\_t)^2, (\\text{clip}(V\_{\\theta}(s\_t), V\_{old}(s\_t)-\\epsilon, V\_{old}(s\_t)+\\epsilon) - R\_t)^2 \\right) \\right\]L\_{VF} = \\mathbb{E}\_t \\left\[ \\max \\left( (V\_{\\theta}(s\_t) - R\_t)^2, (\\text{clip}(V\_{\\theta}(s\_t), V\_{old}(s\_t)-\\epsilon, V\_{old}(s\_t)+\\epsilon) - R\_t)^2 \\right) \\right\]  
    
*   **熵正则项 (LENTL\_{ENT}L\_{ENT})**：鼓励模型保持探索。
*   LENT\=−Et\[H(πθ(⋅|st))\]L\_{ENT} = - \\mathbb{E}\_t \[ H(\\pi\_{\\theta}(\\cdot|s\_t)) \]L\_{ENT} = - \\mathbb{E}\_t \[ H(\\pi\_{\\theta}(\\cdot|s\_t)) \]  
    

### 2\. GRPO (Group Relative Policy Optimization) Loss

GRPO 的核心在于**取消了 Critic 网络**，因此没有价值损失 LVFL\_{VF}L\_{VF}。它的总 Loss 由策略损失和直接的 [KL 散度](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=KL+%E6%95%A3%E5%BA%A6&zhida_source=entity)约束组成。

LGRPO\=LCLIP+βLKL+c2LENTL\_{GRPO} = L\_{CLIP} + \\beta L\_{KL} + c\_2 L\_{ENT}L\_{GRPO} = L\_{CLIP} + \\beta L\_{KL} + c\_2 L\_{ENT}

*   **策略损失 (LCLIPL\_{CLIP}L\_{CLIP})**：公式形式与 PPO 相同，但**优势函数 A^t\\hat{A}\_t\\hat{A}\_t 的计算方式发生了本质变化**。 GRPO 针对每个 Prompt 采样一组（Group）回复 {O1,O2,…,OG}\\{O\_1, O\_2, \\dots, O\_G\\}\\{O\_1, O\_2, \\dots, O\_G\\}，其优势计算为组内得分的归一化：  
    A^i\=ri−mean(r1,…,rG)std(r1,…,rG)\\hat{A}\_i = \\frac{r\_i - \\text{mean}(r\_1, \\dots, r\_G)}{\\text{std}(r\_1, \\dots, r\_G)}\\hat{A}\_i = \\frac{r\_i - \\text{mean}(r\_1, \\dots, r\_G)}{\\text{std}(r\_1, \\dots, r\_G)}  
    这里的 rir\_ir\_i 是第 iii 个回复的奖励得分。
*   **KL 损失 (LKLL\_{KL}L\_{KL})**：在 `verl` 的 GRPO 实现中，KL 散度通常不作为奖励惩罚项，而是直接作为 Loss 的一部分。
*   LKL\=E\[KL(πθ||πref)\]L\_{KL} = \\mathbb{E} \[ KL(\\pi\_{\\theta} || \\pi\_{ref}) \]L\_{KL} = \\mathbb{E} \[ KL(\\pi\_{\\theta} || \\pi\_{ref}) \]  
    常用的估计器包括 [Schulman 估计器](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=Schulman+%E4%BC%B0%E8%AE%A1%E5%99%A8&zhida_source=entity)（k\=r−log⁡r−1k = r - \\log r - 1k = r - \\log r - 1）等，确保 KL 约束始终为正。

* * *

### 核心对比总结

| 特性 | PPO | GRPO |
| --- | --- | --- |
| 优势函数 \\hat{A}\_t | 依赖 Critic 预测，通过 GAE 计算 | 依赖组内对比，通过得分归一化计算 |
| 价值损失 L\_{VF} | 有（需要训练 Critic 网络） | 无（节省约 50% 显存） |
| KL 约束方式 | 通常作为奖励惩罚项（Reward Penalty） | 通常作为直接 Loss 项（KL Loss） |
| 适用场景 | 通用强化学习任务 | 奖励明确且易于采样的任务（如数学、代码） |

### 一、 训练的主流程是如何运转的？

在 `verl` 等主流框架中，一个典型的 PPO/GRPO 迭代周期包含以下四个关键阶段，形成一个闭环：

1.  **Rollout（采样）**：Actor 模型（策略模型）接收 Prompt，利用推理引擎（如 [vLLM](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=vLLM&zhida_source=entity)）生成回复序列。在 GRPO 中，通常会对同一个 Prompt 生成一组（如 8 个或 16 个）不同的回复，以便进行组内对比。
2.  **Evaluation（评估）**：  
    

*   **计算 `old_log_prob`**：使用训练引擎（如 [FSDP](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=FSDP&zhida_source=entity)）计算采样时刻模型对生成序列的对数概率，作为后续重要性采样的基准。
*   **计算 `ref_log_prob`**：参考模型（通常是冻结的 [SFT 模型](https://zhida.zhihu.com/search?content_id=270744054&content_type=Article&match_order=1&q=SFT+%E6%A8%A1%E5%9E%8B&zhida_source=entity)）对序列的对数概率，用于计算 KL 散度，约束模型不要偏离原始分布太远。
*   **计算 `values`**：Critic 模型预测每个 Token 状态的预期回报（仅 PPO 需要，用于降低方差）。
*   **计算 `rewards`**：由 Reward 模型（模型打分）或规则脚本（如编译器检查、数学公式匹配）给出最终得分。

  

1.  **Advantage（优势计算）**：这是强化学习的核心。PPO 使用 GAE（广义优势估计）结合 Critic 的预测值来计算每个动作的优势；而 GRPO 则通过组内得分的归一化（r−meanstd\\frac{r - \\text{mean}}{\\text{std}}\\frac{r - \\text{mean}}{\\text{std}}）直接得出优势。
2.  **Update（更新）**：利用优势函数计算策略梯度，更新 Actor 参数。PPO 还会同步更新 Critic 模型以使其预测更准。

### 二、 为什么不能在 Rollout 阶段直接得到 Log Prob？

虽然在生成时模型已经计算了概率，但架构上通常选择解耦，原因有三：

*   **引擎能力差异**：Rollout 追求极致的吞吐量，通常使用 vLLM 等推理引擎，其算子（如 PagedAttention）针对推理优化；而计算 Log Prob 需要训练引擎（如 PyTorch FSDP/Megatron），它需要保留梯度信息并支持复杂的分布式并行策略。
*   **数值精度与一致性**：推理引擎为了速度常使用 FP16/BF16 甚至量化，而训练需要更高的精度。如果在推理引擎中计算 Log Prob，其算子实现与训练引擎的微小差异（如 FlashAttention 的版本差异）会导致训练时的 `ratio` 在初始阶段不为 1，引发训练不稳定。
*   **计算模式不同**：生成是自回归的（逐 Token），而计算 Log Prob 是全量批处理（Parallel Forward），后者能更充分地利用 GPU 的并行算力。

### 三、 奖励只在最后一个 Token 产生，中间位置怎么办？

在数学竞赛或代码生成任务中，奖励通常是稀疏的（Outcome Reward）。模型通过以下机制将“远期奖励”分配给“当前动作”：

1.  **KL 惩罚密集化**：最终奖励 rt\=Scoret−β⋅KLtr\_t = \\text{Score}\_t - \\beta \\cdot KL\_tr\_t = \\text{Score}\_t - \\beta \\cdot KL\_t。由于 KL 惩罚是逐 Token 计算的，这使得原本只有末尾有值的奖励序列变成了**每个位置都有值的密集奖励**。
2.  **价值回传（Credit Assignment）**：

*   **在 PPO 中**：Critic 模型学习预测“从当前位置到结束的预期总分”。通过 TD-Error，末尾的奖励会像波浪一样向上传递给每一个中间 Token。
*   **在 GRPO 中**：虽然优势函数通常是序列级的（整个回复共享一个优势值），但梯度更新依然是逐 Token 的。高分回复中的每一个 Token 都会受到正向激励。

  

### 四、 逐 Token 的 KL 惩罚是如何计算的？

KL 散度衡量了当前策略 πθ\\pi\_{\\theta}\\pi\_{\\theta} 与参考策略 πref\\pi\_{\\text{ref}}\\pi\_{\\text{ref}} 的差异。最常用的估计器是：

KLt\=log⁡πθ(at|st)−log⁡πref(at|st)KL\_t = \\log \\pi\_{\\theta}(a\_t|s\_t) - \\log \\pi\_{\\text{ref}}(a\_t|s\_t)KL\_t = \\log \\pi\_{\\theta}(a\_t|s\_t) - \\log \\pi\_{\\text{ref}}(a\_t|s\_t)

在 `verl` 等框架中，为了进一步降低方差，还提供了多种 KL 估计器（如 k1, k2, k3）。例如，GRPO 有时会选择直接将 KL 散度作为 Loss 的一部分（`use_kl_loss`），而不是作为奖励惩罚项。这种做法在处理长推理链（Long CoT）时能提供更稳定的梯度，防止模型在探索过程中因 KL 惩罚过大而变得过于保守。

### 五、 为什么 Schulman KL 估计器是非负的？

为了防止训练崩溃，常用 Schulman 估计器：k\=r−log⁡r−1k=r-\\log r-1k=r-\\log r-1（其中 r\=πθ/πrefr = \\pi\_{\\theta}/\\pi\_{\\text{ref}}r = \\pi\_{\\theta}/\\pi\_{\\text{ref}} ）。

从数学上看，函数 f(r)\=r−log⁡r−1f(r) = r - \\log r - 1f(r) = r - \\log r - 1 在 r\=1r=1r=1 时取得最小值 0。由于其二阶导数 f″(r)\=1/r2\>0f''(r) = 1/r^2 > 0f''(r) = 1/r^2 > 0，该函数是严格凸的。这意味着**无论当前模型概率是比参考模型大还是小，惩罚项永远 ≥0\\ge 0\\ge 0**。这消除了“负 KL”现象（即模型因为偏离参考模型反而获得奖励），确保了 KL 始终起到约束作用。

### 六、 GAE 是如何推导优势函数的？

GAE（Generalized Advantage Estimation）通过平衡偏差和方差来优化 PPO 的学习：

1.  **TD-Error (δt\\delta\_t\\delta\_t)**: δt\=rt+γV(st+1)−V(st)\\delta\_t = r\_t + \\gamma V(s\_{t+1}) - V(s\_t)\\delta\_t = r\_t + \\gamma V(s\_{t+1}) - V(s\_t)。它表示“这一步动作带来的实际收益比 Critic 预期的好多少”。
2.  **指数加权移动平均**: At\=∑l\=0∞(γλ)lδt+lA\_t = \\sum\_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta\_{t+l}A\_t = \\sum\_{l=0}^{\\infty} (\\gamma \\lambda)^l \\delta\_{t+l}。

*   当 λ\=0\\lambda=0\\lambda=0 时，GAE 退化为 1-step TD，方差小但偏差大。
*   当 λ\=1\\lambda=1\\lambda=1 时，GAE 变成 Monte Carlo 采样，无偏但方差极大。 通过调节 λ\\lambda\\lambda（通常取 0.95），PPO 能够在长序列中精准捕捉到关键动作的贡献。

  

### 七、 GRPO 与 PPO 的本质区别是什么？

最核心的区别在于 **Critic 模型的去留与优势计算方式**：

*   **PPO (Actor-Critic)**：需要一个额外的 Critic 网络来估计状态价值。这增加了显存开销（通常 Actor 和 Critic 大小相当），且 Critic 的训练本身也存在不稳定性。
*   **GRPO (Group-based)**：**彻底取消了 Critic 模型**。它通过对同一个 Prompt 采样一组回复，利用组内的相对得分来代替绝对价值估计。

*   **显存优势**：节省约 50% 的模型参数显存，让单机能够训练更大规模的模型或更长的上下文。
*   **逻辑优势**：在 Reasoning 任务中，结果的对错是明确的，组内对比比训练一个不稳定的 Critic 来预测得分更直接有效。

  

### 八、 GRPO 能得到每个 Token 的 Loss 吗？

**能，且必须能。** 这是一个常见的误区。虽然 GRPO 的优势函数（Advantage）通常是针对整个序列计算的（即组内比较得出回复 A 比回复 B 好），但其 Loss 函数依然作用于逐个 Token：

Loss = - \\frac{1}{G} \\sum\_{i=1}^{G} \\sum\_{t=1}^{T} \\text{ratio}\_{i,t} \\cdot A\_iLoss = - \\frac{1}{G} \\sum\_{i=1}^{G} \\sum\_{t=1}^{T} \\text{ratio}\_{i,t} \\cdot A\_i

其中 \\text{ratio}\_{i,t}\\text{ratio}\_{i,t} 是第 ii 个回复在第 tt 个 Token 处的概率比率。这意味着，如果整个回复拿了高分（A\_i > 0A\_i > 0），Loss 会促使模型提高该回复中**每一个 Token** 的出现概率。这种机制在海量数据下，通过统计平均能够自动识别出哪些 Token 是导致高分的关键。

### 九、 计算 Loss 时的“Old 策略”到底是谁？

“Old 策略”并不是一个物理存在的模型，而是**采样时刻的模型快照**。

在强化学习的一个迭代周期内，我们会对同一批采样数据进行多次训练（Epochs）。为了保证数学上的正确性，我们需要使用**重要性采样（Importance Sampling）**：

*   **分母**：采样时记录的 `old_log_prob`。
*   **分子**：当前正在更新的模型计算出的 `new_log_prob`。

通过 \\frac{\\pi\_{new}}{\\pi\_{old}}\\frac{\\pi\_{new}}{\\pi\_{old}}，我们能够利用“旧数据”来更新“新模型”，而不需要每更新一次参数就重新采样（那将极其缓慢）。

### 十、 Loss 公式中的 Clip 和 Min 怎么理解？

这是 PPO/GRPO 的“安全气囊”，旨在实现**信任区域（Trust Region）**优化：

*   **Clip**：限制 \\text{ratio}\\text{ratio} 的范围（如 \[0.8, 1.2\]\[0.8, 1.2\]）。如果模型想把某个好动作的概率提高 10 倍，Clip 会强行将其限制在 1.2 倍，防止策略突变。
*   **Min**：实现“悲观下界”。

*   **当优势为正时**：限制奖励的上限，防止模型过度自信。
*   **当优势为负时**：如果模型已经大幅降低了该动作的概率，`min` 会确保惩罚不会因为概率降得太快而失效。

  

这套机制确保了模型在微调过程中始终处于“稳健步进”状态，避免了强化学习中常见的“策略崩溃”风险。

### 十一、PPO的价值损失中的max怎么理解

理解 PPO 价值损失中的 `max` 操作，关键在于明白它是如何实现**“限制更新幅度”**这一目标的。

我们可以通过一个具体的数值例子来拆解。

### 场景设定

*   **旧的预测值 (`values`)**：10.0（这是模型在采样阶段给出的预测）
*   **真实的回报 (`returns`)**：15.0（这是我们希望模型学习的目标）
*   **剪切范围 (`cliprange_value`)**：1.0（意味着我们允许新预测值在 \[9.0, 11.0\] 之间波动）

* * *

### 情况 A：模型更新“太快”了（超出了允许范围）

假设模型经过一次更新，现在的预测值 **`vpreds` = 12.0**。

1.  **计算原始损失 (`vf_losses1`)**： (12.0 - 15.0)^2 = (-3.0)^2 = \\mathbf{9.0}(12.0 - 15.0)^2 = (-3.0)^2 = \\mathbf{9.0}
2.  **计算剪切后的预测值 (`vpredclipped`)**： 12.0 超出了 \[9.0, 11.0\]，所以被强行剪切到边界 **11.0**。
3.  **计算剪切后的损失 (`vf_losses2`)**： (11.0 - 15.0)^2 = (-4.0)^2 = \\mathbf{16.0}(11.0 - 15.0)^2 = (-4.0)^2 = \\mathbf{16.0}
4.  **取 `max(9.0, 16.0)`**： 结果是 **16.0**。

**为什么选大的（16.0）？**

*   如果选小的（9.0），模型会觉得“哇，我预测到 12.0 误差更小了，我要继续往那边更新”。
*   通过选大的（16.0），算法实际上是在告诉模型：“虽然你预测到 12.0 看起来误差小了，但我只承认你更新到 11.0 时的那个大误差。而且因为 11.0 是个常数边界，你的梯度会变小甚至消失，从而**强行停止**你往 12.0 继续跑。”

* * *

### 情况 B：模型更新方向“反了”（误差变大了）

假设模型更新后，预测值 **`vpreds` = 8.0**（离目标 15.0 更远了）。

1.  **计算原始损失 (`vf_losses1`)**： (8.0 - 15.0)^2 = (-7.0)^2 = \\mathbf{49.0}(8.0 - 15.0)^2 = (-7.0)^2 = \\mathbf{49.0}
2.  **计算剪切后的预测值 (`vpredclipped`)**： 8.0 被剪切到下边界 **9.0**。
3.  **计算剪切后的损失 (`vf_losses2`)**： (9.0 - 15.0)^2 = (-6.0)^2 = \\mathbf{36.0}(9.0 - 15.0)^2 = (-6.0)^2 = \\mathbf{36.0}
4.  **取 `max(49.0, 36.0)`**： 结果是 **49.0**。

**为什么选大的（49.0）？**

*   当模型跑偏时，我们不希望“剪切”保护它。我们希望保留那个**最大的原始误差**，给模型一个狠狠的惩罚，把它拉回到正确的方向上。

* * *

### 总结：`max` 的逻辑图

| 预测值状态 | 原始损失 vs 剪切损失 | max 的结果 | 实际效果 |
| --- | --- | --- | --- |
| 在范围内 | 两者相等 | 正常损失 | 正常更新 |
| 超标且变好 | 原始损失 < 剪切损失 | 选剪切损失 | 限制更新，不让你跑太快 |
| 超标且变坏 | 原始损失 > 剪切损失 | 选原始损失 | 保持惩罚，把你拉回来 |

`max` 确保了模型在“变好”时受到限制（不准贪功冒进），在“变坏”时受到全额惩罚（必须改邪归正）。

* * *

### 十二、为什么DeepSeek R1的GRPO的clip ratio能给到10，而PPO一般是0.2

DeepSeek-R1中GRPO算法将clip ratio设置为10，而传统PPO常用0.2，核心原因是**算法设计差异、训练目标不同以及配套约束机制的互补**，具体可从以下三方面详细解析：

### 一、算法本质差异：GRPO与传统PPO的clip逻辑不同

传统PPO的clip ratio（通常取0.1~0.2）核心目的是**限制策略更新幅度**，避免因单次更新过大导致训练不稳定。其核心逻辑是：通过将策略比率\\frac{\\pi\_{\\theta}(a|s)}{\\pi\_{\\theta\_{old}}(a|s)}\\frac{\\pi\_{\\theta}(a|s)}{\\pi\_{\\theta\_{old}}(a|s)}裁剪在\[1-\\varepsilon, 1+\\varepsilon\]\[1-\\varepsilon, 1+\\varepsilon\]范围内，防止参数更新时偏离原始策略过远，进而避免价值函数拟合误差引发的训练震荡。但这一设计会牺牲策略探索的灵活性，尤其不利于需要复杂推理路径的任务。

而DeepSeek-R1采用的**GRPO（Group Relative Policy Optimization）** 算法，对clip机制的依赖显著降低：

1.  **优势计算方式不同**：GRPO不依赖价值模型，而是通过“组内相对优势”计算（A\_i=\\frac{r\_i-mean(\\{r\_1,...,r\_G\\})}{std(\\{r\_1,...,r\_G\\})}A\_i=\\frac{r\_i-mean(\\{r\_1,...,r\_G\\})}{std(\\{r\_1,...,r\_G\\})}），优势值已被标准化，天然降低了极端值对更新的影响，无需通过小clip ratio限制幅度；
2.  **KL约束独立且更强**：GRPO直接在损失函数中加入KL散度项（\-\\beta \\mathbb{D}\_{KL}(\\pi\_{\\theta}\\|\\pi\_{ref})\-\\beta \\mathbb{D}\_{KL}(\\pi\_{\\theta}\\|\\pi\_{ref})），通过定期更新参考模型\\pi\_{ref}\\pi\_{ref}（每400步），稳定策略更新方向。这种“独立KL约束”替代了部分clip的稳定作用，允许clip ratio放大；
3.  **组采样降低方差**：GRPO对每个问题采样多组输出（如16个），通过组内竞争筛选最优策略，样本多样性降低了单一极端更新的风险，为更大的clip ratio提供了安全边界。

### 二、训练目标适配：长链推理需要更大的策略探索空间

传统PPO常用于对齐类任务（如对话生成、指令跟随），目标是在人类偏好范围内优化输出，无需大幅探索新策略，小clip ratio可保证输出一致性。而DeepSeek-R1的核心目标是**激发模型的长链推理能力**，需要模型自主探索非人类标注的推理路径（如自我反思、多路径验证），具体需求如下：

1.  **推理路径多样性需求**：数学、编码等复杂任务需要模型尝试不同推理逻辑，过大的clip限制会导致模型陷入局部最优（如重复单一推理模式），而clip ratio=10几乎不限制策略比率的更新幅度，允许模型探索更长、更灵活的推理链；
2.  **响应长度动态扩展**：训练中模型需要从短响应（32k tokens）逐步扩展到长响应（65k tokens），小clip ratio会隐含惩罚长序列生成（传统PPO的per-token KL penalty会累积惩罚长输出），而GRPO的KL约束不依赖token级惩罚，配合大clip ratio可支持推理长度的自然增长；
3.  **无SFT初始化的探索需求**：DeepSeek-R1-Zero跳过了SFT阶段，直接基于基座模型进行RL训练，需要更大的策略更新自由度来突破原始模型的能力边界，大clip ratio能加速推理模式的涌现（如训练中出现的“自我反思”“aha moment”等行为）。

### 三、配套机制互补：避免大clip ratio导致的训练失控

GRPO设置clip ratio=10并未引发训练不稳定，关键在于其配套约束机制形成了互补：

1.  **语言一致性奖励兜底**：训练中引入语言一致性奖励（Reward\_{language}=\\frac{Num(Words\_{target})}{Num(Words)}Reward\_{language}=\\frac{Num(Words\_{target})}{Num(Words)}），约束模型输出的语言统一性，避免因策略过度探索导致语言混合（如中英混杂）等问题；
2.  **规则化奖励信号**：采用规则化奖励（准确率奖励+格式奖励），而非模糊的人类偏好奖励，奖励信号的可靠性降低了“策略投机”风险，即使clip ratio较大，模型也会聚焦于“正确推理”而非钻奖励漏洞；
3.  **小批量、单轮内迭代**：每个rollout生成的8192个输出被随机拆分为16个mini-batch，仅训练1个内迭代 epoch，减少了参数更新的累积误差，避免大clip ratio下的误差放大。

### 总结：核心逻辑对比

| 维度 | 传统PPO（clip=0.2） | GRPO（clip=10） |
| --- | --- | --- |
| 核心目标 | 稳定对齐人类偏好，限制策略偏离 | 激发长链推理，探索最优推理路径 |
| 优势计算 | 依赖价值模型（GAE），易受误差影响 | 组内相对优势，标准化后更稳健 |
| 稳定机制 | 依赖clip限制更新幅度 | 依赖KL约束+组采样+定期更新参考模型 |
| 适用场景 | 短输出、对齐类任务（对话、指令跟随） | 长输出、推理类任务（数学、编码、STEM） |

简言之，传统PPO的小clip ratio是“以限制换稳定”，而GRPO通过算法优化和配套机制，实现了“大clip换探索+独立约束保稳定”的平衡，最终适配了长链推理任务对策略灵活性的需求。

### 结语

PPO 像是一位严厉的导师（Critic），时刻盯着学生的每一步并给出反馈；而 GRPO 则像是一场选拔赛，让学生们（Group）同台竞技，通过优胜劣汰来自我进化。理解了这十个问题，你就掌握了现代大模型强化学习从 PPO 到 GRPO 演进的核心脉络。

本文转自 <https://zhuanlan.zhihu.com/p/2010833746958509207>，如有侵权，请联系删除。