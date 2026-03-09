参考资料：
https://zhuanlan.zhihu.com/p/32368626065



伪代码

字节（即将）开源一套完整的 LLM + RL工业级解决方案：

*   提出了DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）算法，解决CoT场景下LLM强化学习的训练稳定性与效率问题。
*   实验表明，基于Qwen2.5-32B基础模型，该系统在[AIME 2024](https://zhida.zhihu.com/search?content_id=255496116&content_type=Article&match_order=1&q=AIME+2024&zhida_source=entity)数学竞赛中取得50分的成绩（超越此前最优结果[DeepSeek-R1](https://zhida.zhihu.com/search?content_id=255496116&content_type=Article&match_order=1&q=DeepSeek-R1&zhida_source=entity)的47分），且训练步数减少50%。
*   宣布开源算法、训练框架（基于verl）及17K规模的DAPO-Math数据集。

核心机制：

1.  **Clip-Higher策略**：通过解耦策略更新的上下限，缓解熵崩溃问题
2.  **动态采样机制**：过滤无效样本，提升梯度信号质量
3.  **Token级策略梯度损失**：优化长序列训练稳定性
4.  **超长奖励重塑**：降低截断样本的奖励噪音

链接：

*   [\[2503.14476\] DAPO: An Open-Source LLM Reinforcement Learning System at Scale](http://arxiv.org/abs/2503.14476)
*   [https://github.com/BytedTsinghua-SIA/DAPO](http://github.com/BytedTsinghua-SIA/DAPO)

* * *

主要内容
----

### **1\. 作者和团队信息**

*   这篇文章由来自 **ByteDance Seed**、**清华大学人工智能产业研究院 (AIR)** 以及 **香港大学** 的研究者共同完成。
*   主要贡献者包括 Qiying Yu, Zheng Zhang, Ruofei Zhu 等人。

### **2\. 背景和动机**

*   **发表时间：** 2025年3月（arxiv预印版）
*   **研究问题：** 如何高效、稳定地训练大规模语言模型的推理能力，尤其是在长链式思考 (long Chain-of-Thought, long-CoT) 的场景下。
*   **问题背景**：

*   **测试时扩展（Test-time Scaling）**：如OpenAI的o1、DeepSeek的R1通过强化学习激发模型的复杂推理能力，但关键技术细节未公开
*   **工业级挑战**：直接应用传统PPO/GRPO算法会导致熵崩溃（Entropy Collapse）、奖励噪音等问题，导致训练失败

*   **核心痛点**：

*   **熵崩溃**：策略快速收敛到确定性输出，丧失探索能力
*   **梯度消失**：当整组样本奖励相同时，优势函数归零导致无效更新
*   **序列失衡**：长序列token在样本级损失计算中被稀释影响
*   **截断干扰**：超长样本的惩罚奖励引入噪音

**背景知识补充**：

*   **AIME (American Invitational Mathematics Examination)**：美国数学邀请赛，一项难度较高的数学竞赛，常被用于评估 LLM 的数学推理能力。
*   **Entropy Collapse（熵坍塌）**：在强化学习训练中，模型输出的概率分布变得过于集中，导致探索能力下降，多样性降低的现象。可以理解为模型变得“懒惰”，只倾向于选择少数几个高概率的动作，而忽略了其他可能的动作。如下图中的蓝色线。

![](https://pic4.zhimg.com/v2-19ae468a2c9ec159a015d9422c9a50b1_r.jpg)

### 3\. 相关研究

| 方法 | 优势 | 缺陷 |
| --- | --- | --- |
| PPO-Clip | 经典稳定，信任域控制 | 上限裁剪限制探索性 |
| GRPO | 无需价值函数，组内相对奖励 | 样本级损失导致长序列失衡 |
| DeepSeek-R1 | 首个展示数学推理潜力的方法 | 未公开关键训练细节 |

传统方法在长CoT场景下普遍存在训练不稳定问题，DAPO通过系统性改进实现突破。

### **4\. 核心思路**

本文的核心思路是：

*   **揭示大规模 RL 训练中的关键问题**，例如熵坍塌、奖励噪声和训练不稳定等。
*   **提出 Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) 算法**，该算法包含四个关键技术：

*   Clip-Higher
*   Dynamic Sampling
*   Token-Level Policy Gradient Loss
*   [Overlong Reward Shaping](https://zhida.zhihu.com/search?content_id=255496116&content_type=Article&match_order=1&q=Overlong+Reward+Shaping&zhida_source=entity)

*   **完全开源 RL 系统**，包括算法、代码和数据集，以促进可复现的研究。

DAPO 算法的灵感可能来自于对传统 PPO 和 GRPO 算法的改进，以及对大规模 LLM RL 训练中特有问题（如长 CoT 带来的挑战）的深入分析。

### **5\. 方案与技术**

DAPO 算法主要包含以下四个关键技术：

1.  **Clip-Higher**：解耦 PPO 算法中的 clipping 上下限，增大上限，鼓励模型探索更多样化的输出，避免熵坍塌。
2.  **Dynamic Sampling**：在训练过程中动态调整采样策略，过滤掉那些所有输出都正确或都错误的样本，以保证每个 batch 中的样本都能提供有效的梯度信号。
3.  **Token-Level Policy Gradient Loss**：GRPO算法存在样本级损失使长序列token影响被稀释的问题，文本将损失函数计算的粒度从 sample 级别降低到 token 级别，解决长 CoT 场景下长序列对梯度贡献过小的问题。
4.  **Overlong Reward Shaping**：针对生成长度超过限制的样本，提出一种基于长度的惩罚机制，避免因过度惩罚导致奖励噪声。

这些技术方案主要针对大规模 LLM 在长链推理场景下 RL 训练时遇到的特殊问题，例如探索不足、梯度消失、奖励噪声等。

### **6\. 实验与结论**

![](https://picx.zhimg.com/v2-93f8f27b263838e58f16237c1166b5d9_r.jpg)

*   **实验设计**：

*   使用 Qwen2.5-32B 作为预训练模型进行 RL 训练。
*   在 AIME 2024 数据集上评估模型性能。
*   将 DAPO 算法与 naive GRPO 算法进行比较，并逐步增加 DAPO 的各项关键技术，以分析它们各自的贡献。
*   通过监控训练过程中的关键指标（如奖励、熵、生成长度等）来分析训练动态。

*   **实验结果**：

*   DAPO 算法在 AIME 2024 上取得了 50 分的成绩，超过了 DeepSeek-R1-Zero-Qwen-32B 的 47 分。
*   DAPO 算法仅使用 DeepSeek-R1-Zero-Qwen-32B 一半的训练步骤就达到了更好的性能。
*   各项关键技术都对性能提升有贡献，其中 Dynamic Sampling 的贡献最为显著。

*   **发现与结论**：

*   DAPO 算法能够有效提升 LLM 在长 CoT 场景下的推理能力。
*   Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss 和 Overlong Reward Shaping 等技术对于大规模 LLM RL 训练至关重要。
*   在 RL 训练过程中，LLM 的推理模式会动态演化，出现新的推理方式

### **7\. 贡献**

*   提出了 DAPO 算法，一种用于大规模 LLM RL 训练的有效算法。
*   开源了一个完整的 RL 系统，包括算法、代码和数据集，是首个开源的大规模RL-CoT训练系统
*   揭示了大规模 LLM RL 训练中的一些关键问题，比如策略熵控制在长序列推理中的核心作用
*   提出动态采样理论框架
*   复现DeepSeek-R1结果所需成本降低60%

### **8\. 不足**

1.  **数据集的局限性**：虽然文章提到了对数据集进行了转换，使其更易于解析，但也可能因此引入了偏差，限制了模型的泛化能力。
2.  **对其他任务的泛化性**：文章主要关注数学任务，DAPO 算法在其他类型的推理任务上的表现还有待验证。依赖规则奖励（如AIME整数答案），迁移至开放域需调整。
3.  动态采样增加20%的生成开销
4.  对16k以上长上下文支持仍需优化
5.  **Overfitting的风险**：文章提到在训练集上reward很高，但在验证集上效果不明显，这意味着存在过拟合的风险。

* * *

QA
--

### Q1：为什么 naive GRPO 在 AIME 上只能达到 30 分？

Naive GRPO 可能存在以下问题：

*   **探索不足**：容易陷入局部最优解，无法发现更优的推理路径。
*   **奖励噪声**：不准确的奖励信号会误导模型的训练。
*   **训练不稳定**：RL 训练过程本身就比较复杂，容易出现各种问题。
*   **长CoT场景下的挑战**：在长 CoT 场景下，梯度消失问题会更加严重，导致模型难以学习。

### Q2：传统 PPO 存在什么问题？

传统的PPO算法引入了**裁剪 (Clipping)** 机制。裁剪机制通过限制新策略与旧策略的比率，防止策略更新过大。具体来说，PPO算法会计算新策略与旧策略的比率，并将这个比率裁剪到一个预定义的范围内。这样可以保证策略更新的幅度不会太大，从而提高训练的稳定性。

PPO的目标函数可以表示为：

LCLIP(θ)\=E^t\[min(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)\]L^{CLIP}(\\theta) = \\hat{E}\_t\[min(r\_t(\\theta)\\hat{A}\_t, clip(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}\_t)\]L^{CLIP}(\\theta) = \\hat{E}\_t\[min(r\_t(\\theta)\\hat{A}\_t, clip(r\_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}\_t)\]

其中，rt(θ)r\_t(\\theta)r\_t(\\theta)表示重要性采样率，A^t\\hat{A}\_t\\hat{A}\_t表示优势函数，ϵ\\epsilon\\epsilon表示裁剪范围。

但问题是，裁剪范围是**对称**的，这意味着无论新策略的概率是高于还是低于旧策略，都会受到相同的限制。这种对称裁剪可能会对低概率token和高概率token产生不同的影响。

*   **对低概率token的影响**：由于裁剪范围的限制，低概率token的概率提升空间较小。例如，如果一个token的原始概率为0.01，裁剪范围为ϵ\=0.2\\epsilon=0.2\\epsilon=0.2，那么更新后的概率最大只能达到0.012，仅提升了20%。这意味着模型很难有机会去探索这些低概率token，从而可能导致**探索不足**。
*   **对高概率token的影响**：相比之下，高概率token的概率降低空间较大。例如，如果一个token的原始概率为0.9，裁剪范围为ϵ\=0.2\\epsilon=0.2\\epsilon=0.2，那么更新后的概率可以降低到0.72，降低幅度较大。虽然裁剪可以防止策略更新过大，但也可能限制模型对这些高概率token的利用，从而影响**收敛速度**。

### Q3: Clip-Higher 如何解决熵坍塌问题？

**Clip-Higher策略** 通过解耦上下限裁剪，允许对低概率token和高概率token进行**非对称**的调整。具体来说，Clip-Higher策略使用两个不同的裁剪参数：

*   ϵlow\\epsilon\_{low}\\epsilon\_{low}：用于限制高概率token的概率降低幅度，防止过度利用。
*   ϵhigh\\epsilon\_{high}\\epsilon\_{high}：用于限制低概率token的概率提升幅度，允许更大的探索空间。

在Clip-Higher策略中，目标函数可以表示为：

LCLIP(θ)\=E^t\[min(rt(θ)A^t,clip(rt(θ),1−ϵlow,1+ϵhigh)A^t)\]L^{CLIP}(\\theta) = \\hat{E}\_t\[min(r\_t(\\theta)\\hat{A}\_t, clip(r\_t(\\theta), 1-\\epsilon\_{low}, 1+\\epsilon\_{high})\\hat{A}\_t)\]L^{CLIP}(\\theta) = \\hat{E}\_t\[min(r\_t(\\theta)\\hat{A}\_t, clip(r\_t(\\theta), 1-\\epsilon\_{low}, 1+\\epsilon\_{high})\\hat{A}\_t)\]

通过这种方式，Clip-Higher策略可以更灵活地控制策略更新的幅度，从而更好地平衡探索和利用。

*   **对低概率token的影响**：由于ϵhigh\>ϵlow\\epsilon\_{high} > \\epsilon\_{low}\\epsilon\_{high} > \\epsilon\_{low}，低概率token的概率提升空间更大。例如，如果一个token的原始概率为0.01，ϵhigh\=0.28\\epsilon\_{high}=0.28\\epsilon\_{high}=0.28，那么更新后的概率最大可以达到0.0128，提升了28%。这有助于模型更充分地探索低概率token，从而发现潜在的更优策略。
*   **对高概率token的影响**：由于ϵlow<ϵhigh\\epsilon\_{low} < \\epsilon\_{high}\\epsilon\_{low} < \\epsilon\_{high}，高概率token的概率降低空间更小。这意味着模型在更新策略时，会更加谨慎地对待这些高概率token，防止过度利用导致性能下降。

总而言之，Clip-Higher策略通过解耦上下限裁剪，允许模型更自由地探索低概率token，同时防止过度利用高概率token，从而提高算法的性能和稳定性。

### Q4: 为什么要使用 Dynamic Sampling？

Dynamic Sampling 的目的是**保证每个 batch 中的样本都能提供有效的梯度信号**。

*   **问题**：在训练过程中，模型可能会对某些问题产生非常自信的输出，导致所有输出都正确或都错误。这时优势函数均值为零，导致梯度消失。这些样本无法提供有效的梯度信号，会降低训练效率。
*   **Dynamic Sampling 的作用**：通过过滤掉这些样本，Dynamic Sampling 可以保证每个 batch 中的样本都能对模型的训练产生积极作用。

持续过滤全对/全错样本，实际上构建了一个「中等难度」训练集：

![](https://pic2.zhimg.com/v2-a7583f3eefd5899c68d70d666fc629e3_r.jpg)

训练效果：

![](https://picx.zhimg.com/v2-33551397d01c6dfa449feaa69e6c3539_r.jpg)

### Q5: Token-Level Policy Gradient Loss 如何解决长 CoT 场景下的问题？

Token-Level Policy Gradient Loss 通过**将损失函数计算的粒度从 sample 级别降低到 token 级别**，来解决长 CoT 场景下长序列对梯度贡献过小的问题。

*   **问题**：在 sample 级别计算损失时，每个 sample 的权重是相同的。这意味着，长序列中的每个 token 对梯度的贡献会被平均化，导致长序列对模型更新的影响较小。
*   **Token-Level Policy Gradient Loss 的优势**：通过在 token 级别计算损失，每个 token 的贡献都会被平等地考虑，从而解决了长序列对梯度贡献过小的问题。

![](https://pica.zhimg.com/v2-3d6c2b921dcf87427edc82c6777b99fe_r.jpg)

### Q6: Overlong Reward Shaping 的作用是什么？

Overlong Reward Shaping 的目的是**避免因过度惩罚导致奖励噪声**。

*   **问题**：在 RL 训练中，通常会设置一个最大生成长度。对于超过最大长度的样本，通常会给予一个惩罚。但是，如果惩罚过重，可能会导致模型对那些只是稍微超过最大长度，但推理过程正确的样本也产生负面影响。
*   **Overlong Reward Shaping 的优势**：通过提出一种基于长度的惩罚机制，Overlong Reward Shaping 可以更合理地惩罚过长样本，避免因过度惩罚导致奖励噪声。

分段惩罚函数：

R\_{\\text{length}}(y) = \\begin{cases} 0 & |y| \\leq L\_{\\max}-L\_{\\text{cache}} \\\\ \\frac{(L\_{\\max}-L\_{\\text{cache}}) - |y|}{L\_{\\text{cache}}} & \\text{过渡区} \\\\ -1 & |y| > L\_{\\max} \\end{cases} \\\\R\_{\\text{length}}(y) = \\begin{cases} 0 & |y| \\leq L\_{\\max}-L\_{\\text{cache}} \\\\ \\frac{(L\_{\\max}-L\_{\\text{cache}}) - |y|}{L\_{\\text{cache}}} & \\text{过渡区} \\\\ -1 & |y| > L\_{\\max} \\end{cases} \\\\

其中L\_{cache}=4096L\_{cache}=4096为缓冲区间，平滑过渡惩罚，使超长样本比例下降29%。

![](https://pic1.zhimg.com/v2-bc4445e6968b7f80130687ca7baada0c_r.jpg)

### Q7: DAPO 算法有哪些潜在的改进方向？

*   **更智能的 Dynamic Sampling 策略**：可以根据样本的难度和模型的学习状态来动态调整采样策略，以进一步提高训练效率。
*   **自适应的 Clip-Higher 参数**：可以根据训练的进展情况来自动调整 clipping 的上下限，以更好地平衡探索和利用。
*   **结合 value function**：可以考虑将 value function 引入到 DAPO 算法中，以更准确地估计 advantage，从而提高训练效果。
*   **探索不同的奖励函数**：可以尝试使用更复杂的奖励函数，例如基于人类反馈的奖励函数，以更好地引导模型的训练。
*   **加入正则化项**：可以加入一些正则化项，例如 entropy regularization，以防止模型过拟合。

* * *

伪代码
---

```python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DAPOPolicyNetwork(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # 预训练语言模型
        
    def forward(self, input_ids, attention_mask):
        # 获取token级别的logits
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        return outputs.logits

class DAPOTrainer:
    def __init__(self, policy_net, ref_net, optimizer, args):
        """
        Args:
            policy_net: 当前策略网络
            ref_net: 参考策略网络（旧策略）
            args: 包含超参数的配置对象
        """
        self.policy_net = policy_net
        self.ref_net = ref_net
        self.optimizer = optimizer
        self.eps_low = args.eps_low  # 裁剪下限（默认0.2）
        self.eps_high = args.eps_high  # 裁剪上限（默认0.28）
        self.max_length = args.max_length  # 最大生成长度

    def compute_loss(self, batch):
        """
        计算DAPO的核心损失函数
        """
        # 解包批次数据
        input_ids = batch['input_ids']  # 问题prompt [B, L]
        attention_mask = batch['attention_mask']
        old_logits = batch['old_logits']  # 旧策略的logits [B, L, V]
        actions = batch['actions']  # 生成的token序列 [B, T]
        rewards = batch['rewards']  # 每个样本的奖励 [B,]
        
        # 获取当前策略的logits [B, T, V]
        logits = self.policy_net(input_ids, attention_mask)
        
        # 计算重要性采样比率（核心公式）
        # π_θ(a|s) / π_old(a|s)
        probs = torch.softmax(logits, dim=-1)
        old_probs = torch.softmax(old_logits, dim=-1)
        ratios = probs / old_probs  # [B, T]
        
        # 非对称裁剪（Clip-Higher策略）
        clipped_ratios = torch.clamp(
            ratios, 
            1 - self.eps_low, 
            1 + self.eps_high
        )
        
        # 计算优势函数（Group Relative）
        # 假设已预先计算好优势值 [B, T]
        advantages = batch['advantages']  
        
        # Token级策略梯度损失（关键实现）
        surr1 = ratios * advantages
        surr2 = clipped_ratios * advantages
        policy_loss = -torch.min(surr1, surr2)  # [B, T]
        
        # 平均到token级别
        loss = policy_loss.mean(dim=1).mean()  # 先对token维度平均，再对batch平均
        
        return loss

    def dynamic_sampling(self, dataloader):
        """
        动态采样机制实现
        返回有效样本的DataLoader
        """
        valid_samples = []
        for batch in dataloader:
            # 生成多个响应
            outputs = self.generate_responses(batch['questions'])
            
            # 过滤全对/全错样本（关键逻辑）
            valid_mask = (outputs['rewards'] != 1.0) & (outputs['rewards'] != 0.0)
            valid_batch = {k: v[valid_mask] for k, v in outputs.items()}
            
            valid_samples.append(valid_batch)
            
            # 持续采样直到达到批次大小
            while len(valid_samples) < self.batch_size:
                extra_batch = self.generate_responses(batch['questions'])
                valid_samples.extend(extra_batch)
        
        return DataLoader(valid_samples, batch_size=self.batch_size)

    def generate_responses(self, questions):
        """
        生成带奖励的响应（包含超长奖励塑形）
        """
        # 生成响应
        outputs = self.policy_net.generate(
            questions,
            max_length=self.max_length + 4096,  # 包含缓冲区间
            do_sample=True
        )
        
        # 计算基础奖励（规则奖励）
        base_rewards = self.rule_based_reward(outputs.answers, questions.answers)
        
        # 超长惩罚计算（分段函数实现）
        lengths = outputs.lengths
        length_penalty = torch.zeros_like(base_rewards)
        
        # 缓冲区间 [L_max - L_cache, L_max]
        mask = (lengths > (self.max_length - 4096)) & (lengths <= self.max_length)
        length_penalty[mask] = ((self.max_length - 4096) - lengths[mask]) / 4096
        
        # 超长区间
        mask = lengths > self.max_length
        length_penalty[mask] = -1.0
        
        # 最终奖励
        total_rewards = base_rewards + length_penalty
        
        return {
            'input_ids': outputs.input_ids,
            'rewards': total_rewards,
            'lengths': lengths
        }

    def rule_based_reward(self, pred_answers, true_answers):
        """
        规则奖励计算（AIME场景示例）
        """
        rewards = []
        for pred, true in zip(pred_answers, true_answers):
            if self.is_equivalent(pred, true):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        return torch.tensor(rewards)

    @staticmethod
    def is_equivalent(a, b):
        """数学答案等价性判断（简化示例）"""
        try:
            return abs(float(a) - float(b)) < 1e-6
        except:
            return a.strip() == b.strip()

class LengthAwareDataset(Dataset):
    """
    带长度感知的数据集（支持超长样本处理）
    """
    def __init__(self, data, max_length=16384, cache_size=4096):
        self.data = data
        self.max_length = max_length
        self.cache_size = cache_size
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 应用长度敏感处理
        if len(sample['response']) > self.max_length:
            sample = self._truncate_with_penalty(sample)
        return sample
    
    def _truncate_with_penalty(self, sample):
        """带缓冲区的截断处理"""
        trunc_length = self.max_length + self.cache_size
        truncated = sample['response'][:trunc_length]
        sample['response'] = truncated
        return sample

# ------------------- 训练流程示例 -------------------
if __name__ == "__main__":
    # 初始化组件
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-32B")
    policy_net = DAPOPolicyNetwork(base_model)
    ref_net = DAPOPolicyNetwork(base_model.clone())  # 深拷贝参考网络
    
    # 配置超参数
    class Args:
        eps_low = 0.2
        eps_high = 0.28
        max_length = 16384
        batch_size = 512
        
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-6)
    trainer = DAPOTrainer(policy_net, ref_net, optimizer, Args)
    
    # 动态采样数据加载
    dataset = LengthAwareDataset(load_dapo_math_data())
    dynamic_loader = trainer.dynamic_sampling(DataLoader(dataset, batch_size=32))
    
    # 训练循环
    for epoch in range(100):
        for batch in dynamic_loader:
            # 冻结参考网络参数
            with torch.no_grad():
                old_logits = ref_net(batch['input_ids'])
            
            # 计算损失
            loss = trainer.compute_loss({
                **batch,
                'old_logits': old_logits
            })
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新参考网络
            if epoch % 10 == 0:
                ref_net.load_state_dict(policy_net.state_dict())
```

本文转自 <https://zhuanlan.zhihu.com/p/32368626065>，如有侵权，请联系删除。