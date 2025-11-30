​

目录

收起

主要内容

1\. 作者和团队信息

2\. 背景和动机

3\. 相关研究

4\. 核心思路

5\. 方案与技术

6\. 实验与结论

7\. 贡献

8\. 不足

QA

Q1：为什么传统的 LLM 预训练方法会受到数据量的限制？

Q2：RL在 LLM 训练中有什么优势？

Q3：这篇论文中提出的长文本 CoT 的 RL 方法，是如何实现隐式规划的？

Q4：什么是「部分 rollout」？它在长文本 CoT 的 RL 中有什么作用？

Q5：这篇论文中提到的「长度惩罚」是什么？为什么需要它？

Q6：什么是「课程学习」和「优先采样」？它们是如何提高训练效率的？

Q7：论文中提到的 「Long2short」 方法是什么？为什么要使用这种方法？

Q8：这篇论文的实验部分，是如何证明长文本能力在 RL 训练 LLM 中的重要性的？

伪代码实现

这几天太热闹了，国内几家基础模型公司感觉「不过了」，纷纷放出了自家的先进模型，前有[MiniMax-01](https://zhida.zhihu.com/search?content_id=252945019&content_type=Article&match_order=1&q=MiniMax-01&zhida_source=entity)、[DeepSeek-R1](https://zhida.zhihu.com/search?content_id=252945019&content_type=Article&match_order=1&q=DeepSeek-R1&zhida_source=entity)，Kimi马上也跟上了，不过Kimi只有技术报告，并没有开源。相比moonshot一贯的路线，能放出技术报告也难能可贵了。可能大家都看到了前一段时间[DeepSeek-V3](https://zhida.zhihu.com/search?content_id=252945019&content_type=Article&match_order=1&q=DeepSeek-V3&zhida_source=entity)「破圈」传播的好处。

Kimi一如既往认为**长文本是核心**，在「推理时scaling」也一样，不用搞那些树状搜索什么花里胡哨的，只要上下文够长，CoT就行了。

论文：[Kimi-k1.5/Kimi\_k1.5.pdf at main · MoonshotAI/Kimi-k1.5 · GitHub](https://link.zhihu.com/?target=https%3A//github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)

* * *

主要内容
----

### 1\. 作者和团队信息

*   **核心贡献者**：尽管作者众多，但报告并未明确指出核心贡献者，所有作者按照名字首字母排序。这表明该项目是一个大型团队合作的成果。
*   **团队背景**：根据报告内容，Kimi 团队专注于[多模态大型语言模型](https://zhida.zhihu.com/search?content_id=252945019&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)（LLM）的研发，特别是如何通过强化学习（RL）提升 LLM 的性能。

### 2\. 背景和动机

*   **发表时间**：2025年1月。
*   **研究问题**：该报告主要探讨如何利用强化学习（RL）来提升大型语言模型（LLM）的性能，特别是在长文本理解和复杂推理方面的能力。
*   **问题背景**：

*   **预训练的局限性**：传统的 LLM 预训练依赖于大规模文本数据的下一词预测，但这种方法受限于高质量训练数据的数量。
*   **RL 的潜力**：强化学习（RL）为 LLM 的持续改进提供了新的方向，通过奖励机制，模型可以学习探索并从经验中学习，不受静态数据集的限制。
*   **现有 RL 方法的不足**：之前的研究表明，直接使用 RL 训练 LLM 并没有取得理想的成果，没有展现出明显的优势。

### 3\. 相关研究

*   **预训练模型扩展**：之前的工作主要通过扩大模型参数和训练数据量来提升 LLM 的性能，这在一定程度上受到了数据量的限制。
*   **RL 在 LLM 中的应用**：尽管 RL 在其他领域（如游戏）取得了显著成就，但在 LLM 训练中的应用仍处于探索阶段，尚未有突出的成果。
*   **CoT（Chain-of-Thought）方法**：CoT 是一种通过中间推理步骤来解决复杂问题的技术，但传统的 CoT 方法依赖于预定义的推理路径，限制了模型的探索能力。

### 4\. 核心思路

*   **利用 RL 探索**：Kimi k1.5 的核心思想是利用强化学习，让模型通过试错（探索）来学习解决问题的能力，而不是仅仅依赖于静态数据集。
*   **长文本 CoT 的 RL**：将 RL 应用于长文本的 Chain-of-Thought（CoT）推理过程，使模型能够进行更深入、更复杂的推理。
*   **隐式规划**：通过增加上下文长度，让模型在生成 CoT 的过程中进行隐式的规划、反思和修正，无需显式的搜索树或价值函数。
*   **长文本能力是关键**：该文章的核心观点是，长文本能力是强化学习训练LLM的关键，而不是更复杂的训练技巧。
*   **长文本到短文本**：通过长文本 CoT 模型来指导短文本 CoT 模型的训练，从而在有限的计算资源下获得更好的性能。

### 5\. 方案与技术

*   **多阶段训练**：Kimi k1.5 的训练包括预训练、有监督微调（SFT）和强化学习（RL）等阶段，其中 RL 是重点。

*   **RL Prompt Set Curation**：精心设计高质量的 RL 提示集，包括：

*   **多样性**：覆盖 STEM、编码、通用推理等多个领域。
*   **难度均衡**：包含简单、中等和困难的问题，以促进模型的逐步学习。
*   **可评估性**：问题应能被客观评估，避免模型通过不正确的推理获得正确答案。

*   **Long-CoT SFT**：使用高质量的长文本 CoT 数据集进行有监督微调，使模型具备规划、评估、反思和探索的能力。

*   **强化学习（RL）**：

*   **策略优化**：采用在线策略镜像下降算法（online policy mirror descent）的变体进行策略优化。
*   **长度惩罚**：引入长度惩罚机制，防止模型生成过长的推理过程，提高计算效率。
*   **采样策略**：使用课程学习和优先采样策略来提高训练效率，让模型更关注困难问题。

*   **Long2short 方法**：

*   **模型融合**：将长文本 CoT 模型和短文本 CoT 模型的权重进行平均，得到一个新的模型。
*   **最短拒绝采样**：从多个采样结果中选择最短且正确的答案。
*   **DPO（Direct Preference Optimization）**：使用长文本 CoT 模型生成的答案作为偏好数据来训练短文本 CoT 模型。
*   **Long2short RL**：在标准 RL 训练后，使用长度惩罚对模型进行微调，进一步提高短文本 CoT 模型的效率。

*   **RL 基础设施**：

*   **迭代同步框架**：RL 训练系统采用迭代同步方法，包括 rollout 阶段和训练阶段。
*   **部分 rollout**：使用部分 rollout 技术来处理长文本 CoT，避免单个过长轨迹占用过多资源。
*   **混合部署**：结合 Megatron 和 vLLM，实现训练和推理的混合部署，提高 GPU 利用率。
*   **代码沙箱**：提供安全的代码执行环境，用于评估代码生成任务。  
    

![](https://pic4.zhimg.com/v2-bfd5a9e8a5dc31a983995dd4083eb82d_r.jpg)

### 6\. 实验与结论

*   **评估基准**：

*   **文本基准**：包括 MMLU、IF-Eval、CLUEWSC、C-Eval 等。
*   **推理基准**：包括 HumanEval-Mul、LiveCodeBench、Codeforces、AIME 2024、MATH-500 等。
*   **视觉基准**：包括 MMMU、MATH-Vision、Math Vista 等。

*   **主要结果**：

*   **长文本 CoT 模型**：在多个基准测试中取得领先水平，匹配 OpenAI 的 o1 模型。
*   **短文本 CoT 模型**：在多个基准测试中超越了 GPT-4o 和 Claude Sonnet 3.5 等模型。

*   **长文本缩放**：实验表明，随着上下文长度的增加，模型的性能也持续提升。
*   **Long2short 实验**：Long2short RL 方法在 token 效率方面优于 DPO 和模型融合等方法。
*   **消融研究**：

*   **模型大小与上下文长度**：更大的模型在性能和 token 效率方面都更优，但使用较长的上下文长度，较小的模型可以获得接近的结果。
*   **负梯度**：使用负梯度的方法比 ReST（Reinforced Self-Training）方法表现更好。
*   **课程学习**：课程学习方法可以显著提高模型的性能。  
    

![](https://pic4.zhimg.com/v2-32788b9fe935ea1f365443d2be869567_r.jpg)

long-CoT

  

![](https://pic4.zhimg.com/v2-6dff0cb90a15a6eecba5ae3bb2518255_r.jpg)

short-CoT

### 7\. 贡献

*   **新的 RL 训练方法**：提出了一种简单有效的 RL 框架，通过长文本 CoT 和优化的策略学习，无需复杂的搜索树或价值函数。
*   **长文本能力是关键**：强调了长文本能力在 RL 训练 LLM 中的重要作用，而不是更复杂的训练技巧。
*   **Long2short 方法**：提出了多种 long2short 方法，用于将长文本 CoT 模型的知识迁移到短文本 CoT 模型，提高了 token 效率。
*   **多模态能力**：Kimi k1.5 具备强大的多模态能力，可以在文本和视觉任务中实现高性能。
*   **系统优化**：通过部分 rollout 和混合部署等技术，优化了 RL 训练的效率和可扩展性。

### 8\. 不足

*   **实验细节不足**：报告中缺少一些关键的实验细节，比如具体的数据集划分、超参数设置等，这会影响结果的可复现性。
*   **对负梯度作用的解释不够深入**：虽然通过实验证明了负梯度的有效性，但没有深入分析其背后的原因。
*   **长度惩罚的初始阶段问题**：报告中提到长度惩罚在初始阶段可能会减慢训练速度，但没有给出具体的解决方案。
*   **对「隐式规划」机制的理解**：文章中只是用「隐式规划」来解释长文本CoT的作用，但并没有给出充分的理论依据和实验支撑。
*   **RL 训练的计算资源需求**：尽管提出了优化的方法，但 RL 训练本身对计算资源的要求仍然很高。

* * *

QA
--

### Q1：为什么传统的 LLM 预训练方法会受到数据量的限制？

传统的 LLM 预训练方法主要依赖于对大规模文本数据进行下一词预测。这种方法虽然有效，但它本质上是在学习现有数据的分布。随着模型参数的增大，对高质量、多样化的数据需求也会迅速增加，而互联网上的高质量数据是有限的。此外，通过简单地增加数据规模并不能解决模型理解和推理等复杂问题。因此，这种方法的扩展性受到了限制。

### Q2：RL在 LLM 训练中有什么优势？

强化学习是一种机器学习方法，它让智能体（Agent）通过与环境的互动来学习。智能体在环境中执行动作，环境根据动作给出奖励或惩罚，智能体通过不断尝试和调整策略来最大化累积奖励。

在 LLM 训练中，RL 的优势在于：

*   **探索能力**：模型可以主动探索不同的解决方案，而不是仅仅依赖于预先提供的训练数据。
*   **适应性**：模型可以根据奖励信号来调整其行为，更好地适应不同的任务和环境。
*   **解决复杂问题**：RL 可以用于训练模型解决复杂的推理问题，通过中间步骤的反馈来逐步改进。

### Q3：这篇论文中提出的长文本 CoT 的 RL 方法，是如何实现隐式规划的？

论文的核心思想是利用长文本窗口，让模型在生成 CoT 的过程中，能够回溯、反思和纠正之前的错误。由于上下文窗口足够长，模型可以记住之前的推理步骤，并且根据上下文信息调整后续步骤。  
这种方式无需像传统规划算法那样建立显式的搜索树，而是让模型在生成长文本的过程中完成「隐式规划」。你可以把这想象成一个人在解决复杂问题时，通过不断思考、回顾、纠正来一步步找到答案，而不需要事先制定好详细的计划。

### Q4：什么是「部分 rollout」？它在长文本 CoT 的 RL 中有什么作用？

「部分 rollout」 是一种在 RL 训练中处理长轨迹的技术。在传统的 RL 中，智能体需要完成一个完整的轨迹才能计算奖励。但在长文本 CoT 的 RL 中，一个推理过程可能非常长，如果每次都等待整个过程完成再计算奖励，效率会很低。  
「部分 rollout」 的做法是将长轨迹拆分成多个片段，每个片段都作为独立的 rollout 进行处理。这样，模型可以在不需要等待整个轨迹完成的情况下，就可以接收到反馈信号，从而加快训练速度。  
另外，通过复用之前迭代的轨迹，可以避免每次都从头开始生成新的长文本，节省计算资源。

### Q5：这篇论文中提到的「长度惩罚」是什么？为什么需要它？

在 RL 训练过程中，模型有时会倾向于生成过长的答案，即使这些答案并不一定更好。这是因为模型的目标是最大化奖励，而奖励信号有时可能无法很好地衡量答案的简洁性。  
「长度惩罚」 就是在奖励中加入一个惩罚项，对于过长的答案进行惩罚，鼓励模型生成更简洁、高效的答案。这不仅可以提高计算效率，也可以让模型更符合人类的思维习惯。

### Q6：什么是「课程学习」和「优先采样」？它们是如何提高训练效率的？

*   **课程学习（Curriculum Learning）：** 就像人类学习一样，模型也应该从简单的问题开始，逐步过渡到更复杂的问题。「课程学习」就是根据问题的难度，先让模型学习容易的例子，再逐渐引入更难的例子，这种循序渐进的方式有助于模型更好地掌握知识。
*   **优先采样（Prioritized Sampling）：** 在 RL 训练中，有些问题可能更容易解决，而有些问题则更困难。优先采样就是根据问题的难度或模型对问题的掌握程度来调整采样概率。模型更倾向于采样那些困难的、或者模型不擅长的问题，这样可以集中精力在薄弱环节，提高训练效率。

### Q7：论文中提到的 「Long2short」 方法是什么？为什么要使用这种方法？

「Long2short」 方法指的是将长文本 CoT 模型的知识迁移到短文本 CoT 模型，本质上是一种「蒸馏」，不过目标和策略更多样，不仅要性能，还要 token 效率；更多地关注对教师模型推理策略的学习，而不仅仅是输出。

之所以要用这种方法，是因为长文本 CoT推理过程需要消耗更多的计算资源和时间。对于一些对计算资源有限制或者对响应时间有要求的场景，短文本 CoT 模型更具优势，兼顾性能和效率。

### Q8：这篇论文的实验部分，是如何证明长文本能力在 RL 训练 LLM 中的重要性的？

*   **模型性能与上下文长度的关系**：实验表明，随着上下文长度的增加，模型的性能也在不断提升，这说明长上下文可以带来更强的推理能力。
*   **不同模型大小的对比**：实验表明，虽然更大的模型在初始阶段表现更好，但通过长文本 CoT 和 RL 的训练，较小的模型也可以达到接近的性能。这表明，长文本能力和训练方法比模型大小本身更重要。
*   **消融研究**：通过消融研究，证明了负梯度、课程学习等训练技巧能够增强长文本 CoT 的能力，进一步说明了长文本 CoT 是 RL 的关键。

* * *

伪代码实现
-----

```python3
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

##############################################################################
# 1. 定义一个示例性的数据集和模型 (Toy Dataset & Toy Model)
#    仅用于展示流程，无法对应真实大规模 LLM 或多模态数据
##############################################################################

class ToyDataset(Dataset):
    """
    一个简单的示例数据集，每条数据由(prompt, answer)构成。
    在真实场景下，这里可能来自多模态数据（文本+图像），
    或者高质量数学、代码等多样化数据。
    """
    def __init__(self, size=1000):
        super().__init__()
        self.data = []
        for i in range(size):
            # prompt 和 answer 仅作示例，真实情况下需要高质量多模态数据
            prompt = f"问题{i}"
            answer = f"答案{i}"
            self.data.append((prompt, answer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ToyLLM(nn.Module):
    """
    一个示例性的语言模型，用简单的embedding和线性层搭建。
    真实的LLM通常采用 Transformer Decoder 等复杂结构，并支持多模态输入。
    """
    def __init__(self, vocab_size=1000, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embed(input_ids)             # [batch_size, seq_len, hidden_dim]
        # 假设只用最后一个 token 的表示来预测
        last_token = x[:, -1, :]             # [batch_size, hidden_dim]
        logits = self.lm_head(last_token)    # [batch_size, vocab_size]
        return logits


##############################################################################
# 2. 有监督微调 (SFT) 的示例性流程
#    在报告中，SFT 先于 RL，并为后续的 Long-CoT 或多模态打下基础
##############################################################################

def supervised_finetune(model, dataloader, tokenizer, lr=1e-4, epochs=1):
    """
    在报告中，SFT 涉及多领域数据、去重和质量控制。
    这里仅用一个非常简单的方式演示“有监督微调”的流程。
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for prompts, answers in dataloader:
            # 将prompt和answer转成token id，这里仅作伪造
            # 实际上需要使用真正的tokenizer并且注意多模态处理
            inputs = []
            labels = []
            batch_size = len(prompts)
            for p, a in zip(prompts, answers):
                # 伪造输入：简单起见，将 prompt+answer 拼起来并构造 token
                tokens = tokenizer(p + a) 
                inputs.append(tokens[:-1])
                labels.append(tokens[1:])
            
            # 对齐序列长度，真实情况需要更复杂的padding逻辑
            max_len = max(len(t) for t in inputs)
            input_ids = []
            label_ids = []
            for t_in, t_lb in zip(inputs, labels):
                t_in = t_in + [0]*(max_len - len(t_in))   # 用0表示pad
                t_lb = t_lb + [0]*(max_len - len(t_lb))
                input_ids.append(t_in)
                label_ids.append(t_lb)

            input_ids = torch.tensor(input_ids)  # [batch_size, seq_len]
            label_ids = torch.tensor(label_ids)  # [batch_size, seq_len]

            optimizer.zero_grad()
            logits = model(input_ids)            # [batch_size, vocab_size]
            # 这里为了简单只计算最终token的预测loss（示例）
            loss = loss_fn(logits, label_ids[:, -1])
            loss.backward()
            optimizer.step()


##############################################################################
# 3. 部分 rollouts (Partial Rollout) 与 RL 训练准备
#    在报告中，为了支持长上下文，采用“部分rollout”技术。
##############################################################################

class PartialRolloutBuffer:
    """
    维护长序列的“部分rollout”缓存示例。
    在真实场景下，需要将长序列拆分到不同iteration中，并重复使用已生成的片段。
    """
    def __init__(self):
        # 存储结构: list of (prompt, partial_response, is_finished, reward)
        self.storage = []

    def add_step(self, prompt, partial_response, is_finished, reward):
        self.storage.append((prompt, partial_response, is_finished, reward))

    def sample(self, batch_size=8):
        # 这里简单随机采样，真实情况往往需要难度值、成功率等信息做采样策略
        batch = random.sample(self.storage, min(batch_size, len(self.storage)))
        return batch


class RewardModel(nn.Module):
    """
    示例性的奖励模型。现实中可能基于代码测试用例、数学答案校验、模型判断等。
    这里仅做一个简单的线性分类器，判断回答正确与否(0/1)。
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state):
        # 简单线性映射到reward
        return torch.sigmoid(self.linear(hidden_state)).squeeze(-1)


##############################################################################
# 4. RL 训练核心：在线策略镜像下降 (policy mirror descent) + 长度惩罚
#    这里给出一个极简化的实现示例，演示 RL 大致流程
##############################################################################

def compute_length_penalty(reward, generated_length, alpha=0.1):
    """
    针对报告中的长度惩罚 (length penalty) 思路：
    - alpha 为长度惩罚的系数
    - 这是一个极简化版本，在真实场景中需更灵活地处理
    """
    # 当回答正确时：长度越长，奖励越低
    # 当回答错误时：可进一步减分
    # 这里只是演示：reward -= alpha * generated_length
    return reward - alpha * generated_length


def rl_training_loop(model, 
                     reward_model,
                     partial_buffer: PartialRolloutBuffer,
                     tokenizer,
                     lr=5e-5, 
                     temperature=1.0, 
                     rl_iterations=5,
                     batch_size=8):
    """
    极简的RL训练示例，使用类似“离线” policy mirror descent 的思路。
    注意：这只是演示核心思路，与报告中的大规模多机分布式差距巨大。
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for iteration in range(rl_iterations):
        # 每次迭代，从缓存中采样部分rollout
        batch_data = partial_buffer.sample(batch_size)

        if not batch_data:
            print("无可采样的Rollout数据，跳过本次迭代。")
            continue

        # 构建训练所需的输入
        # report中：对于每条 (prompt, partial_resp)，需要在长token中间插入
        # 在这里仅做非常简化的处理
        inputs = []
        old_log_probs = []
        advantages = []

        for (prompt, partial_resp, is_finished, raw_reward) in batch_data:
            token_ids = tokenizer(prompt + partial_resp)
            # reward带上长度惩罚
            length_penalty_reward = compute_length_penalty(
                raw_reward, 
                generated_length=len(token_ids),
                alpha=0.01
            )

            # 这里省略如何计算 old_log_prob，只演示思路
            # 真实场景需要从 reference policy 中获取 old_log_prob
            old_log_prob = 0.0
            advantage = length_penalty_reward  # 简化处理 advantage
            inputs.append(token_ids)
            old_log_probs.append(old_log_prob)
            advantages.append(advantage)

        # 对齐长度
        max_len = max(len(t) for t in inputs)
        padded_inputs = []
        for t in inputs:
            t = t + [0]*(max_len - len(t))
            padded_inputs.append(t)

        padded_inputs = torch.tensor(padded_inputs)  # [batch_size, seq_len]
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # 计算 loss
        # 这里只演示：logits = model.forward(...) → 取最后一个token的 log_prob
        # 并与 advantages 结合。
        optimizer.zero_grad()
        logits = model(padded_inputs)  # [batch_size, vocab_size]
        # 针对最后一个token的 log_prob
        log_probs = torch.log_softmax(logits, dim=-1)
        # 我们用一个非常粗糙的方式来近似"正确token"=0，或随便选
        # 这里仅为演示，不代表真实算法
        chosen_log_prob = log_probs[:, 0]

        # Policy Mirror Descent:  L = - ( advantage * [log_pi - log_pi_old] )
        # 这里忽略了 entropy regularizer、归一化等关键细节
        # 仅做简化演示
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        policy_loss = - torch.mean(advantages * (chosen_log_prob - old_log_probs))

        policy_loss.backward()
        optimizer.step()

        print(f"[RL Iteration {iteration+1}/{rl_iterations}] policy_loss={policy_loss.item():.4f}")


##############################################################################
# 5. 封装一个主要训练流程 demo: 
#    1) 初始化数据和模型 -> 2) SFT -> 3) Long-CoT SFT (仅示意) -> 4) 部署 RL 训练
#    5) RL 结束后可做 Long2short 等技术 (此处只做 placeholder)
##############################################################################

def toy_tokenizer(text: str):
    """
    一个示例 tokenizer，非常简化：直接返回字符级别ID或按空格分词。
    真实的LLM会使用专门的分词方式，还要支持多模态的处理逻辑。
    """
    # 这里仅返回 ASCII 编码的简单数字映射
    return [ord(c) % 100 for c in text]  # 仅保留余数避免过大

def main():
    # 1) 数据与模型初始化
    dataset = ToyDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = ToyLLM()
    reward_model = RewardModel()
    partial_buffer = PartialRolloutBuffer()

    # 2) 有监督微调 (SFT)
    print("=== 阶段1：有监督微调 SFT ===")
    supervised_finetune(model, dataloader, tokenizer=toy_tokenizer, lr=1e-4, epochs=1)

    # 3) Long-CoT SFT (仅作演示，这里不做二次训练，只打个提示)
    #    真实情况会再进行“长上下文”数据的精调，使模型具备更长的推理链。
    print("=== 阶段2：Long-CoT SFT (示例中跳过) ===")
    # 在真实训练中，这里会载入另一个高质量的长CoT数据集进行专门SFT

    # 4) 进行部分rollout示例(生成一些假数据) -> 放入 partial_buffer
    #    假设我们已经有了若干 (prompt, partial_response, reward)
    #    真实场景中，这些数据在 rollouts worker 上生成、送入 replay buffer
    print("=== 创建 假的 Partial Rollout 数据 ===")
    for i in range(50):
        prompt = f"[PROMPT] 数学题{i}: 计算 2+2=?"
        partial_resp = "[COT] 先假设2+2=5，然后自我反思... 改正：结果=4"
        # 假设我们有奖励(1.0表示正确,0表示错误,此处随机)
        reward = 1.0 if random.random()>0.3 else 0.0
        is_finished = True
        partial_buffer.add_step(prompt, partial_resp, is_finished, reward)

    # 5) 启动 RL 训练
    print("=== 阶段3：RL 训练 ===")
    rl_training_loop(model, 
                     reward_model,
                     partial_buffer,
                     tokenizer=toy_tokenizer,
                     lr=5e-5,
                     temperature=1.0,
                     rl_iterations=5,
                     batch_size=8)

    # 6) Long2short: 将长文本CoT模型的知识迁移到短文本CoT模型示例
    #    此处仅作“占位符”，在真实系统中可用 DPO、model merging 等方法。
    print("=== 阶段4：Long2short (示例中只做提示，不做具体实现) ===")
    print("可以在此处实现 DPO 或 最短拒绝采样等方法，将长文本推理经验迁移到短文本模型中。")


if __name__ == "__main__":
    main()
```

这是一个极度简化的教学/概念示例，用于演示 Kimi k1.5 技术报告中提出的部分核心思路，比如：

*   多阶段训练（先 SFT，再 Long-CoT SFT，最后 RL 等）；
*   部分 rollout 缓存 (partial rollouts)；
*   RL 策略优化（policy mirror descent、长度惩罚）；
*   Long2short 的理念（将长文本推理的知识迁移到短文本）的占位。

本文转自 <https://zhuanlan.zhihu.com/p/19612718816>，如有侵权，请联系删除。