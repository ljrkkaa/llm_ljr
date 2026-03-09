https://zhuanlan.zhihu.com/p/14710836610

https://zhuanlan.zhihu.com/p/13936916587  (全文)

![](https://pica.zhimg.com/v2-a4f52e1b0f3735d5844c42e5764483b8_r.jpg)

*   论文地址：[https://arxiv.org/pdf/2412.15115](http://arxiv.org/pdf/2412.15115)
*   Github：[https://github.com/QwenLM/Qwen2.5](http://github.com/QwenLM/Qwen2.5)
*   HF：[https://huggingface.co/Qwen](http://huggingface.co/Qwen)
*   [魔搭](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=%E9%AD%94%E6%90%AD&zhida_source=entity)：[https://modelscope.cn/organization/qwen](http://modelscope.cn/organization/qwen)

核心内容
----

*   **预训练**：[预训练数据](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE&zhida_source=entity)从之前的 7T token（7万亿）扩展到 **18T（18万亿）token**，重点关注知识、代码和数学，通过不同数据配比分阶段预训练，为常识、专家知识、推理能力提供基础。
*   **后训练**：[sft](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=sft&zhida_source=entity) 了 **100w+ 指令数据**，还做了**[离线 DPO](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=%E7%A6%BB%E7%BA%BF+DPO&zhida_source=entity)** 和**[在线 GRPO](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=%E5%9C%A8%E7%BA%BF+GRPO&zhida_source=entity)** 的多阶段强化学习。增强人类偏好，改进了长文生成（从2K个token到8K个token）、结构化输入和输出（例如表格和JSON）、简单的 Function/tool call、指令遵循能力。
*   **模型权重**：开源稠密模型包括 0.5B、1.5B、3B、7B、14B、32B、72B 的 base、instruct 原始 bfloat16 精度模型及其对应的不同精度的量化模型；闭源 API 开放两个 [MoE 模型](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=MoE+%E6%A8%A1%E5%9E%8B&zhida_source=entity)：[Qwen2.5-Turbo](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Qwen2.5-Turbo&zhida_source=entity)（1M即100w上下文）和 [Qwen2.5-Plus](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Qwen2.5-Plus&zhida_source=entity)。
*   **评测**：语言理解、推理、数学、编程、人类偏好对齐等多个基准测试中表现出色。开源旗舰模型 Qwen2.5-72B-Instruct 能跟 Llama-3-405B-Instruct 模型 battle 一下；闭源的俩模型能跟 GPT-4o-mini 和 GPT-4o 做一下pk。
*   **专有模型**：训练了数学模型 [Qwen2.5-Math](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Qwen2.5-Math&zhida_source=entity)、代码模型 [Qwen2.5-Coder](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Qwen2.5-Coder&zhida_source=entity)、慢思考模型 QwQ（应该还有刚发布的多模态模型QvQ）。

模型架构与 Tokenizer
---------------

**稠密模型：**还是 [Transformer Decoder-only](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Transformer+Decoder-only&zhida_source=entity) 的架构，重点改进包括以下几点。

*   用于高效KV缓存利用的**分组查询注意力**（[Grouped Query Attention](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=Grouped+Query+Attention&zhida_source=entity), GQA, Ainslie等人，2023）
*   用于非线性激活的 **SwiGLU** 激活函数（Dauphin等人，2017）
*   用于编码位置信息的**[旋转位置嵌入](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E5%B5%8C%E5%85%A5&zhida_source=entity)**（Rotary Positional Embeddings, RoPE, Su等人，2024）
*   注意力机制中的 QKV 偏置（Su，2023）
*   带有预归一化的 [RMSNorm](https://zhida.zhihu.com/search?content_id=251964885&content_type=Article&match_order=1&q=RMSNorm&zhida_source=entity) 确保稳定训练（Jiang等人，2023b）

![](https://pic4.zhimg.com/v2-b6338f785366ac29d1fdb79c71446617_r.jpg)

**MoE模型**：通过将标准前馈网络（FFN）层替换为专门的MoE层实现，其中每层包括多个FFN专家和一个路由机制，该机制将token分派给前K个专家。按 Qwen1.5-MoE 的方法实现**细粒度专家分割、共享专家路由**。  
  
**Tokenizer**：还是沿用了上一代，带有 **151643** 个常规 token 的**字节级字节对编码（BBPE）**，新版本把 special token 从3个扩到了22个（比如tool call的还有其他能力的）。

```json
{
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>"
    ]
}
```

预训练
---

重点包括：复杂的数据过滤和评分、精细化预训练数据配比、超参数优化、超长文预训练。

*   **预训练数据（共18万亿token）**

*   **数据过滤**：用 Qwen2-Instruct 过滤数据质量，多维度分析评估和打分，保留高质量数据，过滤低质量数据。
*   **数学和代码**：混了 Qwen2.5-Math 和 Qwen2.5-Coder 的训练数据。
*   **合成数据**：用 Qwen2-72B-Instruct 和 Qwen2Math-72B-Instruct 生成合成数学/代码/知识数据，并用内部的通用 RM模型和 Qwen2-Math-RM-72B 过滤。
*   **数据配比**：用Qwen2-Instruct模型对不同领域数据做分类。发现电子商务、社交媒体和娱乐等领域在网络规模数据中显著过度表示，通常包含重复的、基于模板的或机器生成的内容。相反，科技和学术研究等领域虽然包含更高质量的信息，但传统上代表性不足。因此对前者下采样，对后者上采样。

*   **超参数选择**

*   前人用 Scaling Law 一般是确定一定算力下最佳模型大小，但这篇工作也用来确定不同模型（包括MoE）的训练参数，比如**最佳学习率μ和 batch size B 如何随着模型大小N和数据量D的变化而变化**（但是结论没放出来呐！！）
*   实验涵盖 44M~14B稠密模型和44M~1B激活参数的MoE模型，用了 0.8B~600B token 的数据训练。

*   **超长文**

*   **除 Qwen2.5-Turbo 之外的模型**：两阶段训练，初始阶段4096 token，随后跟qwen2一样在第二阶段扩展到 32768。同时用 ABF 技术把 RoPE 基频 base 值从 1w 增加到 100w。
*   **Qwen2.5-Turbo**：逐步上下文扩展策略，32768->65536->131072->262144。RoPE base 是 1000w。每阶段数据配比包括 40% 的当前阶段最大长度序列和 60% 的较短序列。这种渐进式训练方法使得模型能够平滑适应不断增加的上下文长度，同时保持有效处理和泛化不同长度序列的能力。
*   **推理侧两个策略**：YARN 和 Dual Chunk Attention（DCA），把序列长度增加四倍，使 Qwen2.5-Turbo 能处理 100w token，其他模型能处理 131072 token。这些方法降低长文困惑度而且还保持了短文性能，确保不同输入长度的质量一致性。

后训练
---

核心是100w高质量sft以及两阶段RL。

### SFT

针对之前模型局限性的任务，比如长文生成、数学、代码、指令遵循、结构化数据理解、逻辑推理、system prompt，一共 100w+ sft 数据，**序列长度 32768，学习率从 7e-6 逐渐降到 7e-7**。为了防止过拟合，weight decay 为 0.1，梯度范数限制在最大值 1.0。

*   **长文生成**：通过回译（back-translation）从预训练数据反向生成带输出长度限制的长文生成任务 query 并用 Qwen2 过滤低质量数据。训完模型最高生成 8192 token（一般模型大概 2000 token），qwen团队还有一个长文生成的工作也挺有意思（[https://github.com/QwenLM/Self-Lengthen](http://github.com/QwenLM/Self-Lengthen)）。
*   **数学**：用了 Qwen2.5-Math 的 CoT 数据，包括开源数据、K12问题集、合成的问题。用拒绝采样和RM模型指导答案生成。
*   **代码**：用了 Qwen2.5-Coder 的 sft 数据，用特定编程语言的 agent 协作构建 40 种编程语言的 sft 数据，用了代码 Q&A 网站（可能是stackoverflow吧） 和 github 代码片段合成指令，用沙箱做静态代码检查和自动化单元测试来保证质量和正确性。
*   **指令遵循**：用了基于代码的验证框架。LLM生成指令和相应的验证代码，以及全面的单元测试进行交叉验证。通过这种基于反馈结果的拒绝采样，保证指令遵循。
*   **结构化数据**：构建了一个全面的结构化理解数据集，包括传统任务（表格问答、事实验证、错误更正、结构理解）以及涉及结构化和半结构化数据的复杂任务。通过将 CoT 纳入模型的响应中加强模型从复杂数据结构中推理和提取有意义见解的能力。
*   **逻辑推理**：构建 **7w 条**涵盖多领域的query，包括多项选择题、真/假问题和开放式问题。采用一系列推理方法训模型，如演绎推理、归纳概括、类比推理、因果推理和统计推理。通过迭代改进，过滤掉包含错误答案或有缺陷的推理过程的数据。
*   **多语言**：使用翻译模型将高资源语言的指令转换为各种低资源语言，从而生成相应的响应候选。为确保这些响应的准确性和一致性，评估了每种多语言响应与其原始对应数据之间的语义对齐，保留了原始响应的逻辑结构和风格细微差别，从而在不同语言之间保持了它们的完整性和连贯性。
*   **System prompt**：构建了**数百个**通用系统提示，以提高后训练中系统提示的多样性，确保系统提示与对话之间的一致性。使用不同的系统提示进行评估，显示模型保持良好性能和较低方差，表明鲁棒性有所提高。
*   **响应过滤**：用专门的 critic model 和 multi-agent 的评分打标，所有评分系统都觉得没毛病的响应才留下来。

### RL

*   **离线RL**：专注RM模型难评的能力（推理、事实性、指令遵循）精心构建数据训模型，确保离线RL信号可学习且可靠。与在线强化学习（RL）相比，离线RL允许预先准备训练信号，这在有标准答案但使用RM难以评估的任务中特别有利。用SFT模型重新采样输出，通过**人工+自动化质检**的响应被用作直接偏好优化（**DPO**）训练的chosen，未通过作为 reject。构建 **15000 条**数据，用 **Online Merging Optimizer** 训一个epoch，学习率 7e-7。
*   **在线RL**：用RM模型检测输出质量的细微差别，包括“**真实性、有用性、简洁性、相关性、无害性、无偏见**”。使模型生成精确、连贯、结构良好的响应，同时保持安全性和可读性。用了开源和内部更高复杂性的数据，response 用不同阶段的 sft、dpo、grpo 模型用不同 temperature 采样。人工+自动化标注，而且 dpo 数据也纳入其中。在线强化学习框架采用 Group Relative Policy Optimization（**GRPO**），其中训 RM 和训 RL 的数据集一样。训练时优先处理响应分数方差较高的query（让模型更快地学习到那些不确定性较高的情境），其中每个query**采样8个响应**。所有模型都使用 2048 的 global batch size 和每个episode 2048个样本（一对查询和响应）进行训练。

### 超长文

为了进一步扩展Qwen2.5-Turbo的上下文长度，在后训练期间引入了更长的SFT数据。SFT分两阶段，第一阶段跟其他模型一样，**32k上下文**，第二阶段训超长文（**32k混256k**，即最多 262144 token）。在RL阶段使用的训练策略与其它Qwen2.5模型相似，**专注于短指令**。原因有几个：

*   长文RL计算成本很高。
*   缺乏为长文任务提供适当奖励信号的RM。
*   发现仅对短指令采用RL也能显著提高模型在长文对齐。

评测
--

### 基座模型

70B以上的大基座模型评测：

![](https://pic1.zhimg.com/v2-17f9292b56e91b666492a9874825796c_r.jpg)

14B~30B的中基座模型评测：

![](https://pic3.zhimg.com/v2-dd730e12b2ec0291ba149dc7eef398b4_r.jpg)

7B左右基座模型评测：

![](https://pic4.zhimg.com/v2-86bf9c211ec02c17847ccba4f3931239_r.jpg)

更小基座模型评测：

![](https://pic1.zhimg.com/v2-c40547bba6d3d1b72cb06641c37fb33a_r.jpg)

### 指令微调模型

70B+指令模型评测：

![](https://pic4.zhimg.com/v2-cd7c5d6cfce44295d91e28fe8f592625_r.jpg)

14B~30B的指令模型评测（咦，怎么 GPT4o-mini 也在？莫非...）：

![](https://pic1.zhimg.com/v2-6e719b4e5259dcc6c64b1502723631d4_r.jpg)

7B左右及以内指令模型评测：

![](https://pic1.zhimg.com/v2-e96aae132f231233433b9590bc6d050c_r.jpg)

![](https://pica.zhimg.com/v2-e874b0add3065ae18c88d949403e8f3c_r.jpg)

  
内部评测（英文榜）：

![](https://pic4.zhimg.com/v2-cec9abda72a69e7be5b29d65be0a0da3_r.jpg)

  
内部评测（中文榜）：

![](https://pic1.zhimg.com/v2-cdf32465d99d68ebc525c7c32d7780d8_r.jpg)

  
多语言评测：

![](https://pica.zhimg.com/v2-fbbb64de681b0f8baf91406709ce38e4_r.jpg)

### 奖励模型评测

![](https://pica.zhimg.com/v2-cf5f92d7899308dcdfb7c01ddc597d7a_r.jpg)

当前奖励模型评估基准并不能准确预测在其指导下训练的RL模型的性能。换句话说，在RM基准上得分更高并不一定与生成的RL模型的优越性能相关。所以对奖励模型进行更具预测性的评估方法的进一步研究是有必要的。

### 超长文评测

三个榜单评测：

![](https://pic4.zhimg.com/v2-c140167127f9121325bdfffac528d4bf_r.jpg)

大海捞针：

![](https://pic2.zhimg.com/v2-3a4056422aed239a2b6fbb8093903253_r.jpg)

首个 token 生成时间：

![](https://pic3.zhimg.com/v2-885ea53c270f88bb039f4265419005c0_r.jpg)

引入稀疏注意力机制，显著提高了推理速度，对处理长文时用户体验至关重要。对于1M token的序列，这种方法将注意力机制的计算负载减少了12.5倍。图3展示了Qwen2.5-Turbo在不同硬件配置下的首个token生成时间（TTFT），实现了3.2到4.3倍的速度提升。

结论
--

Qwen2.5代表了大型语言模型（LLMs）的重要进步，具有增强的18万亿个token的预训练和复杂的后训练技术，包括监督微调和多阶段强化学习。这些改进增强了人类偏好对齐、长文本生成和结构数据分析的能力，使Qwen2.5非常适合遵循指令的任务。

Qwen2.5提供多种尺寸，包括从0.5B到72B参数的开放权重和包括成本效益高的MoE变体如Qwen2.5-Turbo和Qwen2.5-Plus在内的专有模型。评测表明，Qwen2.5-72B-Instruct的性能与最先进的Llama-3-405B-Instruct相匹配（尽管其规模只有后者的六分之一）。Qwen2.5还作为专业化模型的基础，展示了其在特定领域应用的多功能性。Qwen2.5的稳健性能、灵活架构和广泛可用性使其成为学术研究和工业应用的宝贵资源，定位其为未来创新的关键参与者。在未来，我们将专注于推进稳健的基础模型：

*   纳入更广泛、更多样化、更高质量的数据，迭代改进基础和指令调整的大型语言模型（LLMs）。
*   继续开发多模态模型。将各种模态整合到统一框架，促进文本、视觉和听觉领域的无缝、端到端信息处理。
*   增强模型的推理能力，通过扩展推理计算资源来实现。

本文转自 <https://zhuanlan.zhihu.com/p/14710836610>，如有侵权，请联系删除。