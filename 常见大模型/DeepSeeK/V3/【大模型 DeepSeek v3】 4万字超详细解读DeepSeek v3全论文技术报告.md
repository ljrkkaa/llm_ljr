​

目录

收起

1 前言

1.1 DeepSeek V3 训练资源和费用情况

1.2 贡献

2 分词 BPE（Byte Pair Encoding）和BBPE（Byte-Level Byte Pair Encoding）

2.1 BPE（Byte-Pair Encoding）

2.2 BBPE（Byte-level Byte-Pair Encoding）

3 deepseek v3 结构性创新（MLA+MOE+MTP）

3.1 标准化RMSNorm

3.2 注意力MLA（Multi-head Latent Attention）

3.3 混合专家DeepSeekMoE （Mixture-of-Experts）

3.4 多token预测（MTP，Multi-Token Prediction）

4 deepseek v3 基础设施Infrastructures

4.1 计算集群

4.2 训练框架

4.3 采用FP8 训练

4.4 部署和推理 Inference and Deployment

4.5. 硬件设计建议（Suggestions on Hardware Design，这一节不重要，可以不看 ）

5\. 预训练（Pre-Training）

5.1 数据构建

5.2 超参数模型

5.3 长上下文扩展

5.4 评估（Evaluations）

5.5 讨论（Discussion）

6 后训练（Post-Training ）

6.1 有监督微调（Supervised Fine-Tuning ）

6.2 强化学习

6.3 强化学习后训练实验验证（Evaluations）

6.4 讨论（Discussion）

7 结论、局限性和未来方向

导航栏

1 前言
----

Deepseek v3论文知识点，细节全在这篇文章中，放心食用。[DeepSeek V3](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=DeepSeek+V3&zhida_source=entity) 是由[量化资管](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=%E9%87%8F%E5%8C%96%E8%B5%84%E7%AE%A1&zhida_source=entity)巨头**[幻方量化](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=%E5%B9%BB%E6%96%B9%E9%87%8F%E5%8C%96&zhida_source=entity)**创立的子公司杭州**深度求索人工智能基础技术研究有限公司。**凭借其创新的**架构设计**和高效的**训练策略**，已成为全球AI领域的重要突破。DeepSeek-V3是一个强大的专家混合（[Mixture-of-Experts](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=Mixture-of-Experts&zhida_source=entity)，MoE）语言模型，总共**671B参数**，每个token激活**37B参数**（可以理解为有多个专家，但每个token只会选择一部分专家进行推理，所以一个token的预测，只会用到37B参数），DeepSeek-V3 使用了 **多头潜在注意力（[Multi-head Latent Attention](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=Multi-head+Latent+Attention&zhida_source=entity)， MLA）**和 **DeepSeekMoE（DeepSeek Mixture-of-Experts，MoE）架构**。 DeepSeek-V3开创了采用**专家选择辅助无损失函数（auxiliary-loss-free strategy**，在不实用采样loss帮助下，让每个专家访问次数接近）的负载均衡策略和**多令牌预测（multi-token prediction，每次预测多个token）**训练。采用**[FP8混合精度训练](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=FP8%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AE%AD%E7%BB%83&zhida_source=entity)（FP8 mixed precision，加速训练、推理，减少GPU显存），**设计了**[DualPipe算法](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=DualPipe%E7%AE%97%E6%B3%95&zhida_source=entity)**来实现高效的流水线并行（减少空泡情况，计算和通信同时进行，提升每张显卡的利用率）。对DeepSeek-V3进行了两阶段的上下文长度扩展。在第一阶段，最大上下文长度扩展到32K，在第二阶段，进一步扩展到128K。DeepSeek-V3预训练使用了**14.8万亿**个不同的高质量token训练，然后进行监督微调和强化学习（GRPO）阶段，以使其与人类偏好保持一致，并进一步释放其潜力。

[https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek\_V3.pdfgithub.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek\_V3.pdf](http://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)

[https://github.com/deepseek-ai/DeepSeek-V3github.com/deepseek-ai/DeepSeek-V3](http://github.com/deepseek-ai/DeepSeek-V3)

### **1.1 DeepSeek V3 训练资源和费用情况**

在一张H800情况下，预训练需要训练266万小时，花费532万美元；上下文长度扩展32k和拓展到128k需要训练12万小时，花费23万美元；后训练0.5万小时，花费10万美元。共计278万小时，花费557万美元。实际情况使用了2k多张H800训练了58天。训练十分稳定，没有不可恢复的损失峰值或执行任何回滚操作。

![](https://pic1.zhimg.com/v2-5bd858a0640c3d45939285ffec83bda2_r.jpg)

**司南OpenCompass 大语言模型公开学术榜单（每个token只有37B被激活，参与推理过程中 ）**

![](https://pica.zhimg.com/v2-14ff6f1a267be157ad5ce692c71a482a_r.jpg)

2025年2月

### 1.2 贡献

1.  **创新的负载平衡策略和训练目标（Architecture: Innovative Load Balancing Strategy and Training Objective）**：（1）**开创了一种辅助无损耗技术（auxiliary-loss-free strategy）用于负载平衡的策略**，目的是为了让每个路由专家被选中的概率接近。（2）**多token预测（multi-token prediction）**，有利于一次性预测多个，目的是建立辅助loss，让模型有一次性考虑多个未来token的能力，同时可以用于推理加速的推测解码。
2.  **训练效率提升：**（1）采用FP8混合精度训练。（2）通过算法、框架和硬件的协同设计，克服了通信瓶颈跨节点MoE训练，实现近满计算通信重叠，提高了训练效率。（3）使用了2k多张H800训练了58天，花费557万美元，后训练只需10万美元。
3.  **后训练，DeepSeek-R1 知识蒸馏：**（1）引入了一种创新的方法，创建长链思维（CoT）数据集（具体而言是从 DeepSeek V3和R1中提取），并将其融入标准语言模型（LLM）。
4.  **验证结果：**（1）在许多数据集上超过的公开大模型，接近闭源大模型，中文理解能力强于英文。（2）代码、数学和推理方便，DeepSeek-V3达到了业界领先水平的表现。

2 分词 **BPE（Byte Pair Encoding）**和**BBPE（Byte-Level Byte Pair Encoding）**
------------------------------------------------------------------------

文本在融入模型的第一步，需要把长文本序列分分解成一个一个词汇（可以理解成token，可能一个字、几个字、符号或者其它表示方式），再转换成字典表的索引，送入模型进行编码，模型才能识别。

[https://tiktokenizer.vercel.app/tiktokenizer.vercel.app/](https://link.zhihu.com/?target=https%3A//tiktokenizer.vercel.app/)

### **2.1 BPE（**Byte-Pair Encoding**）**

1.  **初始化词汇表**：将文本中的每个字符作为初始词汇表的一部分。例如，对于文本 `"low lower lowest"`，初始词汇表为：`{'l', 'o', 'w', 'e', 'r', 's', 't'}`。
2.  **统计字符对频率**：统计文本中所有相邻字符对的出现频率。例如，`"low"` 中的字符对为 `('l', 'o')` 和 `('o', 'w')`。
3.  **合并频率最高的字符对**：将频率最高的字符对合并为一个新的子词，并将其添加到词汇表中。例如，如果 `('l', 'o')` 是频率最高的字符对，则合并为 `'lo'`，词汇表更新为：`{'lo', 'w', 'e', 'r', 's', 't'}`。
4.  **重复合并过程**：重复步骤 2 和 3，直到达到预定的词汇表大小或合并次数（下一次合并low）。
5.  **分词**：使用最终的词汇表对文本进行分词。例如，`"lowest"` 可能被分词为 `['low', 'est']`。

### **2.2 BBPE（**Byte-level Byte-Pair Encoding**）**

BBPE是BPE的一种变种，由Google Brain团队提出。**它将BPE从字符级别扩展到字节级别**，将文本中的每个字符转换为UTF-8编码的字节序列，其它流程和BPE一样。比如汉字 “国”，UTF-8 编码对应的字节序列为`0xe5 0x9b 0xbd` 。

**1.以中文句子“好好学习”为例**，用BBPE（Byte-level BPE）进行分词处理，初始化词汇表词汇表初始包含所有单字节（0-255）。将每个字符转换为UTF-8字节序列：

*   “好” → `\xe5\xa5\xbd`
*   “学” → `\xe5\xad\xa6`
*   “习” → `\xe4\xb9\xa0`

整个句子的字节序列：`\xe5\xa5\xbd\xe5\xa5\xbd\xe5\xad\xa6\xe4\xb9\xa0`  
**2\. 统计相邻字节对的频率**：

*   `\xe5\xa5` 出现2次（“好” ×2）
*   `\xa5\xbd` 出现2次（“好” ×2）
*   `\xe5\xad` 出现1次（“学”）
*   `\xad\xa6` 出现1次（“学”）
*   `\xe4\xb9` 出现1次（“习”）
*   `\xb9\xa0` 出现1次（“习”）

**3\. 合并最高频字节**对假设我们先合并频率最高的字节对 `\xe5\xa5` 和 `\xa5\xbd`（因为它们出现2次）：合并后，`\xe5\xa5` 和 `\xa5\xbd` 成为新的词元，词汇表更新。更新后的词汇表：`vocab = ["\x00", "\x01", ..., "\xff", **"\xe5\xa5", "\xa5\xbd"**]` ，加粗的是新加入字典的词汇。

**4\. 重新统计字节对频率：**

*   `\xe5\xa5\xbd` 出现2次
*   `\xe5\xad` 出现1次
*   `\xad\xa6` 出现1次
*   `\xe4\xb9` 出现1次
*   `\xb9\xa0` 出现1次

假设我们继续合并 `\xe5\xa5\xbd`（因为它是新的高频词元）：合并后，`\xe5\xa5\xbd` 成为一个更大的词元，词汇表更新。更新后的词汇表：`vocab = ["\x00", "\x01", ..., "\xff", **"\xe5\xa5", "\xa5\xbd", "\xe5\xa5\xbd"**]`,后面三个就是合并后新增加的词元。

**5\. 重复上面2、3操作**

*   优点：**跨语言通用**：基于字节级别，可以处理多种语言。
*   缺点：**序列长度可能增加**：对于某些语言，可能导致序列变长。

3 deepseek v3 结构性创新（MLA+MOE+MTP）
--------------------------------

![](https://pic3.zhimg.com/v2-2edf26aa04ed5fcd3930d3b8c3217d50_r.jpg)

DeepSeek v3的Trandformer Block MOE+MLA

上图为deepseek的Transformer最小模块，MAL（Multi-head Latent Attention）和MOE（Mixture-of-Experts）分别替换来原始**Attenton层**和**FFN层。再由最小模块多次重复堆叠，搭建出deepseek V3。**

### **3.1 标准化RMSNorm**

**RMSNorm（Root Mean Square Normalization）**的主要思想是通过计算输入向量的均方根（Root Mean Square, RMS），然后使用这个值来归一化输入向量（对每个元素进行缩放）。与LayerNorm等其他归一化方法相比，**RMSNorm不需要计算均值和标准差，**计算上更加高效。有助于稳定和加速深度神经网络的训练过程，特别是在Transformer架构等大规模模型中应用时效果显著。RMSNorm可以表示为： (1)RMSNorm(x)\=x1n∑i\=1nxi2+ϵ⋅γ\\text{RMSNorm}(\\mathbf{x}) = \\frac{\\mathbf{x}}{\\sqrt{\\frac{1}{n}\\sum\_{i=1}^{n}x\_i^2 + \\epsilon}} \\cdot \\gamma \\tag1\\text{RMSNorm}(\\mathbf{x}) = \\frac{\\mathbf{x}}{\\sqrt{\\frac{1}{n}\\sum\_{i=1}^{n}x\_i^2 + \\epsilon}} \\cdot \\gamma \\tag1

*   n 是向量 x 的维度。
*   xix\_i x\_i 表示向量中的第 i 个元素。
*   ϵ \\epsilon \\epsilon 是一个很小的常数（例如 10−810^{-8} 10^{-8} ），目的是为了防止除零错误。
*   γ\\gamma \\gamma 是一个可学习的参数，通常初始化为1。它允许模型在训练过程中学习到最佳的缩放因子。

### 3.2 注意力MLA（Multi-head Latent Attention）

![](https://pic1.zhimg.com/v2-71711518c7fcbb2eff874f92dacbfe6e_r.jpg)

近些年，Attention的模型出现了很多改进版本（GQA，MQA等），但主要思想是，减少KV的数量，达到减少KV Cache显存的使用量，MLA的思想是对KV的纬度降为，实现减少对显存的占用率。**简洁版MLA结构图：**

![](https://pic2.zhimg.com/v2-8c46faf5921b27c771c2faae93619501_r.jpg)

**MLA公式：**

![](https://pic1.zhimg.com/v2-e9eb02f7c07afaf17b60ce0b2f4ead74_r.jpg)

对于注意力机制，DeepSeek-V3采用了MLA架构。设 d 表示嵌入维度， ℎ \_ℎ \_ℎ 表示注意力头的数量， ℎ \_ℎ \_ℎ 表示每个头的维度，ℎ∈ℎ\_ ∈ ℎ\_ ∈ 表示给定注意力层中第 个标记的注意力输入。MLA的核心是通过对注意力键（keys）和值（values）进行低秩联合压缩，以减少推理过程中的键值（KV）缓存。其中， ctKV∈Rdcc^{KV}\_{t}∈R^{d\_c}c^{KV}\_{t}∈R^{d\_c} 是键和值的压缩潜在向量； dc(≪dhnh)d\_c(\\ll {d\_hn\_h})d\_c(\\ll {d\_hn\_h}) 表示（KV）压缩维度； WDKV∈Rdc×dW^{DKV}∈R^{dc×d}W^{DKV}∈R^{dc×d} 是下投影矩阵； WUK,WUV∈Rdhnh×dc​W^{UK},W^{UV}∈R^{d\_hn\_h×d\_c​}W^{UK},W^{UV}∈R^{d\_hn\_h×d\_c​} 分别是键和值的上投影矩阵； WRK∈RdhR×dW^{RK}∈R^{d^R\_h×d}W^{RK}∈R^{d^R\_h×d} 是用于生成带有[旋转位置嵌入](https://zhida.zhihu.com/search?content_id=253619237&content_type=Article&match_order=1&q=%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E5%B5%8C%E5%85%A5&zhida_source=entity)（Rotary Positional Embedding, RoPE）的解耦键的矩阵；RoPE(·) 表示应用RoPE矩阵的操作；\[⋅;⋅\]表示拼接操作。在MLA中，只有蓝色框中的向量需要在生成过程中缓存，这使得KV缓存显著减少，同时保持与标准多头注意力（MHA）相当的性能。

对于注意力查询，也进行低秩压缩，这可以减少训练过程中的激活内存使用。“其中 ctQ∈Rdc′c^Q\_t∈R^{d\_c^′}c^Q\_t∈R^{d\_c^′} 是查询的压缩隐向量；dc′(≪dhnh)d\_c^′(\\ll {d\_hn\_h})d\_c^′(\\ll {d\_hn\_h})表示查询的压缩维度； WDQ∈Rdc′×d​,WUQ∈Rdhnh×dc′W^{DQ}∈R^{d\_c^′×d ​},W^{UQ}∈R^{d\_hn\_h×d\_c ^′}W^{DQ}∈R^{d\_c^′×d ​},W^{UQ}∈R^{d\_hn\_h×d\_c ^′} 分别是查询的下投影矩阵和上投影矩阵； WQR∈RdhRnh×dc′​W^{QR}∈R^{d\_h^R n\_h×d\_c ^′​}W^{QR}∈R^{d\_h^R n\_h×d\_c ^′​} 是生成携带RoPE（旋转位置编码）解耦查询的矩阵。最终，注意力查询（queries, qt,iq\_{t,i}q\_{t,i} ）、键（key, kj,ik\_{j,i}k\_{j,i} ）和值（value, vj,icv^c\_{j,i}v^c\_{j,i} ）结合起来产生最终的注意力输出 utu\_tu\_t ; WO∈Rd×dhnhW^O∈R^{d×d\_hn\_h}W^O∈R^{d×d\_hn\_h} 表示输出投影矩阵。”

**671B代码流程图，流程图是更具官网代码画的：**

![](https://pic3.zhimg.com/v2-385a46e4c8ba5306b77bf9f3c2859756_r.jpg)

**流程看着很复杂，实际很简单（代码越简洁，画图看起来就越复杂）**。输入序列批大小batch\_size=2，序列长度seq\_len=32，字映射纬度toekn\_embedding=7168，n\_heads=128每个特特征向量分解为128个头。**其中对q分支做了低纬映射，由7168映射到1536（q\_lora\_rank=1536），再升纬度24576（128\*（128+64））。对kv分支也做了低纬映射，由7168映射到64和512（kv\_lora\_rank=512），减少KV cache显存使用，其中标上颜色的需要KV cache缓存保存。**q和k都由两部分组成，没有加了旋转位置编码（RoPE）的q和k，会计算一个注意力分数（scores0），加了旋转位置编码（RoPE）的q和k，会计算一个注意力分数（scores1），再把sscores0乘一个softmax\_scale系数加上scores1得到注意力分数（scores），再与kv相乘，最后通过wkv\_b0和wo还原到原始尺寸，**可以看出来kv即作为k的一部分，也作为v**。

**代码：**

```text
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x
```

总结把原始的Multi-head Self-Attention设计成MLA（Multi-head Latent Attention）主要作用：**1、是为了降低了kv-cache显存缓存。2、降低使用降维度减少计算量。3、在注意力分数中引入旋转位置编码，对时序更加敏感。**

[![](https://picx.zhimg.com/v2-cb38e75f8a4fcdbf3e950575e3ecd0c6.png?source=7e7ef6e2&needBackground=1)wenjtop：注意力MHA、MQA、GQA、Linear Attention到MLA45 赞同 · 4 评论 文章](https://zhuanlan.zhihu.com/p/18071594122)

### 3.3 混合专家DeepSeekMoE （Mixture-of-Experts）

**3.3.1 DeepSeekMoE是替换了原始self-attention前向传播，DeepSeekMoE可以理解为在前向传播层有很多专家，每个token会选择一些擅长当前任务专家进行特征处理，（这也就是为什么671B的参数，为什么只有37B激活），这样既能保证准确率，又能减少计算量。其中Experts（专家）分为两组shared expert和routed expert，shared expert：是共享专家，每一个输入都会通过它的计算，671B中n\_shared\_experts=1。routed expert：是路由专家，是根据输入选择性选择topk个(8个)被激活，671B中n\_routed\_experts=256。**

![](https://pic2.zhimg.com/v2-bbf15bf47b244e834114f58fe0f14119_r.jpg)

上图中左边是共享专家，右边是路由专家，下面是公式：

![](https://picx.zhimg.com/v2-b833cc5bac2712632766f36031026c6f_r.jpg)

其中， u\_tu\_t表示输入序列中第t个token， N\_sN\_s 表示共享专家数量， N\_rN\_r 表示路由专家数量， K\_rK\_r 表示被激活的路由专家数， g\_{i,t}g\_{i,t}表示第i个路由专家的第t个token的门阈值； s\_{i,t}s\_{i,t}表示第i个专家与第t个token的亲和度；Topk表示亲和度最高的路由专家， e\_ie\_i 表示token到路由专家映射（全连接层，shape\[dim, 路由专家个数\]）。公式12，有三部分组成：残差结构，多个shared expert和多个routed expert。每一个输入token u\_tu\_t 会通过e\_ie\_i全连接层nn.Linear(input\_dim, n\_routed\_experts)，得到n\_routed\_experts个分数，再挑选前**Topk=8**个routed expert做为当前token的路径。**每一个专家就是三层全连接网络，专家代码：**

```text
dim: int = 7168
inter_dim: int = 18432
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**topk筛选代码（每一个token会选不同的几个routed\_experts）：**

```text
topk=8
n_expert_groups=1
# 返回每一个token需要的路由专家权值和选定的专家指标
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices
```

**3.3.2 辅助无损负载均衡（Auxiliary-Loss-Free Load Balancing）：**

之前的设计会存在一个问题，某些专家会被多次激活，影响负载均衡，所以引入了奖惩值，使每一个专家激活次数接近。设计了辅助无损负载均衡，对于每一个专家引入了一个偏置项 b\_ib\_i ,它会联合亲和度分数 s\_{i,t}s\_{i,t} 去决定当前路由专家是否会被选中。全程监控每个训练步骤的专家负荷，**对于访问概率较低的专家会在偏置项 bb 上增加 ，使该专家有更高的概率被访问。对于访问概率较高的专家会在偏置项 bb 上减去 ，降低该专家选中的概率。** 是一个叫做偏差更新速度的超参数。

![](https://pic3.zhimg.com/v2-bf657bc56aa6c23c613632cd6ce62576_r.jpg)

**3.3.3 互补序列辅助损失（Complementary Sequence-Wise Auxiliary Loss）：**

DeepSeek-V3主要依靠辅助无丢失策略来实现负载平衡，**为了防止任何单个序列内的极端不平衡，还采用了互补序列辅助损失**， \\alpha\\alpha 是平衡因子超参数（极小的值，可学习系数）； f\_if\_i 表示可以理解为在一个序列中每个token访问第i个路由专家的频率占序列长度 TT 的比例； P\_iP\_i 可以理解为在一个序列中每个token访问第i个路由专家的概率均值； 表示序列中令牌的数量， L\_{Bal}L\_{Bal} 考虑在一个序列中，访问所有专家的频率和概率两个因子，**再求损失最小化，**就可以实现在每个序列上的专家负载得到平衡。

![](https://pic1.zhimg.com/v2-0ff22c124330e7df36a9bc692ee54c1e_r.jpg)

**3.3.4 路由专家节点限制：**为了减少节点之前的通信**，确保每个令牌将被发送到最多 节点（M台服务器），K\_rK\_r 表示被激活的路由专家数，根据分布在每个节点上的专家中最高 \\frac{ }{ }\\frac{ }{ } 个亲和度得分的总和来选定的（m=4，k=8，表示选中4个节点，每个节点选中2个路由专家）。**在这个约束条件下，MoE训练框架几乎可以实现完全的计算和通信重叠。

**3.3.5 不丢弃token**。采用了有效的负载均衡策略，DeepSeek-V3 在整个训练过程中能保持良好的负载均衡，不会丢弃任何标记。还实施了特定的部署策略以确保推理负载均衡，所以在推理期间也不会丢弃token。

### **3.4 多token预测（**MTP，**Multi-Token Prediction）**

![](https://pic1.zhimg.com/v2-425c3440dd702e6f89f3f59e8795d312_r.jpg)

为 DeepSeek-V3 设计并制定了一个多token预测（MTP）目标，将预测范围扩展到每个位置的多个后续token。一方面，MTP 增强了训练信号的密度，提高数据效率。另一方面，MTP 可能使模型能够预先规划其表示，从而更好地预测后续token。与之前并行训练不同，MTP是依次预测额外的token，并在每次预测深度保持完整的因果链。**其中每一个MTP模块由一个共享嵌入层Embedding Layer组成，共享输出层OutHead， 一个Transformer block 模块和投影矩阵（ projection matrix，**是指将低维特征映射到高维空间的矩阵，或是指将高维特征映射到低维空间的矩阵**）组成。**深度k表示，p表示预测结果，表示第k个MTP模型，k=0表示主模块（main model）。如果主模块输入t\_1t\_1，主模块会得到预测结果 t\_2t\_2 和预测隐藏表征特征 h^{k=0}\_{i=1}h^{k=0}\_{i=1} ( t\_1t\_1 在第0层的表征特征)，会把这两个值送入到k=1的MTP模块，分别通过RMSNorm直接拼接起来，再通过Transformer block 和OutHead，得到 t\_3t\_3 和 h^{k=1}\_{i=2}h^{k=1}\_{i=2} ，串行送入到k=2的MTP模块得到t\_4t\_4 和 h^{k=2}\_{i=3}h^{k=2}\_{i=3}，串行送入到k=3的MTP模块等，直到最后一层。也就是说输入 t\_1t\_1 会循环预测后面多个token。

![](https://picx.zhimg.com/v2-8a3accf2a56bf64a6afedd78d795fd5f_r.jpg)

抛开btach\_size这个纬度，每次计算loss会做多个序列预测的总和。上面有了输入 t\_1t\_1 预测后面系列token，同时还会输入 t\_1和t\_2t\_1和t\_2 预测后面多个序列token，同时还会输入 t\_1、t\_2和t\_3t\_1、t\_2和t\_3 预测后面多个序列token，同时还会输入 t\_1、t\_2、t\_3和t\_4t\_1、t\_2、t\_3和t\_4 预测后面多个序列token。如上图所示，在代码实现中会同时输入 t\_1、t\_2、t\_3和t\_4t\_1、t\_2、t\_3和t\_4 ，使用mask操作，实现上面描述功能。 L^k\_{MTP}L^k\_{MTP} 表示在深度为k层，序列索引2+k到T+1的所有预测交叉熵损失， L\_{MTP}L\_{MTP} 表示所有深度的 L^k\_{mtp}L^k\_{mtp} 损失的均值。

![](https://picx.zhimg.com/v2-0446a5b5cb83f8236738bd4ad5f6c17f_r.jpg)

4 deepseek v3 基础设施Infrastructures
---------------------------------

### **4.1 计算集群**

计算集群（Compute Clusters ）DeepSeek-V3在配备**2048个NVIDIA H800 GPU**的集群上进行训练。H800集群中每个节点包含**8个GPU**，节点内通过**NVLink**和**NVSwitch**连接（带宽为 160 GB/s）。在不同的节点之间，使用**IB （InfiniBand）**互连来通信（50 GB/s）。

![](https://pic4.zhimg.com/v2-2fe85bf77cd1ea64f93790561a47ca97_r.jpg)

如上图所示，DeepSeek-V3整个训练架构，**TP1（Tensor Parallelism，张量并行）**，用于处理单个操作（如矩阵乘法）所需的数据量过大而无法放入单个GPU或加速器的情况。在张量并行中，一个大的操作被分割成多个较小的部分，并且这些部分可以并行地在不同的设备上执行。DeepSeek-V3采用16路管道并行**PP16 (Pipeline Parallelism，流水线并行，PP，将一个模型分割成16个部分或阶段）**， 跨越8个节点的64路专家并行**EP64（Expert Parallelism，EP，路由专家并行，一共256个专家，每一张卡4个专家，每一个token只会挑选8个路由专家）**，**DP128（Data Parallelism, 数据并行）**表示batch\_size=128，上图一个分支处理2个数据； **ZeRO-1（梯度优化算法策略，减少显存使用）**。在一个阶段中，Attention占用64张、路由专家占用64张卡显卡，所以每个阶段占用128张卡。每一个模型会拆分成16个阶段，所以一共需要**2048**\=128\*16张显卡。

[](https://zhuanlan.zhihu.com/p/11387819228)

### **4.2** 训练框架

DeepSeek-V3的训练由HAI-LLM框架提供支持，该框架是一个高效、有效的算法由工程师从头开始精心制作的轻量级培训框架。实验发现，跨节点之前的通信是主要导致低效训练的原因。通常计算完把数据传输给其它节点计算，再把结果传输到当前节点计算，会导致计算核有一半时间都在等待数据的情况，所以设计了DualPipe算法来实现高效的流水线并行。与现有的PP方法相比，DualPipe具有更少的管道气泡（显卡空闲情况），并且实现，前向和后向进程重叠计算和通信阶段（通信和计算同时进行，不会或少出现等待数据情况），高效的跨节点全对全通信内核。同时充分利用IB和NVLink带宽，节省专用于通信的流多处理器（SMs，Streaming Multiprocessors），实现高效训练。

**4.2.1 双管道和计算通信重叠（DualPipe and Computation-Communication Overlap）**

**对比早期训练方法：**

**（1）、早期模型并行训练方法**，模型才分成4份，GPU0计算的结果，传给GPU2计算，依次前向传播到GPU3，再反向传播，最后一个回到GPU0得到，最后通信到所有卡上更新梯度参数。

![](https://pic3.zhimg.com/v2-20d6b192b10a0bb53157e751148122ea_r.jpg)

**（2）、GPipe：batch数据拆分成多份并行。**

![](https://pic3.zhimg.com/v2-452034a4cb10761106ccacfab41e4f68_r.jpg)

**（3）、前向传播和反向传播交叉计算**

![](https://pic4.zhimg.com/v2-a70d323128e7233e7967427a38ade30f_r.jpg)

**对于DeepSeek-V3**，可以发现上面几种方法，虽有有一定方法缓解气泡率（显卡空闲情况），但显卡还是利用率不高的情况。实验发现，通信开销由跨节点专家并行引入导致低效的计算与通信比率约为1:1。只要合理分配计算和通信，能实现近乎零的全对全通信开销。DualPipe 的核心思想在于将计算与通信在一对独立的前向和后向分块中进行重叠。具体而言，将每个分块划分为四个部分：**attention（计算）, all-to-all dispatch（通信）, MLP（专家计算）, and all-to-all combine（通信）。**

![](https://pic4.zhimg.com/v2-dc6c9bc0c626350b3131b2c1534d8f33_r.jpg)

如图所示，输入首先经过attention计算，再attention的输出分配给其它节点需要的路由专家，对应路由专家进行计算，最后把计算结果，传回给原始节点GPU分支上。（feed forward network就是专家模块），整个过程就可以分为：**attention（计算）, all-to-all dispatch（通信）, MLP（专家计算）, and all-to-all combine（通信）。**

![](https://pica.zhimg.com/v2-8cd3af6e4f82da1fcf605f1f399be732_r.jpg)

**一对单独的向前和向后的重叠策略**。橙色表示前向传播（计算预测结果），绿色表示反向传播（计算梯度），蓝色表示反向权重计算，紫色PP表示通信，红色表示障碍。所有对所有和PP通信都可以完全隐藏。**如上图更这箭头方向（t0时刻，反向mlp计算，前向向路由专家传输数据通信，t1时刻，反向向路由专家传输数据通信，前向路由专家计算结果.....），可以发现每个时刻都同时在通信和计算。双管道调研（DualPipe）**deepseek v3 数据拆分成多份流水线并行，前向和反向同时进行，两个batch错位，第一个btach计算，第二个btach通信。通信和计算可以同时进行，经一步缓解气泡率（显卡空闲情况）。

![](https://picx.zhimg.com/v2-3a6c5366647a565e122fcb7cde4b58bf_r.jpg)

使用实例DualPipe调度8个PP队列和20个微批，两个方向。反向微批与正向微批是对称的，因此为了说明简单，我们省略了它们的批号。由共享的黑色边框包围的两个单元相互重叠计算和通信

![](https://pic2.zhimg.com/v2-02b02cac6bb5cc1ea8167a7a4af37687_r.jpg)

此外，即便是在通信负担不那么沉重的更广泛场景中，DualPipe 仍展现出效率优势。在表中，总结了不同 PP 方法下的管道泡和内存使用情况。如表所示，与 ZB1P和 1F1B相比，DualPipe 显著减少了管道气泡，同时仅使峰值激活显存增加了 \\frac{1}{pp}\\frac{1}{pp} 倍。尽管 DualPipe 需要保留模型参数的两个副本，但这并不会显著增加显存消耗，因为在训练期间使用了较大的专家。与 Chimera相比，DualPipe 只要求管道阶段和微批次能被 2 整除，而不需要微批次能被管道阶段整除。此外，对于 DualPipe 来说，无论是管道泡还是激活内存都不会随着微批次数量的增加而增加。

![](https://pic2.zhimg.com/v2-99edfc92b760f96675740480691db397_r.jpg)

对比不同管道并行方法： 表示一个前向传块的执行时间， 表示一个完整的后向块的执行时间， 表示一个“后向计算梯度”的块的执行时间，而 & 则表示两个相互重叠的前向和后向块的执行时间。

**4.2.2 跨节点全对全通信的高效实现（Efficient Implementation of Cross-Node All-to-All Communication ）**

跨节点的 GPU 通过 IB（InfiniBand）完全互联，而节点内的通信则通过 NVLink（NVIDIA 网络互联）处理。NVLink 提供的带宽为 160 GB/s，大约是 IB（50 GB/s）的 **3.2** 倍。为了有效利用 IB 和 NVLink 的不同带宽，将每个token的调度限制在**最多 4 个节点上（最多只访问4个节点的专家）**，从而减少 IB 流量。**对于每个token，在做出路由决策时，它首先会通过 IB 传输到目标节点上具有相同节点内索引的 GPU 上（比如node2 上的GPU6需要，访问node9上的GPU2，3，4需要，node2 上的GPU6把数据传给node9上的GPU6，再由GPU6通过NVLink传给GPU2，3，4，相比于使用节点之间通信IB分别传送3次，快了2.2=（3.2\*3）/（3.2+1）倍。**一旦到达目标节点，就可以通过 NVLink 立即转发到节点上的多个路由专家，而不被随后到达的token阻塞。这样一来，通过 IB 和 NVLink 进行的通信实现了完全重叠，每个toekn能够高效地为每个节点平均选择 3.2 名专家，而不会因 NVLink 而产生额外的开销。

**4.2.3 为了在训练过程中减少内存占用，采用了以下技术。**

**重新计算 RMSNorm 和 MLA Up-Projection（Up-Projection，是指将低维特征映射到高维空间的过程）**。在反向传播过程中，重新计算所有 RMSNorm 操作和 MLA Up-Projection（是指将低维特征映射到高维空间的过程），从而无需持久保存其输出激活值。通过轻微的开销，这种方法显著降低了存储激活值所需的内存需求。

**在 CPU 中使用指数移动平均值（Exponential Moving Average in CPU）**。**EMA 参数存储在 CPU 内存中，并在每次训练步骤后异步更新。**这种方法使能够维护 EMA 参数，而无需额外的内存或时间开销。在训练过程中，尤其是在学习率衰减之后，EMA参数可以提供一个更稳定、更准确的模型性能估计。**EMA能够平滑参数的波动，避免由于单次训练步骤中的噪声导致的性能波动。**其中，α 是一个接近1的小数，表示衰减因子，决定了新旧参数的权重分配。公式如下：

![](https://pica.zhimg.com/v2-98c2ed8d5a74d42649ea5ee236775ee6_r.jpg)

**共享嵌入和输出头用于多标记预测（Shared Embedding and Output Head for Multi-Token Prediction.）**。通过 DualPipe 策略，将模型的最浅层（包括嵌入层）和最深层（包括输出头）部署在同一个 （Pipeline Parallelism，流水线并行）级别上。这种安排使得多标记预测（MTP）模块与主模型之间能够物理共享嵌入（Embedding Layer）和输出头（Output Head）的参数及梯度。（可以理解为第一层和倒数一层放在一个GPU上，第二层和倒数第二层放在一个GPU上，（Embedding Layer）和输出头（Output Head）的参数及梯度就在同一张GPU上）

### 4.3 采用FP8 训练

**4.3.1 混合精度框架（Mixed Precision Framework）**

**以低精度训练中广泛采用的技术为基础，提出了一种利用FP8数据格式训练DeepSeek-V3的细粒度混合精度框架，减少计算量，显卡（节点）之前的通信量**。虽然低精度训练颇具前景，但其往往受限于激活值、权重和梯度中存在的异常值（主要是精度益处）。为了解决这一挑战并有效地扩大 FP8 格式动态范围，引入了一种精细粒度量化策略：以 1× 元素为单位的分块分组或以 × 元素为单位的块式分组（后面会详细讲解）。与增加精度的累加过程相结合，相关的去量化开销在很大程度上得到了缓解，这是实现准确的 FP8 通用矩阵乘法（GEMM，General Matrix Multiplicatio）的关键方面。此外，为了进一步减少 MoE 训练中的内存和通信开销，在 FP8 中缓存和分发激活值，同时将低精度优化器状态存储在 BF16 中。值得注意的是，与 BF16 基线相比， **FP8 训练模型的相对损失误差始终保持在 0.25%以下**，这是一个在训练随机性可接受范围内的水平。

[](https://zhuanlan.zhihu.com/p/11305727778)

![](https://pic2.zhimg.com/v2-fc9a345019465c9d08c69cc2befce545_r.jpg)

FP8乘FP8涉及到加法操作防止益处使用FP32保存结果。通用矩阵乘法（GEMM，General Matrix Multiplicatio）接受FP8张量作为输入，并在BF16或FP32中产生输出。**与Linear操作符相关的所有三个gem，即Fprop（向前传播）、Dgrad（反向传播梯度）和Wgrad（权重梯度），都在FP8中执行。为什么要求两次梯度，如 y=w\_2f\_2y=w\_2f\_2 ,f\_2=w\_1f\_1f\_2=w\_1f\_1。在反向过程中，分别要对 w\_2w\_2 求导（就是Wgrad，权重梯度）和 f\_2f\_2 求导（Dgrad，反向传播梯度，是为了求后面 w\_1w\_1 和 f\_1f\_1 的导数）。**该设计在理论上使计算速度比原来的BF16方法提高了一倍。此外，FP8 Wgrad gem允许将激活存储在FP8中，以便在反向传递中使用，减少了内存消耗。

尽管FP8格式具有效率优势，但由于对低精度计算的敏感性，某些运算符仍然需要更高的精度。一些低成本的运算器还能以几乎不增加总体训练成本的方式来实现更高的精度。经过仔细研究，对以下组件保持了原有的精度（例如 BF16 或 FP32）：**嵌入模块（embedding module）、输出头（output head）、MoE 门控模块（MoE gating modules，也就是选择路由分数的计算）、归一化运算符（normalization operator）以及注意力运算符（attention operators）。**这些针对高精度的保留措施确保了 DeepSeek-V3 训练过程的稳定性。为了进一步保证数值稳定性，**将主权重、权重梯度和优化器状态存储在更高精度下**。这些高精度组件会产生一些内存开销，可以通过在分布式训练系统中跨多个DP秩的高效分片来最小化它们的影响。

**4.3.2 通过量化和乘法提高精度（Improved Precision from Quantization and Multiplicatio）**

**（1）细粒度的量化（Fine-Grained Quantization）**，为了防止精度溢出（ overflows and underflows ，上益和下益），早期一种标准做法，输入分布会通过将输入张量的最大绝对值缩放至 FP8 格式的最大可表示值来与该格式的可表示范围进行对齐，提出了一种细粒度量化方法来减轻特征异常值带来的量化误差，这种方法使得低精度训练对激活异常值高度敏感，严重降低了量化精度。提出了一种精细粒度的量化方法，该方法在更精细的层面上进行缩放操作。

![](https://pic1.zhimg.com/v2-b050da3d8d0d4057ba3e8a54ab40884a_r.jpg)

如图所示，（1）对于激活值，以 1x128 的小块为单位（即，每个标记对应 128 个通道），对元素进行分组和缩放；（2）对于权重，以 128x128 的小块为单位（即，每个 128 个输入通道对应 128 个输出通道），对元素进行分组和缩放。这种方法确保量化过程能够更好地适应异常值，因为它会根据较小的元素组来调整缩放比例。（**在做矩阵运算，把输入激活值拆分成\[1x128\]，权重才分成\[128x128\]，加载到tensor core中计算，最后在乘上输入的激活值放缩因子和权重放缩因子还原到FP16或者FP32。累加操作会使用到CUDA Core FP32，防止累加精度溢出。**）

![](https://pic2.zhimg.com/v2-79bce44a51148a2a942db0d227959371_r.jpg)

左边输入，右边权重

如上图在做矩阵运算，把输入 激活值拆分成多个\[1x3\]，权重拆分成多个\[3x3\]，加载到tensor core中计算乘法。最后在乘上输入的激活值放缩因子（scale\_i）和权重放缩因子（scale\_w）还原到FP16或者FP32。累加操作会使用到CUDA Core FP32。

**（2）提高积累精度（Increasing Accumulation Precision.）。**低精度的 GEMM （通用矩阵乘法）操作常常会遭遇溢出问题，**其精度很大程度上取决于高精度的累加操作**，**这种操作通常是在 FP32 精度下进行的。**然而，观察到，在 NVIDIA H800 GPU 上，FP8 GEMM 的累加精度被限制在保留约 14 位，这明显低于 FP32 累加精度。当内维度 K 较大时，这个问题会变得更加显著（在大规模模型训练中，典型的场景是批量大小和模型宽度都增大）。**以两个随机矩阵的 GEMM 操作为例，K = 4096 时，在我们的初步测试中，Tensor Cores 中有限的累加精度导致最大相对误差接近 2%。**尽管存在这些问题，但在一些 FP8 框架中，有限的累加精度仍然是默认选项，严重限制了训练精度。

![](https://pic4.zhimg.com/v2-dd50e420e6b29a68955ff6e2551b928f_r.jpg)

**WGMMA：（Warpgroup-level Matrix Multiply-Accumulate，线程组级别的矩阵乘加运算）；Tensor Cores是专门为加速矩阵乘法和累加（Matrix Multiply-Accumulate, MMA）操作设计的专用硬件单元，擅长低精度计算（如 FP16、BF16、INT8）；CUDA Cores是NVIDIA GPU 上的基本计算单元，适用于广泛的并行计算任务，擅长高精度计算（如 FP32、FP64、INT32 等）；Low Prec Acc：低精度加法；FP32 Register：32位寄存器。**为了解决加法溢出问题，采用了针对 CUDA Core的提升精度策略。该过程如图所示。**在 Tensor 核心上执行矩阵乘法累加（MMA，Matrix Multiply-Accumulate）时，中间结果使用有限的位宽进行累加。一旦达到间隔 ，这些部分结果将被复制到 CUDA 核心上的 FP32 寄存器中，进行全精度 FP32 累加。**正如之前所提到的，细粒度量化沿内维度 K 应用每个组的缩放因子。这些缩放因子可以在 CUDA Core上高效地进行乘法运算，作为量化过程，且额外的计算成本极低。这种修改降低了单个warpgroup（GPU上的一个执行单元）的WGMMA（Warpgroup-level Matrix Multiply-Accumulate，线程组级别的矩阵乘加运算）指令发出速率（这句话可以理解为，分块操作让，每个warpgroup在单位时间内能发起的WGMMA指令数量减少了）。在H800架构中，通常可以有两个WGMMA同时存在：当一个warpgroup执行提升操作时（Promotion Operation，数据格式转换将数据从较低精度转换为较高精度的过程），另一个warpgroup能够执行MMA（Matrix Multiply-Accumulate，矩阵乘法加法）操作。这种设计使得这两种操作可以重叠进行，保持Tensor Cores的高利用率。根据实验，设置 _NC_​=128 个元素（小块矩阵单位长度），相当于4个WGMMA，代表了最小累积间隔可以显著提高精度，而无需引入大量的开销。

**（3）整数和位数（Mantissa over Exponents）**与之前Nvidia工作中采用的混合FP8格式，在前向传播（Fprop）中使用E4M3（4位整数和3位小数），在反向传播中的梯度计算（Dgrad）和权重梯度计算（Wgrad）中使用E5M2（5位整数和2位小数），deepseek在所有张量上采用E4M3格式以获得更高的精度。认为这种方法的可行性归功于**细粒度量化策略**，即分块和块状缩放。通过在较小的元素组上操作，能有效地共享这些组内元素的指数位，从而缓解了有限动态范围的影响。

**（4）在线量化（Online Quantization）**延迟量化这种方法通过维护先前迭代中的最大绝对值历史来推断当前值，为了确保准确的缩放因子并简化框架，在线计算每个1x128激活块或128x128权重块的最大绝对值。基于此推导出缩放因子，并将激活或权重在线量化为FP8格式。**目的是有效求出缩放因子**。

**4.3.3 低精度存储和通信（ Low-Precision Storage and Communication ）**

结合FP8训练框架，通过将缓存的激活和优化器状态压缩成较低精度的格式，进一步减少内存消耗和通信开销。

**（1）低精度优化器状态**。采用 BF16数据格式而非 FP32（单精度浮点数 32 位）来追踪 AdamW优化器中的一阶矩和二阶矩，且不会造成可察觉的性能下降。不过，主权重（由优化器存储）以及梯度（用于批量大小累加）仍保留为 FP32 格式，以确保整个训练过程中的数值稳定性。参考下面文章

[](https://zhuanlan.zhihu.com/p/11305727778)

**（2）低精度激活值。权重梯度（Wgrad）运算采用 FP8（8 位浮点数）格式来执行**。为了减少内存消耗，对于线性算子的反向传播过程而言，**将激活值缓存为 FP8 格式**。不过，为了实现低成本的高精度训练，针对若干算子需要进行特殊考量。**注意力算子之后线性层的输入：**这些激活值也会在注意力算子的反向传播过程中被使用，这使得它们对精度非常敏感。专门为这些激活值采用了定制的 **E5M6** 数据格式。此外，在反向传播过程中，这些激活值将从 1x128 的量化块转换为 128x1 的块。为了避免引入额外的量化误差，所有的缩放因子都采用舍入缩放，即 2 的整数次幂。**混合专家（MoE）架构中 SwiGLU 算子的输入：**为了进一步降低内存成本，缓存 SwiGLU 算子的输入，并在反向传播过程中重新计算其输出。这些激活值也使用细粒度量化方法以 FP8 格式存储，从而在内存效率和计算精度之间取得平衡。

**（3）低精度通信。**通信带宽是混合专家（MoE）模型训练中的一个关键瓶颈。为了缓解这一挑战，在 MoE 上投影（up-projections， 是指将低维特征映射到高维空间的过程）之前将激活值量化为 FP8 格式，并应用与MoE上投影中的FP8前向传播（Fprop）兼容的调度组件。与注意力算子之后线性层的输入类似，此激活值的缩放因子为 2 的整数次幂。类似的策略也应用于 MoE 下投影（down-projections， 是指将高维特征映射到低维空间的过程）之前的激活梯度。对于前向和反向合并组件，将它们保留为 BF16 格式，以在训练流程的关键部分保持训练精度。

### 4.4 部署和推理 Inference and Deployment

将DeepSeek-V3部署在H800集群上，其中每个节点内的gpu使用NVLink互连，并且整个集群中的所有gpu通过IB完全互连。为了同时确保在线服务的服务水平目标（SLO）和高吞吐量，采用以下部署策略，将**预填充（prefilling）和解码（decoding）阶段分开，预填充阶段理解问题，把过程激活参数传给，解码部分回答问题。**

**预填充（prefill，理解问题）：**计算机密集型操作，问题使用并行计算，可以理解成读题，推理第一个token，一次性计算出所有的kv Cache缓存，直接传给其它节点，主要用于推理第一个token。TTFP计算首个token的时间。**预填充阶段的最小部署单元由 4 个节点组成，一共使用32张卡H800，其中attention采用4路张量并行(TP4，4路模型并行)和序列并行（SP），8路数据并行（batch\_size 8数据并行），Moe采用32路专家并行（EP32，一共256个路由专家，一张卡8个专家）。使用32个冗余专家，动态调整。**对于 MoE 的全连接通信，采用与训练时相同的方法：首先通过 IB 在节点间传输标记，然后通过 NVLink 在节点内的 GPU 之间转发。**在浅层的密集 MLP 中使用 1 路张量并行来节省 TP 通信。**为了在 MoE 部分的不同专家之间实现负载均衡，确保每个 GPU 处理的令牌数量大致相同。引入了冗余专家的部署策略，即复制高负载专家并对其进行冗余部署。高负载专家是根据在线部署期间收集的统计数据进行检测的，并且会定期（例如每 10 分钟）进行调整。确定冗余专家集合后，会根据观察到的负载情况，在节点内的 GPU 之间仔细重新安排专家，力求在不增加跨节点全对全通信开销的情况下尽可能均衡 GPU 之间的负载。对于 DeepSeek-V3 的部署，**在预填充阶段设置了 32 个冗余专家。**对于每个 GPU，除了它所托管的原始 8 个专家外，还将额外托管一个冗余专家。每次推理步骤中只有 9 名专家会被激活。实时计算出全局最优的路由方案。

**解码（decode，推理答案）：**访问密集型，逐生成token，也就是生成答案部分，生成第二个toekn开始，会频繁访问kv Cache缓存。计算后面每个token的时间。在解码过程中，每个令牌在路由过程中都会选择 9 个专家（一个共享专家和8个路由专家），共享专家会始终被选中。**解码阶段的最小部署单元由 40 个节点和 320 个 GPU 组成。注意力部分采用 TP4 与 SP 结合 DP80，而 MoE 部分使用 EP320。对于 MoE 部分，每个 GPU 仅承载一个专家，64 个 GPU 负责承载冗余专家和共享专家。**调度和组合部分的所有到所有通信通过 IB 上的直接点对点传输来实现低延迟。有效解决低解码时延，缓解负载不均衡问题。与预先填充类似，会基于我们在线服务的统计专家负载，在一定的时间间隔内定期确定冗余专家的集合。此外，为了提高吞吐量并隐藏所有对等通信的开销，还在解码阶段同时处理两个具有相似计算工作负载的微批次。与预填充不同，在解码阶段，注意力消耗的时间比例更大。因此将一个微批次的注意力与另一个的**分发+MoE+回传进行**重叠。在解码阶段，每个专家的批次大小相对较小（通常在 256 个标记以内），瓶颈在于内存访问而非计算。由于 MoE 部分只需要加载一个专家的参数，内存访问开销极小，因此使用更少的流多处理器（SMs，Streaming Multiprocessors）不会显著影响整体性能。因此为了避免影响注意力部分的计算速度，可以仅分配一小部分流多处理器（SMs，Streaming Multiprocessors）来**分发+MoE+回传（dispatch+MoE+combine）。**

### 4.5. 硬件设计建议（Suggestions on Hardware Design，这一节不重要，可以不看 ）

**4.5.1. 通信硬件（Communication Hardware）**

在 DeepSeek - V3 中，实现了计算与通信的重叠，以在计算过程中隐藏通信延迟。与串行的计算和通信方式相比，这显著降低了对通信带宽的依赖。然而，当前的通信实现依赖于昂贵的流式多处理器（SMs）（例如，为了实现这一目的，在 H800 GPU 可用的 132 个 SMs 中分配了 20 个），这会限制计算吞吐量。此外，使用 SMs 进行通信会导致显著的效率低下，因为张量核心会完全未被充分利用。

目前，流多处理器（SMs，Streaming Multiprocessors）主要执行以下任务，用于all-to-all通信：

*   **传输数据：**在同一节点内，从单个 GPU 聚合发往多个 GPU 的 InfiniBand（IB）流量，同时在 IB 和 NVLink 域之间转发数据。
*   **转换数据：**在远程直接内存访问（RDMA）缓冲区（已注册的 GPU 内存区域）和输入 / 输出缓冲区之间传输数据。
*   **执行回传数据操作：**专家计算完结果，回传到原始GPU操作all-to-all combine）。
*   **管理细粒度内存布局：**在分块数据通过IB和NVLink传输到多个领域专家。

希望看到未来的供应商开发能够将这些通信任务从宝贵计算单元SM（Streaming Multiprocessor）上卸载下来的硬件，作为GPU协处理器或类似NVIDIA SHARP Graham等人（2016）提出的网络协处理器。此外，为了减少应用程序编程的复杂性，希望硬件能够从计算单元的角度统一InfiniBand（横向扩展）和NVLink（纵向扩展）网络。通过这种统一接口，计算单元可以通过提交基于简单原语的通信请求，在整个IB-NVLink统一域内轻松完成读取、写入、多播和归约等操作。这样不仅提高了计算效率，还简化了大规模并行计算环境中的编程模型。

**3.5.2. 计算硬件（Compute Hardware ）**

**张量核中更高的FP8 GEMM积累精度：**在当前NVIDIA Hopper架构的Tensor Core实现中，FP8 GEMM（通用矩阵乘法）使用定点积累，即在加法前通过右移对齐最大指数的尾数积。实验显示，它仅使用每个尾数积经过符号填充右移后的最高14位，并截断超出此范围的位数。然而，例如，为了从32次FP8×FP8乘法的累积中获得精确的FP32结果，至少需要34位的精度。因此，建议未来的芯片设计应增加Tensor Core中的累积精度，以支持全精度累积，或者根据训练和推理算法的精度要求选择合适的累积位宽。这种方法确保了误差保持在可接受范围内，同时维持计算效率。通过提高累积阶段的精度，可以更准确地捕捉数值计算过程中的细节，从而提升模型训练和推理的整体性能与准确性。

**乘法阶段：**每次FP8×FP8的乘法会产生一个中间结果，这个结果理论上应该是比FP8更精确的格式（例如，至少需要FP16或更高精度），但由于当前实现中仅保留了最高14位的尾数积，并且进行了右移操作，导致信息丢失。

**累积阶段：**将这32个中间结果相加得到最终的FP32结果。理想情况下，为了确保最终结果的准确性，需要至少34位的累积精度来避免舍入误差和其他数值不稳定性。但是，由于实际操作中只用了14位进行累积，这可能导致累积结果的误差超出可接受范围，进而影响模型的准确性和性能。

**支持Tile和Block级别的量化**。当前的GPU仅支持每个张量（per-tensor）量化，缺乏对更细粒度量化（如我们的tile和block级别量化）的原生支持。在现有实现中，当达到 区间时，部分结果将从Tensor Cores复制到CUDA核心，乘以缩放因子，并加到CUDA核心上的FP32寄存器中。尽管结合精确的FP32累积策略，解量化开销显著减少，但Tensor Cores与CUDA核心之间频繁的数据移动仍然限制了计算效率。因此，建议未来的芯片设计应通过允许Tensor Cores接收缩放因子并使用组缩放实施矩阵乘累加（MMA）操作来支持更细粒度的量化。这种方式使得整个部分和累积及解量化过程可以直接在Tensor Cores内部完成，直到生成最终结果，从而避免频繁的数据移动。这种方式不仅提高了计算效率，还减少了由于数据在不同计算单元间传输带来的延迟和能耗。此外，它还能更好地支持高效、大规模并行计算的需求，特别是在深度学习模型训练和推理过程中，对于需要高精度和低延迟的应用场景尤为重要。通过这样的改进，可以进一步提升硬件平台的整体性能和灵活性，满足日益增长的高性能计算需求。

**支持在线量化。**尽管研究表明在线量化非常有效，但当前的实现难以有效地支持这一技术。在现有的过程中，需要从HBM（高带宽内存）读取128个BF16激活值（上一次计算的输出）进行量化，然后将量化的FP8值写回HBM，仅为了再次读取这些值用于矩阵乘累加（MMA）操作。为了解决这种低效问题，建议未来的芯片设计应将FP8转换和TMA（张量内存加速器）访问整合为一个单一的融合操作，使得量化可以在激活值从全局内存传输到共享内存的过程中完成，从而避免频繁的内存读写操作。此外，还建议支持warp级别的转换指令以加速处理，这将进一步促进层归一化与FP8转换的更好融合。另一种方法是采用近内存计算（near-memory computing），即将计算逻辑放置在靠近HBM的位置。在这种情况下，BF16元素可以直接在从HBM读入GPU时转换为FP8，这样可以减少大约50%的外部内存访问次数。通过这些改进措施，不仅可以显著提高计算效率，还能降低由于频繁的内存读写带来的延迟和能耗。特别是对于深度学习模型中的大规模并行计算需求，这种方法能够更好地满足高性能计算的要求，并且有助于提升整体硬件平台的性能和灵活性。同时，减少外部内存访问也有助于降低系统功耗，延长设备的使用寿命。

**支持转置GEMM操作。**当前的架构在融合矩阵转置与GEMM（通用矩阵乘法）操作时显得非常繁琐。在我们的工作流程中，前向传播过程中的激活值被量化为1x128的FP8块并存储。在反向传播过程中，需要从内存中读取这些矩阵，进行解量化、转置、重新量化为128x1的块，并存储回HBM（高带宽内存）。为了减少内存操作，建议未来的芯片设计应支持直接从共享内存中以转置方式读取矩阵，然后再进行MMA操作，这对于训练和推理所需的精度都非常重要。结合FP8格式转换与TMA（张量内存加速器）访问的融合，这种增强将显著简化量化工作流程。具体来说：（1）直接转置读取：在进行MMA操作之前，允许直接从共享内存中读取转置后的矩阵，避免了先读取再转置的传统步骤，从而减少了不必要的内存读写操作。（2）格式转换与TMA访问的融合：通过将FP8格式转换与TMA访问融合为单一操作，可以在数据从全局内存传输到共享内存的过程中完成量化或解量化，进一步减少了内存操作次数。这些改进措施不仅能够提高计算效率，还能显著降低由于频繁的内存读写带来的延迟和能耗。特别是在深度学习模型的训练和推理过程中，这种方法可以更好地满足大规模并行计算的需求，同时提升硬件平台的整体性能和灵活性。此外，减少外部内存访问还有助于降低系统功耗，延长设备的使用寿命。这样，不仅可以实现更高效的计算，还能确保数值稳定性和结果的准确性。

5\. 预训练（Pre-Training）
---------------------

整个训练分为三个阶段，：预训练，序列扩张和后训练：

![](https://pic4.zhimg.com/v2-7186bddd9df4e6e7be170b25df69d017_r.jpg)

### 5.1 数据构建

与DeepSeek-V2相比，DeepSeek-V3在优化预训练语料库时增加了数学和编程样本的比例，并扩展了多语言覆盖范围，超出了英语和中文的范畴。此外，改进了数据处理管道，以最小化冗余同时保持语料库的多样性。实施了文档打包方法以确保数据完整性，**但在训练过程中没有引入跨样本注意力掩码。**最终，DeepSeek-V3的训练语料库包含14.8万亿个高质量且多样化的标记。

在DeepSeekCoder-V2（DeepSeek-AI，2024a）的训练过程中，观察到“中间填充”（Fill-in-Middle, FIM）策略不仅不会损害模型的下一个标记预测能力，还能使模型根据上下文线索准确预测中间文本。为了与DeepSeekCoder-V2保持一致，在DeepSeek-V3的预训练中也纳入了FIM策略。具体来说，使用前缀-后缀-中间（Prefix-Suffix-Middle, PSM）框架来组织数据，**训练采用率设定为 0.1**，**意思是通过前文和后文的内容预测中间的内容，fim\_begin表示序列开始； f\_{pre}f\_{pre} 表示序列前面部分；fim\_bole表示文章中间； f\_{suf}f\_{suf} 表示序列后面部分；fim\_end表示序列结束； f\_{middle}f\_{middle} 表示序列中间需要预测的部分；eos\_token表示整个预测结束**。结构如下：。

![](https://picx.zhimg.com/v2-3d1b6e1a6f6195122abef6bd20e2271b_r.jpg)

**DeepSeek-V3 的分词器采用了基于字节级别的 BPE技术，并且词汇表扩展到了 128K 个标记。**分词器的预分词器和训练数据经过了修改，以优化多语言压缩效率。此外，与 DeepSeek-V2 相比，新的预分词器引入了结合标点符号和换行符的标记。然而，这种技巧在模型处理没有终端换行符的多行提示时可能会引入标记边界偏差，特别是在针对少样本评估提示时。为了解决这个问题，在训练期间随机拆分一定比例的此类组合标记，这使模型接触到更广泛的特殊情况，并减轻了这种偏差。

### 5.2 超参数模型

```text
# DeepSeek-V3 16B
{
    "vocab_size": 102400,
    "dim": 2048,
    "inter_dim": 10944,
    "moe_inter_dim": 1408,
    "n_layers": 27,
    "n_dense_layers": 1,
    "n_heads": 16,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "n_activated_experts": 6,
    "route_scale": 1.0,
    "q_lora_rank": 0,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mscale": 0.707
}

# DeepSeek-V3 236B
{
    "vocab_size": 102400,
    "dim": 5120,
    "inter_dim": 12288,
    "moe_inter_dim": 1536,
    "n_layers": 60,
    "n_dense_layers": 1,
    "n_heads": 128,
    "n_routed_experts": 160,
    "n_shared_experts": 2,
    "n_activated_experts": 6,
    "n_expert_groups": 8,
    "n_limited_groups": 3,
    "route_scale": 16.0,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128
}

# DeepSeek-V3 671B
{
    "vocab_size": 129280,
    "dim": 7168,
    "inter_dim": 18432,
    "moe_inter_dim": 2048,
    "n_layers": 61,
    "n_dense_layers": 3,
    "n_heads": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "n_activated_experts": 8,
    "n_expert_groups": 8,
    "n_limited_groups": 4,
    "route_scale": 2.5,
    "score_func": "sigmoid",
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "dtype": "fp8"
}
```

**模型超参数：将 Transformer 层的数量设为 61 层（n\_layers），隐藏维度设为 7168（dim）。所有可学习参数都以标准差 0.006 的方式随机初始化。在多层潜在注意力（MLA）机制中，我们将注意力头的数量设为 128（n\_heads），每个头的维度设为 128（v\_head\_dim）。KV 压缩维度设为 512（kv\_lora\_rank），查询压缩维度设为 1536（q\_lora\_rank）。对于解耦的查询和密钥，将每个头的维度替换为除了前三个层之外的所有 FFN（前向传播神经网络）使用 MoE（多专家）层。每个 MoE 层由 1 个共享专家和 256 个路由专家组成**，**每个专家的中间隐藏维度为 2048（moe\_inter\_dim）。**在路由专家中，**每个令牌将激活 8 个专家**，**并且每个令牌将确保最多被发送到 4 个节点**。**多令牌预测深度 设为 1，即除了确切的下一个令牌之外，每个令牌还将预测一个额外的令牌。**与 DeepSeek-V2 相同，DeepSeek-V3 还在压缩的潜在向量之后使用额外的 RMSNorm 层，并在宽度瓶颈处乘以额外的缩放因子。在这种配置下，DeepSeek-V3 总共包含 671 亿个参数，其中每个标记激活的参数数量为 37 亿个。

训练超参数。采用 AdamW 优化器（Loshchilov 和 Hutter，2017 年），其超参数设置为 1 = 0.9、 2 = 0.95 以及权重衰减率为 0.1。在预训练期间，将最大序列长度设定为 4K，并使用 14.8T 个标记对 DeepSeek-V3 进行预训练。至于学习率调度方面，首先在前 2K 步中将其从 0 线性提升至 2.2 × 10−4 。然后，将学习率保持在 2.2 × 10−4 的恒定值直至模型消耗 10T 训练令牌。随后按照余弦衰减曲线在 4.3T 令牌内逐渐将学习率降低至 2.2 × 10−5 。在最后 500B 令牌的训练过程中，在前 333B 令牌中保持学习率为 2.2 × 10−5 ，在剩余的 167B 令牌中切换到另一个恒定的学习率为 7.3 × 10−6 。梯度裁剪范数设置为 1.0 。采用批次大小调度策略，在前 469B 令牌的训练中，批次大小从 3072 逐渐增加到 15360 ，然后在剩余的训练中保持 15360 。利用管道并行性将模型的不同层部署在不同的 GPU 上，并且对于每一层，路由专家将均匀部署在属于 8 个节点的 64 个 GPU 上。至于节点限制的路由，每个令牌最多会被发送至 4 个节点（即 = 4）。对于无辅助损失的负载均衡，对前 14.3T 令牌将偏差更新速度 设置为 0.001 ，对剩余的 500B 令牌将其设置为 0 。对于平衡损失，将 设为 0.0001，仅仅是为了避免在任何单个序列中出现极端的不平衡情况。MTP 损失权重 对于前 10T 个标记设为 0.3，对于剩余的 4.8T 个标记设为 0.1。

### 5.3 长上下文扩展

DeepSeek-V3采用与DeepSeek-V2（DeepSeek-AI，2024c）类似的方法，以在DeepSeek-V3中实现长上下文能力。在预训练阶段之后，应用了YaRN（Peng等人，2023a）进行上下文扩展，并进行了两个额外的训练阶段，每个阶段包含1000步，逐步将上下文窗口从4K扩展到32K，然后再扩展到128K。YaRN配置与DeepSeek-V2中使用的一致，仅应用于解耦共享键 k\_t^Rk\_t^R ​。在这两个阶段中，超参数保持一致，具体为：尺度 = 40, = 1, = 32，以及缩放因子 \\sqrt{t}=0.1ln(s)+1\\sqrt{t}=0.1ln(s)+1 。**在第一阶段，序列长度设置为32K，批量大小为1920。在第二阶段，序列长度增加到128K，批量大小减少到480。**两个阶段的学习率均设置为7.3 × 10⁻⁶，与预训练阶段的最终学习率相匹配。通过这一双阶段扩展训练，DeepSeek-V3能够处理长度达128K的输入，同时保持强大的性能表现。证明了其在上下文窗口长度达到128K时的一致稳健性。

### 5.4 评估（Evaluations）

**5.4.1 评价基准（Evaluation Benchmarks）**

DeepSeek-V3的基本模型是在英语和中文占多数的多语言语料库上进行预训练的，因此在主要以英语和中文为基准的一系列基准以及多语言基准上评估其性能，数据集包含：

*   多学科选择题数据集包括（Multi-subject multiple-choice）：MMLU（2020）；MMLU-Redux（2024）；MMLU-Pro（2024b）；MMMLU（2024b）；C-Eval（2023）；CMMLU（2023）。
*   语言理解和推理数据集包括（Language understanding and reasoning）：HellaSwag（2019）；PIQA（2020）；ARC（2018）；BigBench Hard (BBH，2022）
*   闭卷问答数据集包括（Closed-book question answering）：TriviaQA（2017）；NaturalQuestions（2019）；
*   阅读理解数据集包括（Reading comprehensio）：RACE（2017）；DROP（2019）；C3（2019a）；CMRC（2019）
*   指代消解数据集包括（Reference disambiguatio）：CLUEWSC（2020）；WinoGrande（2019）
*   语言模型数据集包括（Language modeling）：The Pile（2020）
*   中文理解和文化数据集包括（Chinese understanding and culture）：CCPM（2021）
*   数学数据集包括（Math）：GSM8K（2021）；MATH（2021）；MGSM（2023）；CMath（2023）
*   代码数据集包括（Code）：HumanEval（2021）；LiveCodeBench-Base (0801-1101，2024）；MBPP（2021）；CRUXEval（2024）
*   标准化考试数据集包括（Standardized exams）：AGIEval（2023）。

**5.4.2 验证结果（Evaluation Results ）**

![](https://pic2.zhimg.com/v2-63c5aaaafaf760807f14ae689a8801bb_r.jpg)

如上表，将DeepSeek-V3的基础模型与最先进的开源基础模型进行了比较，包括之前发布的DeepSeek-V2-Base（DeepSeek-AI，2024c）、Qwen2.5 72B Base（Qwen，2024b）和LLaMA-3.1 405B Base（AI@Meta，2024b）。使用内部评估框架对所有这些模型进行了评估，并确保它们共享相同的评估设置。DeepSeek-V3-Base在各方面全面超越了DeepSeek-V2-Base和Qwen2.5 72B Base，并在大多数基准测试中超过了LLaMA-3.1 405B Base，**基本上成为最强的开源模型。**

### **5.5** 讨论（**Discussion）**

**5.5.1 多标记预测的消融研究（Ablation Studies for Multi-Token Prediction）**

![](https://pic3.zhimg.com/v2-0bf584f39e9c2b17f6b4ae38342a2650_r.jpg)

表中展示了MTP策略的消融实验结果。具体来说，在不同规模的两个基线模型上验证了MTP策略。在小规模实验中，训练了一个包含157亿总参数的基线MoE模型，使用了1.33万亿个令牌的数据进行训练。在大规模实验中，我们训练了一个包含2287亿总参数的基线MoE模型，使用了5400亿个令牌的数据进行训练。在这两个基线模型的基础上，保持训练数据和其他架构不变，附加了一个深度为1的MTP模块，并训练了两个使用MTP策略的模型以进行比较。需要注意的是，在推理过程中，直接丢弃了MTP模块，因此对比模型的推理成本是完全相同的。从表中可以看出，MTP策略在大多数评估基准上一致地提升了模型的性能。

**5.5.2 辅助无损失函数平衡策略的消融研究（Ablation Studies for the Auxiliary-Loss-Free Balancing Strateg）**

![](https://pic4.zhimg.com/v2-8f9e0d97dde999cb61e501f75c89bf85_r.jpg)

在表中，展示了无辅助损失平衡策略的消融实验结果。在不同规模的两个基线模型上验证了这一策略。在小规模实验中，训练了一个包含157亿总参数的基线MoE模型，使用了1.33万亿个令牌的数据进行训练。在大规模实验中，训练了一个包含2287亿总参数的基线MoE模型，使用了5780亿个令牌的数据进行训练。这两个基线模型纯粹依赖辅助损失来促进负载均衡，并使用带有top-K亲和度归一化的sigmoid门控函数。它们控制辅助损失强度的超参数分别与DeepSeek-V2-Lite和DeepSeek-V2相同。在这两个基线模型的基础上，保持训练数据和其他架构不变，移除了所有辅助损失，并引入了无辅助损失平衡策略以进行比较。

**6 后训练（Post-Training ）**
-------------------------

### **6.1 有监督微调（Supervised Fine-Tuning ）**

对指令调优数据集进行了管理，使其包含跨越多个域的150万个实例，每个域使用针对其特定需求量身定制的不同数据创建方法；对于涉及推理相关的数据集，包括专注于数学、编程竞赛问题和逻辑谜题的数据集，通过使用内部的DeepSeek-R1模型生成数据。具体来说，虽然R1生成的数据表现出较高的准确性，但也存在一些问题，如过度思考、格式不佳和内容过长。目标是平衡R1生成的推理数据的高准确性和常规格式推理数据的清晰性和简洁性。建立方法论首先通过结合监督微调（SFT）和强化学习（RL）的训练流程，开发一个针对特定领域的专家模型，如代码、数学或一般推理。这个专家模型将作为最终模型的数据生成器。训练过程包括为每个实例生成两种不同类型的SFT样本：

1.  **第一种样本**：将问题与其原始答案配对，格式为<问题, 原始答案>。
2.  **第二种样本**：在问题和R1生成的答案中加入系统提示，格式为<系统提示, 问题, R1答案>。

系统提示经过精心设计，包含引导模型生成富含反思和验证机制响应的指令。在RL阶段，模型使用高温度采样生成结合了R1生成数据和原始数据模式的响应，即使没有明确的系统提示也是如此。经过数百个RL步骤后，中间的RL模型学会整合R1模式，从而战略性地提升整体性能。完成RL训练阶段后，采用拒绝采样来为最终模型精选高质量的SFT数据，其中专家模型被用作数据生成源。这种方法确保最终训练数据保留了DeepSeek-R1的优势，同时生成简洁有效的响应。

**非推理数据：**对于非推理数据，如创意写作、角色扮演和简单问答，我们使用DeepSeek-V2.5生成响应，并聘请人工标注员验证数据的准确性和正确性。

**SFT设置：**在SFT数据集上对DeepSeek-V3-Base进行两个epoch的微调，使用余弦衰减学习率调度，初始学习率为5 × 10⁻⁶，逐渐降低到1 × 10⁻⁶。在训练过程中，每个序列由多个样本打包而成，但采用样本掩码策略，以确保这些示例保持隔离且相互不可见。

**拒绝采样在大模型训练的流程：**背景已经有一个监督微调（SFT）的模型、质量不高的数据集（可以是前面监督微调模型的训练数据集，也可以新爬取的数据集）和奖励模型（reward model）：

1.  **生成模型回答**：使用经过监督微调（SFT）的模型，对这些质量不高的数据中的问题进行解答，从而得到模型生成的答案（answer）。
2.  **答案打分**：利用奖励模型，分别对数据里原本的答案和模型生成的答案进行评分（比如考虑正确率、逻辑清晰度等）。
3.  **更新训练数据**：选取得分排名第一（或前几）的答案，替换质量不高数据中的原始答案，进而生成新一版的训练数据。
4.  **模型再训练**：使用新生成的训练数据，对步骤 1 中用于生成答案的模型进行训练。
5.  **循环迭代**：**重复执行步骤 1 - 4**，即模型针对 SFT 数据生成回答、对答案打分、用得分第一（或前几）的答案替换得到新版数据、使用新数据训练模型，如此循环往复。

### 6.2 强化学习

**6.2 .1 奖励模型（Reward Model ）**

在强化学习过程中采用了一个基于规则的奖励模型（RM）和一个基于模型的奖励模型（RM）。

**基于规则的奖励模型（Rule-Based RM**）**：**对于可以使用特定规则验证的问题，我们采用基于规则的奖励系统来确定反馈。例如，某些数学问题有确定的结果，我们要求模型在指定格式（如在一个方框内）提供最终答案，这样我们可以应用规则来验证其正确性。同样地，对于LeetCode上的问题，我们可以利用编译器根据测试用例生成反馈。通过尽可能使用基于规则的验证方法，我们确保了更高的可靠性，因为这种方法不易受到操纵或利用。

**基于模型的奖励模型（Model-Based RM）：**对于有自由形式标准答案的问题，我们依赖于奖励模型来判断回答是否符合预期的标准答案。相反，对于那些没有明确标准答案的问题，比如涉及创意写作的问题，奖励模型则负责根据问题及其对应的答案作为输入提供反馈。该奖励模型是从DeepSeek-V3 SFT检查点训练得到的。为了增强其可靠性，我们构建了不仅提供最终奖励还包括导向该奖励的思考链的偏好数据。这种方法有助于减少在特定任务中出现奖励作弊的风险。

**6.2.2 GRPO（Group Relative Policy Optimization）**

**参考这篇文章有详细介绍：**

[](https://zhuanlan.zhihu.com/p/22051002772)

### 6.3 强化学习后训练实验验证（Evaluations）

**6.3.1. 标准验证（Standard Evaluation ）**

**代码任务**：在工程任务中，DeepSeek-V3落后于Claude-Sonnet-3.5-1022，但显著优于其他开源模型。在算法任务中，DeepSeek-V3表现出色，特别是在HumanEval-Mul和LiveCodeBench等基准测试中超越了所有基线。其成功主要归功于**先进的知识蒸馏技术**，增强了代码生成和问题解决能力。

**数学基准测试**：DeepSeek-V3在AIME、MATH-500和CNMO 2024等数学基准测试中表现出色，比第二好的Qwen模型高出约10%。这一成就得益于从DeepSeek-R1继承的知识蒸馏技术，特别适用于non-o1-like模型。

**标准评估**：表6显示DeepSeek-V3是表现最好的开源模型，并且在与前沿闭源模型的竞争中表现出色。

**英语基准测试**：在MMLU、MMLU-Pro、MMLU-Redux和GPQA-Diamond等基准测试中，DeepSeek-V3表现出色，与顶级模型并驾齐驱，甚至在某些情况下超越它们。在长上下文理解基准测试如DROP、LongBench v2和FRAMES中，DeepSeek-V3展示了其在处理长上下文任务方面的强大能力。在事实知识基准SimpleQA上，DeepSeek-V3的表现稍逊于GPT-4o和Claude-Sonnet，但在中文知识方面表现出色。在指令遵循基准测试中，DeepSeek-V3显著优于其前身，显示出改进的理解和遵守用户定义格式约束的能力。

**中文基础测试：**Qwen和DeepSeek是两个具有代表性的模型系列，对中英文都有强大的支持。在实际基准中文SimpleQA上，DeepSeek-V3超过Qwen2.5- 72b 16.4分，尽管Qwen2.5是在一个更大的语料库上训练的，涉及18T个令牌，比DeepSeek-V3预训练的14.8T令牌多20%。在中国教育知识评价的代表性标杆C-Eval和中国Winograd模式挑战（CLUEWSC）上，DeepSeek-V3和Qwen2.5-72B表现出相似的性能水平，表明这两个模型都对cha进行了很好的优化

![](https://pic1.zhimg.com/v2-176d9208fdd4c15a1678ce95c7cacad0_r.jpg)

**6.3.2. 开放验证（Open-Ended Evaluation ）**

DeepSeek-V3在多种开放生成任务评估中表现出色，不仅在复杂任务（如编码和调试）中展现了强大的能力，还在简单任务上实现了显著改进。通过在Arena-Hard和AlpacaEval 2.0等基准测试中的优异表现，DeepSeek-V3为开源模型在挑战性领域的应用树立了新的标杆，并显著缩小了与闭源前沿模型的性能差距。

![](https://pic4.zhimg.com/v2-b39c8d2dc0504ceba58d7e23d3faf29f_r.jpg)

英语开放式会话评价。对于AlpacaEval 2.0，使用长度控制的胜率作为度量。

**5.3.4. 作为生成奖励模型的DeepSeek-V3（DeepSeek-V3 as a Generative Reward Model ）**

对比了DeepSeek-V3与最先进模型（即GPT-4o和Claude-3.5）的判断能力。下表展示了这些模型在RewardBench（Lambert等人，2024）中的表现。DeepSeek-V3的表现与最佳版本的GPT-4o-0806和Claude-3.5-Sonnet-1022相当，同时超越了其他版本。此外，DeepSeek-V3的判断能力还可以通过投票技术进一步增强。因此，使用DeepSeek-V3结合投票技术来对开放性问题提供自我反馈，从而提高对齐过程的有效性和鲁棒性。

![](https://picx.zhimg.com/v2-9a5bda9a4bea0b0a09789a50212328c7_r.jpg)

### **6.4 讨论（Discussion）**

**6.4.1 来自DeepSeek-R1的蒸馏(Distillation from DeepSeek-R1)**

基于DeepSeek-V2.5对来自DeepSeek-R1的知识蒸馏的贡献进行了消融研究。基线模型是在短CoT（Chain of Thought）数据上训练的，而其竞争对手使用的是由上述专家检查点生成的数据。**展示了蒸馏数据的有效性**，在LiveCodeBench和MATH-500基准测试中均显示出显著的改进。实验揭示了一个有趣的权衡：**蒸馏带来了更好的性能，但也显著增加了平均响应长度**。为了在模型准确性和计算效率之间保持平衡，为DeepSeek-V3在蒸馏过程中精心选择了最优设置。研究表明，从推理模型中进行知识蒸馏是后训练优化的一个有前景的方向。尽管目前的工作集中在从数学和编码领域蒸馏数据，但这种方法显示出在各种任务领域中的广泛应用潜力。在这些特定领域中展示出的有效性表明，长CoT蒸馏可能对于提升其他需要复杂推理的认知任务中的模型性能非常有价值。在未来的研究中，进一步探索这一方法在不同领域的应用仍然是一个重要方向。

**6.4.2 自我奖励(Self-Rewarding )**

奖励在强化学习（RL）中起着至关重要的作用，指导优化过程。在一些可以通过外部工具进行验证的领域，如某些编码或数学场景，RL展示了卓越的有效性。然而，在更一般的场景中，通过硬编码构建反馈机制是不切实际的。在开发DeepSeek-V3的过程中，对于这些更广泛的上下文，采用了宪法AI方法（Bai等人，2022），利用DeepSeek-V3自身的投票评估结果作为反馈来源。这种方法产生了显著的对齐效果，显著提升了DeepSeek-V3在主观评价中的表现。通过整合额外的宪法输入，DeepSeek-V3可以朝着宪法方向进行优化。这种结合补充信息与LLM（大型语言模型）作为反馈来源的范式至关重要。LLM作为一种多功能处理器，能够将来自不同场景的非结构化信息转化为奖励，最终促进LLM的自我改进。除了自我奖励之外，还致力于发现其他通用且可扩展的奖励方法，以持续提升模型在一般场景中的能力。

**6.4.3 多token预测验证（Multi-Token Prediction Evaluation ）**

与其仅预测下一个单个标记，DeepSeek-V3通过多标记预测（MTP）技术预测接下来的两个标记。结合推测解码框架（Leviathan等人，2023；Xia等人，2023），这种方法可以显著加速模型的解码速度。一个自然的问题是关于额外预测的标记的接受率。根据我们的评估，在各种生成主题中，第二个标记预测的接受率在85%到90%之间，显示出一致的可靠性。这种高接受率使得DeepSeek-V3能够显著提高解码速度，达到每秒1.8倍的令牌处理速度（Tokens Per Second, TPS）。

**7 结论、局限性和未来方向**
-----------------

在这篇论文中，介绍了DeepSeek-V3，一个拥有671B总参数和37B激活参数的大型MoE（Mixture of Experts）语言模型，训练于14.8T个令牌。除了MLA和DeepSeekMoE架构外，它还开创了一种无辅助损失的负载均衡策略，并设定了一个多标记预测训练目标以实现更强的性能。由于支持FP8训练和细致的工程优化，DeepSeek-V3的训练成本效益高。后训练阶段成功地从DeepSeek-R1系列模型中提炼了推理能力。全面评估表明，DeepSeek-V3已经成为目前最强的开源模型，其性能可与领先的闭源模型如GPT-4o和Claude-3.5-Sonnet相媲美。尽管性能强大，但其训练成本仍然经济高效，仅需2.788M H800 GPU小时即可完成包括预训练、上下文长度扩展和后训练在内的全流程。

尽管承认其强大的性能和成本效益，也认识到DeepSeek-V3在部署方面存在一些局限性。首先，为了确保高效的推理，推荐的DeepSeek-V3部署单元相对较大，这对小型团队可能是一个负担。其次，尽管DeepSeek-V3部署策略已经实现了比DeepSeek-V2快两倍以上的端到端生成速度，但仍有一定的改进空间。幸运的是，随着更先进硬件的发展，这些局限性有望自然得到解决。DeepSeek始终坚持长期主义的开源模型路线，旨在稳步接近通用人工智能（AGI）的最终目标。未来，计划在以下几个方向进行战略性研究投资：

1.  **将持续研究和优化模型架构，旨在进一步提高训练和推理效率，努力实现对无限上下文长度的支持。**
2.  **将尝试突破Transformer架构的限制，从而拓展其建模能力。**
3.  **将持续迭代训练数据的数量和质量，并探索引入额外的训练信号来源，旨在推动数据规模在更多维度上的扩展。**
4.  **将持续探索和迭代模型的深度思考能力，通过扩展其推理长度和深度来增强其智能和问题解决能力。**
5.  **将探索更全面和多维度的模型评估方法，以防止在研究过程中倾向于优化固定的一组基准，这可能会产生误导性的模型能力印象并影响我们的基础评估。**

  

导航栏
---

[](https://zhuanlan.zhihu.com/p/400628805)

**如果有疑问，建议，错误，欢迎留言大家指正。。。**

**更新不易啊，欢迎点赞关注。。。**

**内容还在持续更新中。。。**

本文转自 <https://zhuanlan.zhihu.com/p/22985118565>，如有侵权，请联系删除。