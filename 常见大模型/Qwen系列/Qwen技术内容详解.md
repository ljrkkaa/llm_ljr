本文转自 [https://blog.csdn.net/weixin_49659123/article/details/140180512](https://blog.csdn.net/weixin_49659123/article/details/140180512)，如有侵权，请联系删除。

一、引言
--------

今日份LLM详解是介绍另一个中文开源大模型——阿里的通义千问基座大模型Qwen。

本文基于Qwen的[技术报告](https://so.csdn.net/so/search?q=%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A&spm=1001.2101.3001.7020)，详解了Qwen从预训练到RLHF对齐的技术内容，并增加一些技术详解，力求cover全貌的过程中尽量地解释一些重要的细节🥸 （至于这个度，只能说尽量把控）

---

技术报告：[https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN\_TECHNICAL\_REPORT.pdf](https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf "https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf")

[Github](https://so.csdn.net/so/search?q=Github&spm=1001.2101.3001.7020)：[https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen "https://github.com/QwenLM/Qwen")

技术报告中介绍了整个Qwen系列的模型，以及使用SFT和RLHF对齐训练的模型（Qwen-Chat，以及其改进版本Qwen-Chat-RLHF）。

此外，阿里还发布了专门的编码和数学模型，如Code-Qwen, Code-Qwen-Chat和基于Qwen的数学模型Math-Qwen-Chat。

除此之外，还有多模态LLM, Qwen-VL和 Qwen-VL-Chat。

![](https://i-blog.csdnimg.cn/direct/7e6f849032be4958a6fe2ef62321a6d9.png)

本篇内容主要介绍Qwen、Qwen-Chat、Qwen-Chat-RLHF 的预训练、对齐训练技术细节

---

_Note：由于Code模型和Math模型暂时没有开源，多模态Qwen-VL模型本身有自己的论文，本次分享对三种模型就不做介绍了_

二、预训练
----------

预训练阶段包括学习大量数据，以获得对世界及其各种复杂性的全面理解。这不仅包括基本的语言能力，还包括算术、编码和逻辑推理等高级技能。

### 2.1 预训练数据

基座模型Qwen使用了高达**3万亿个token**的数据进行预训练，数据主要涉及公共网络文档、百科全书、书籍、代码等，**数据涉及多语言，但以中文和英文为主**。

为了保证预训练数据的质量，研究团队制定了一套全面的数据预处理程序：

* **文本数据抓取** 对于Web数据，从HTML中提取文本内容
* **语言识别：** 采用语言识别工具确定语种
* **去重：** 为了增加数据的多样性，采用了重复数据删除技术，包括语言规范化(Normalization)后的精确匹配重复数据删除和使用MinHash和LSH算法的模糊重复数据删除；
* **质量过滤：** 结合规则和机器学习的方法过滤低质量数据，即通过多个模型对内容进行评分，包括语言模型、文本质量评分模型
* **安全控制：** 使用模型识别并过滤涉及暴力、偏见、色情等潜在冒犯性内容
* **Up-sample采样：** 针对某些高质量源的数据进行上采样，以确保多样化的高质量内容。
* **引入指令数据：** 引入高质量的指令数据
* **防止Data leakage：** 为了消除了与评估中使用的测试集中存在的任何数据重叠的问题，删除了任何与测试集中指令样本存在超过13-gram的文本内容

通过这些技术手段,QWEN从原始数据中提炼了高达3万亿token的高质量的预训练数据,为模型提供了可靠的知识来源。

### 2.2 分词器（Tokenizer）

词表大小影响者模型的训练效率和下游任务效果，Qwen的分词(Tokenization)采用的是基于BPE(Byte Pair Encoding)的方法，兼顾中文和英文等多语言场景的效率，主要步骤如下:

1. 首先，基于开源分词器tiktoken的cl100k基础词表进行初始化。
2. 然后，针对中文场景,向词表中增添常用的中文字和词,扩充词表规模。
3. 同时，参考GPT-3.5和LLaMA的实现，将数字切分成单个数字（如将"123"分词为"1"、"2"、"3"），最终词表大小约为152K。

> 将数字分成各个数字(digit)的原因主要与模型的训练效率和下游任务的表现有关：
>
> * **增强模型的泛化能力**：通过将数字拆分成单个数字，模型可以更好地理解和处理不同的数字组合，而不仅仅是记住特定的数字。这种方法提高了模型在处理新数字和未见过的数字组合时的泛化能力。
> * **减少词汇表的大小**：数字通常有无数种组合形式，如果每个数字都作为一个单独的词汇存在于词汇表中，会显著增加词汇表的大小。而通过拆分数字，可以有效地减少词汇表的大小，从而提高模型的训练效率和推理速度。

下图展示了 Qwen tokenizer的压缩性能。将 Qwen 与其他几种tokenizers进行了评估，包括 XLM-R、LLaMA、Baichuan和InternLM。

研究结果表明，Qwen在大多数语言中都取得了比竞争对手更高的压缩效率（表现为更低的compression ratio），这意味着服务成本可以显著降低，因为来自Qwen的token数量较少，可以比其竞争对手传递更多的信息。

此外，研究团队还进行了初步实验，以确保缩放Qwen的词汇量不会对[预训练模型](https://so.csdn.net/so/search?q=%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)的下游性能产生负面影响。尽管实验的词汇量有所增加，但实验表明Qwen在下游评估中保持了其性能水平。

### 2.3 模型架构

Qwen采用了改进版的 Transformer架构，具体改进如下：

* Embedding and output projection：使用**Untied Embedding，**即对于输入嵌入层和输出投影层不进行权重共享，是两个单独的权重。

> #### Untied Embedding
>
> * **Untied Embedding**指的是输入嵌入（input embedding）和输出投影（output projection）使用独立的权重矩阵，而不是共享相同的权重（tied weights）。
> * **原因：**基于初步的实验结果，**使用独立的权重矩阵（untied weights）可以获得更好的性能**。这可能是因为**输入嵌入和输出投影在功能上有不同的需求，使用独立的权重可以更好地满足各自的需求**，从而提高模型的整体表现。独立的权重矩阵允许输入嵌入和输出投影有更多的灵活性，可以更好地适应不同的任务需求，提高模型的准确性和性能。
> * **代价**: 采用untied embedding方法的主要代价是**增加了内存消耗**。因为输入嵌入和输出投影使用独立的权重矩阵，这意味着需要存储两个不同的矩阵，从而增加了模型的内存需求。
> * **选择untied embedding的原因**: 为了在性能和内存消耗之间找到平衡点，基于实验结果，选择untied embedding方法可以显著提升模型的性能，尽管会增加内存消耗。"

* 使用了RoPE（旋转位置编码）进行位置编码。RoPE在当代大型语言模型中已被广泛采用，比如 PaLM 和 LLaMA。为了优先考虑模型性能并获得更高的精确度，使用 FP32 精确度的逆频率矩阵，而不是 BF16 或 FP16。

> #### RoPE
>
> ##### **使用FP32的原因**:
>
> * **逆频率矩阵的作用**: 在RoPE中，位置编码依赖于**逆频率矩阵（inverse frequency matrix）来计算位置向量的旋转**。位置向量的旋转通过旋转矩阵实现，而这个旋转矩阵又依赖于位置和频率之间的精确关系。
> * **高精度计算**: 如果逆频率矩阵的精度不够高，会导致位置编码不准确。这会直接影响模型对序列中相对位置关系的理解，从而影响模型在处理位置敏感任务时的性能。为了优先考虑模型的性能和获得更高的精确度，选择了FP32精度。这种高精度有助于减少数值计算中的误差，提高模型的稳定性和准确性。

* 在大多数层中移除了Bias，但在QKV层保留以提升模型的外推能力。

> #### **移除大多数层的偏差项**
>
> * 根据Chowdhery等人的研究，移除大多数层的偏差项有助于**提高模型的稳定性和性能**。这种做法可以**减少参数数量**，从而**降低模型的复杂度和过拟合的风险**。此外，移除偏差项**有助于模型在训练过程中更好地泛化**，从而在不同的数据集上表现得更为一致。
>
> #### **在注意力机制的QKV层中添加偏差项**
>
> * Su的研究指出，在注意力机制的QKV（Query，Key，Value）层中添加偏差项可以增强**模型的外推能力**（extrapolation ability）。QKV层是注意力机制的核心部分，通过添加偏差项，可以更好地捕捉输入数据的特征，从而提升模型在处理未知或未见数据时的表现。这种增强外推能力的做法对于处理复杂任务和应对多样化的数据输入非常重要。
>
> ---
>
> _总结来说，移除大多数层的偏差项是为了提高模型的泛化能力和稳定性，而在QKV层中添加偏差项则是为了增强模型的外推能力。这两种做法的结合旨在优化模型的整体性能。_

* 使用了前归一化（Pre-Norm）进行规范化。

> #### Pre-Norm（前归一化）
>
> 在Pre-Norm结构中，层归一化是在每个子层（如多头注意力和前馈网络）之前应用的。
>
> **优点**：
>
> 1. **稳定性**：Pre-Norm有助于在训练早期阶段稳定梯度，因为归一化是在每个子层的输入之前进行的，可以缓解梯度消失和梯度爆炸的问题。
> 2. **收敛性**：通常情况下，Pre-Norm结构收敛速度较快，因为归一化使得每个子层的输入具有较好的分布，促进了模型的有效学习。
>
> **缺点**：
>
> 1. **深层网络的训练**：在非常深的网络中，Pre-Norm可能会导致网络中的信息传递不够充分，因为归一化会削弱信息的传递。
>
> #### Post-Norm（后归一化）
>
> 在Post-Norm结构中，层归一化是在每个子层（如多头注意力和前馈网络）之后应用的。
>
> **优点**：
>
> 1. **信息传递**：Post-Norm结构在处理深层网络时表现更好，因为它可以更好地保留和传递子层中的信息，有助于捕捉复杂的特征和模式。
> 2. **性能提升**：在某些情况下，Post-Norm可以提高模型的最终性能，特别是对于需要捕捉长程依赖关系的任务。
>
> **缺点**：
>
> 1. **训练难度**：Post-Norm结构在训练初期可能会面临梯度消失或梯度爆炸的问题，因为归一化是在子层之后进行的，可能导致梯度不够稳定。
> 2. **收敛速度**：相较于Pre-Norm，Post-Norm结构的收敛速度通常较慢，需要更多的训练步骤和调整超参数来稳定训练过程。
>
> #### 比较与选择
>
> 选择Pre-Norm还是Post-Norm取决于具体的应用场景和模型需求：
>
> 1. **收敛速度和训练稳定性**：如果优先考虑训练的收敛速度和稳定性，Pre-Norm可能是更好的选择。
> 2. **深层模型和信息传递**：对于需要训练非常深的模型或需要捕捉复杂依赖关系的任务，Post-Norm可能更合适。
>
> ---
>
> _在现代Transformer模型中，_Pre-Norm_是最广泛使用的方法，与_Post-Norm_相比，_Pre-Norm_已被证明可以提高训练稳定性；最近的研究提出了更好的训练稳定性的替代方法，Qwen的研究团队计划在模型的未来版本中进行探索。_

* 使用RMSNorm而不是Layer Norm来进行规范化。

> #### Layer Normalization（层归一化）
>
> Layer Normalization 是在模型的每一层对其输入进行归一化处理的技术。它的主要公式为：
>
> ![](https://i-blog.csdnimg.cn/direct/1e2f69185f4c4945a95c6cc0c1626a71.png)
>
> **优点**：
>
> 1. **稳定性**：通过对每一层的输入进行归一化，可以缓解梯度消失和梯度爆炸的问题，从而提高模型的稳定性。
> 2. **适用性广泛**：Layer Normalization 在各种神经网络结构中表现良好，尤其适用于循环神经网络（RNNs）和变压器（Transformers）。
> 3. **无批量依赖**：与批归一化（Batch Normalization）不同，Layer Normalization 不依赖于批量数据的大小，这使得它在小批量甚至单样本情况下依然有效。
>
> **缺点**：
>
> 1. **计算复杂度**：Layer Normalization 需要计算均值和标准差，这在计算上相对复杂。
> 2. **对特征缩放敏感**：有时会对特征缩放较为敏感，需要额外调整超参数。
>
> #### RMS Normalization（RMS归一化）
>
> RMS Normalization 是一种通过均方根（Root Mean Square）值对输入进行归一化的方法。其主要公式为：
>
> ![](https://i-blog.csdnimg.cn/direct/574a0375f1904c32829ef493aaba8d38.png)
>
> **优点**：
>
> 1. **计算效率高**：RMS Normalization 的计算比 Layer Normalization 更简单，因为它只需计算均方根值，而不需要计算均值和标准差。
> 2. **稳定性好**：RMS Normalization 也能缓解梯度消失和梯度爆炸的问题，提升模型的训练稳定性。
> 3. **适用性**：特别适用于需要简化计算的模型，如某些大规模预训练模型。
>
> **缺点**：
>
> 1. **信息损失**：由于 RMS Normalization 只考虑了均方根值，可能会丢失一些与输入分布相关的信息。
>
> ---
>
> _由于LLM的参数量之大以及预训练规模之大，RMS Normalization 计算更为简单，适合在需要高效计算的大规模预训练模型。_

* 使用了SwiGLU作为激活函数；并按照以往研究中的常见做法，将前馈网络（FFN）的维度从隐藏大小的 4 倍降至隐藏大小的8/3。

> #### SwiGLU
>
> SwiGLU（SwiGLU Activation）是一种改进的激活函数，专为Transformer架构中的前馈神经网络（FFN）设计。它结合了Swish和Gated Linear Units（GLU）的优点，提供了更强的表达能力和更好的性能。
>
> ![](https://i-blog.csdnimg.cn/direct/529dfd6b3188453e8cb5baf779983ead.png)
>
> * ##### Swish
>
> Swish激活函数是对于ReLU激活函数的改进，其公式如下：
>
> ![Swish=x\cdot sigmoid(\beta x)](https://latex.csdn.net/eq?Swish%3Dx%5Ccdot%20sigmoid%28%5Cbeta%20x%29)
>
> 其中，ß 为可学习参数。Swish可以比ReLU激活函数更好，因为它在0附近提供了更平滑的转换（见上图），这可以带来更好的优化。
>
> * ##### Gated Linear Unit (GLU)
>
> GLU（Gated Linear Unit）激活函数是一种用于神经网络的激活函数，它具有门控机制，可以帮助网络更好地捕捉序列数据中的长期依赖关系。其公式如下：
>
> ![GLU(x,W,V,b,c)=sigmoid(xW+b)\bigotimes (xV+c)](https://latex.csdn.net/eq?GLU%28x%2CW%2CV%2Cb%2Cc%29%3Dsigmoid%28xW+b%29%5Cbigotimes%20%28xV+c%29)
>
> ![\bigotimes](https://latex.csdn.net/eq?%5Cbigotimes)表示逐元素相乘
>
> 在该公式中，左侧部分有一个Sigmoid函数，把这个函数替换为其他函数就可以得到GLU函数的各种变体；如替换成线性函数、ReLU函数等，把sigmoid替换成swish就得到了SwiGLU函数：
>
> ![SwiGLU(x,W,V,b,c)=swish(xW+b)\bigotimes (xV+c)](https://latex.csdn.net/eq?SwiGLU%28x%2CW%2CV%2Cb%2Cc%29%3Dswish%28xW+b%29%5Cbigotimes%20%28xV+c%29)
>
> #### 为什么选择SwiGLU
>
> * Swish对于负值的响应相对较小克服了 ReLU 某些神经元上输出始终为零的缺点
> * GLU 的门控特性，这意味着它可以根据输入的情况决定哪些信息应该通过、哪些信息应该被过滤。这种机制可以使网络更有效地学习到有用的表示，有助于提高模型的泛化能力。在大语言模型中，这对于处理长序列、长距离依赖的文本特别有用。
> * SwiGLU 中的参数 W1,W2,W3,b1,b2,b3可以通过训练学习，使得模型可以根据不同任务和数据集动态调整这些参数，增强了模型的灵活性和适应性。
> * 计算效率相比某些较复杂的激活函数（如 GELU）更高，同时仍能保持较好的性能。这对于大规模语言模型的训练和推理是很重要的考量因素。
>
> Rreference:[https://mingchao.wang/1fb1JNJ6/![icon-default.png?t=N7T8](https://i-blog.csdnimg.cn/blog_migrate/003a2ce7eb50c2e24a8c624c260c5930.png)https://mingchao.wang/1fb1JNJ6/](https://mingchao.wang/1fb1JNJ6/ "https://mingchao.wang/1fb1JNJ6/")_笔者Note：这篇参考文章也详细解释了为什么把FFN的维度从4h降到了8/3h_

根据以上的修改，最终的到的不同大小的模型的隐藏维度、注意力头数量、层数如下图所示：

![](https://i-blog.csdnimg.cn/direct/a59ef67478504b04a8dce9dc5a48da8d.png)

---

_笔者Note：由于这部分大模型构架的改动点都非常具有代表性，因此笔者在这里详细解释了这些技术的what/why_

### 2.4 模型训练

* 采用标准的自回归语言模型训练目标
* 训练时上下文长度为2048
* 注意力模块采用Flash Attention技术，以提高计算效率并减少内存使用

> Flash Attention请参考：
>
> [通透理解FlashAttention与FlashAttention2：全面降低显存读写、加快计算速度-CSDN博客文章浏览阅读2.8w次，点赞167次，收藏333次。因此，可以确认：在 MQA 中，除了 query 向量还保存着 8 个头，key 和 value 向量都只剩 1 个「公共头」了，这也正好印证了论文中所说的「所有 head 之间共享一份 key 和 value 的参数」所以，上面讲到计算注意力的主要瓶颈是显存访问，因此减少对HBM的读写次数，有效利用更高速的SRAM来进行计算是非常重要的，而GPU有大量的线程来执行某个操作，称为。(需要注意的是，模型训练通常会影响到算子融合的效果，因为为了后向传递计算梯度，通常需要将某些中间结果写入到HBM中)\_flashattention![](https://i-blog.csdnimg.cn/blog_migrate/be19846480ab44ce477585fc567aeaa0.png)https://blog.csdn.net/v\_JULY\_v/article/details/133619540](https://blog.csdn.net/v_JULY_v/article/details/133619540 "通透理解FlashAttention与FlashAttention2：全面降低显存读写、加快计算速度-CSDN博客")

* 采用AdamW优化器，设置β1=0.9，β2=0.95,，=1e-8
* 使用余弦学习率计划，为每种模型设定一个峰值学习率，学习率会衰减到峰值的10%
* 使用BFloat16混合精度加速训练

###  2.5 **外推能力扩展**

众所周知，Transformer模型在其注意机制的上下文长度方面有很大的限制。随着上下文长度的增加，二次复杂度计算导致计算和内存成本的急剧增加。为了解决这个问题，Qwen选择在Inference阶段实现上下文长度的扩展 （而不是训练阶段，因为在训练阶段扩展上下文会导致训练成本的急速增长）。

由于Qwen使用的位置编码是RoPE，虽然RoPE相较于绝对位置编码具有可外推的优点，但是在直接使用外推的过程中还是存在一些问题；因此Qwen针对这些问题，来进行了改进。

#### 2.5.1 RoPE的问题：直接外推会出现比较大的Attention Score

RoPE（相对位置编码）使用正弦和余弦函数将位置信息嵌入到词汇向量的旋转矩阵中。然而，由于以下原因，RoPE直接外推会导致Attention Score显著增加：

1. **正弦和余弦函数的周期性**：

   * 正弦和余弦函数是周期性的，周期为![2\pi](https://latex.csdn.net/eq?2%5Cpi)。在训练数据中，位置通常在一个相对较小的范围内（例如，0到512或0到2048），这些位置的编码值会保持在周期的某一部。
   * 当位置超出这个范围时（例如，位置变为3000或3500），编码值会进入正弦和余弦函数的另一个周期。由于这些函数的周期性，这些位置的编码值可能与训练数据中的编码值非常不同，导致模型在计算注意力分数时出现剧烈变化。
2. **高频成分的影响**：

   * 在RoPE编码中，较高维度的编码（即频率较高的正弦和余弦成分）会对较大的位置变化更加敏感。这意味着，随着位置数值的增加，这些高频成分会迅速变化。
   * 对于较大的位置值，正弦和余弦函数的值可能会经历快速变化。这种快速变化会导致Attention机制中查询向量和键向量的点积（即Attention Score）出现显著波动。

#### 2.5.2 NTK-aware 插值 & Dynamic NTK插值

为了解决以上RoPE直接外推产生的Attention score过大，以及RoPE嵌入插值时丢失高频信息(losing high frequency information when interpolating the RoPE embeddings)的问题，Qwen利用了“动态NTK感知插值”来实现inference阶段的上下文长度扩展。

##### NTK-aware 插值

“NTK-aware”插值，核心思想是：**高频外推，低频内插**。

实现方式：

* 我们不是将RoPE的每个维度平均缩放一个因子s，而是通过减少高频的缩小和增加低频的放大来将插值压力分散到多个维度；
* 虽然人们可以通过许多方法获得这样的变换，但最简单的方法是对θ的值进行基础更改

![](https://i-blog.csdnimg.cn/direct/5e275817dde343adaf0bf3f2153b7af8.png)

与位置插值PI相比，该方法在扩展非微调模型的上下文大小方面表现得更好

然而，这种方法的一个主要缺点是，由于它不仅仅是一种插值方案，一些维度被轻微外推到“超出边界”的值，因此使用“NTK-aware”插值进行微调的结果不如PI；此外，由于存在“越界”值，理论尺度因子s并不能准确描述真实的上下文扩展尺度。在实践中，对于给定的上下文长度扩展，尺度值s必须设置得高于预期尺度。

##### Dynamic NTK插值

在很多用例中，以从1到最大上下文大小不等的序列长度进行多次前向传递。一个典型的例子是自回归生成，其中序列长度在每一步之后递增1

有两种方法可以应用使用比例因子s的插值方法("NTK-aware" and "Dynamic NTK"):

1. 在整个推理周期中，嵌入层是固定的，包括缩放因子s=L′/L，其中L′是固定数量的扩展上下文大小；
2. 在每次前向传递中，位置嵌入更新缩放因子(the position embedding updates the scale factor)：s=max(1,l′/L)，其中l′是当前序列的序列长度

上述方法中，方法1的问题在于模型在长度小于L时可能出现性能折扣，当序列长度大于L′时可能出现突然退化；

对此，故提出了方法2，这种推理时间方法为动态缩放方法，当与“NTK-aware”插值相结合时，我们称之为“动态NTK”插值。

> 关于NTK-aware插值，具体请参考：
>
> [大模型长度扩展综述：从直接外推ALiBi、插值PI、NTK-aware插值、YaRN到S2-Attention-CSDN博客文章浏览阅读6.3k次，点赞29次，收藏53次。下半年以来，我全力推动我司大模型项目团队的组建，我虽兼管整个项目团队，但为了并行多个项目，最终分成了三个项目组，每个项目都有一个项目负责人，分别为霍哥、阿荀、朝阳，有何问题 欢迎随时留言评论，thanks了解几种外推方案做了什么然后再顺着苏剑林文章的思路来看为什么这样做但总觉得不够深刻moe我有了解过GLaM，Mistral那边的没了解过打算了解下，估计也大差不差。\_ntk-aware![](https://i-blog.csdnimg.cn/blog_migrate/be19846480ab44ce477585fc567aeaa0.png)https://blog.csdn.net/v\_JULY\_v/article/details/135072211](https://blog.csdn.net/v_JULY_v/article/details/135072211 "大模型长度扩展综述：从直接外推ALiBi、插值PI、NTK-aware插值、YaRN到S2-Attention-CSDN博客")

#### 2.5.3 LogN-Scaling

根据**熵不变性**以及一些合理的假设，我们可以得到一个新的缩放因子，从而得到一种Scaled Dot-Product Attention：

![](https://i-blog.csdnimg.cn/direct/d9bae2affc1c47689b6a03721a9a9db6.png)

（这里的𝜅是一个跟𝑛,𝑑都无关的超参数）

LogN-Scaling可以根据上下文长度与训练长度的比值，对Q和V的点积进行重新缩放，确保注意力值的熵随着上下文长度的增长而保持稳定。

> 关于熵不变形以及LogN-Scaling，具体请参考：
>
> [从熵不变性看Attention的Scale操作 - 科学空间|Scientific Spaces当前Transformer架构用的最多的注意力机制，全称为“Scaled Dot-Product Attention”，其中“Scaled”是因为在$Q,K$转置相乘之后还要除以一个$\\sqrt...![icon-default.png?t=N7T8](https://i-blog.csdnimg.cn/blog_migrate/003a2ce7eb50c2e24a8c624c260c5930.png)https://spaces.ac.cn/archives/8823](https://spaces.ac.cn/archives/8823 "从熵不变性看Attention的Scale操作 - 科学空间|Scientific Spaces")

#### 2.5.4 分层窗口Self-Attention

* 分层窗口：使用分层窗口Self-Attention，将注意力限制在一个上下文窗口内，防止模型关注到太远的内容；
* 不同窗口大小：在不同层采用不同的窗口大小，较低的层使用较短的窗口，而较高的层使用较长的窗口（研究团队观察到Qwen模型在处理长上下文时在不同层次上的建模能力存在差异，较低的层次相对于较高的层次更加敏感于上下文长度的扩展；为此，为每个层分配了不同的窗口大小，对较低的层使用较短的窗口，对较高的层使用较长的窗口）。

综合这些技术，Qwen模型在inference阶段可以处理8192（相较于训练所使用的2048，扩展了4倍）个token的长序列，外推能力优异。

同时，实验结果表明，应用以上所述的关键技术可以使模型在上下文长度增加时始终保持低困惑度，这表明这些技术在增强模型理解和生成长文本的能力方面发挥了重要作用。

![](https://i-blog.csdnimg.cn/direct/1168264551cd4025944ba6bc2ff01354.png)

### 2.6 **预训练效果**

为评估Qwen模型zero-shot 和 few-shot能力，研究团队使用了一系列数据集进行全面的基准评估；同时将 Qwen与最新的开源基座模型进行比较，包括 LLAMA、LLAMA2、MPT、Falcon、Baichuan2、ChatGLM2、InternLM、XVERSE 和 StableBeluga2。评估共涉及 7 个常用基准，分别是：MMLU（5-shot）、C-Eval（5-shot）、GSM8K（8-shot）、MATH（4-shot）、HumanEval（0-shot）、MBPP（0-shot）、BBH（Big Bench Hard）（3-shot）

* 三种参数规模的Qwen模型在所有下游任务中都表现出了卓越的性能。在多个基准测试中，明显优于其他开源模型，甚至优于体量更大的模型
* 在3个数据集上（C-EVAL、MATH、HumanEval），Qwen-14B版优于LLaMA2-70B模型。
* Qwen-7B 的表现超越 LLaMA2-13B，并比肩Baichuan2-13B
* 值得注意的是，Qwen-1.8B虽然参数量相对较少，但在某些任务中仍具有一定竞争力，在某些情况下甚至超过了更大的模型。

三、对齐
--------

预训练的大型语言模型在实际应用中常常表现出与人类行为不一致，因此通常不适合作为直接的AI助手。最近的研究表明，采用对齐技术，例如监督微调（SFT）和基于人类反馈的强化学习（RLHF），可以显著提高语言模型进行自然对话的能力。本文将介绍Qwen模型如何进行SFT和RLHF，并评估它们在聊天环境中的表现。

### 3.1 **有监督微调SFT**

为了获得对人类行为的理解，第一步是执行SFT，它根据聊天类型的数据（包括Query和Response）对预训练的LLM进行微调。

#### 3.1.1 SFT 数据

在SFT数据处理过程中，研究团队主要采用了以下手段：

* **使用ChatML格式**：Qwen采用ChatML样式(OpenAI 2022年提出)的格式来进行模型训练。ChatML格式利用特殊符号表示不同类型信息（如系统设置、用户输入、助手输出等，这有助于模型有效区分信息）。

![](https://i-blog.csdnimg.cn/direct/c5276ca601a144eab35b828c4e0da6e0.png)

* **对话流式风格**：采用会话流式对话风格，而不是简单的问答形式，使模型学会真实的人机交互。
* **任务的多样性**：通过专注于**不同任务**的自然语言生成来提高模型的有用性。
* **去除格式化数据**：为了确保模型能够泛化到广泛的场景，特别排除了在提示模板中格式化的数据（这些数据可能会限制模型功能）。
* **安全性**：通过注释与安全问题相关的数据（如暴力、偏见和色情）来优先考虑语言模型的安全性。

#### 3.1.2 SFT训练

在SFT的训练过程中，研究团队具体采用了以下手段：

* **训练目标**：与预训练一致，使用Next Token Prediction 作为SFT的训练任务。
* **训练Loss**：在训练过程中对系统和用户输入应用loss mask （即将用户/系统的输入mask掉，不计算loss，仅对Qwen的回复部分计算loss）。
* **优化器**：使用AdamW优化器，超参数β1、β2和ϵ为别为0.9、0.95和1e−8；学习率先增后恒定。
* **Batch Size**：序列长度限制在2048，训练batch size=128。
* **学习率**：训练4000步,在前1430步中，学习率逐渐增加，达到2e−6的峰值。
* **防过拟合**：为了防止过拟合，权重衰减的值设置为0.1，dropout设置为0.1，梯度裁剪的限制为1.0。

### **3.2** RLHF

虽然SFT已被证明是有效的，但我们承认其泛化和创造性能力可能有限，并且容易过度拟合。
为了解决这个问题，我们实施了基于人类反馈的强化学习(RLHF)，以进一步使SFT模型与人类偏好保持一致。

这个过程包括训练一个奖励模型(Reward Model)，并使用近端策略优化(PPO) (Schulman et al.， 2017)进行policy training。

#### 3.2.1 偏好模型（Reward Model）

在构建偏好模型时，分成了两步：偏好模型预训练、偏好模型微调。

**预训练偏好模型(**preference model pretraining，PMP)**：**

1. 比较数据集：使用了一个庞大的比较数据集，该数据集由样本对组成，每个样本对包含对单个查询的两个不同的响应及其相应的人类偏好。
2. 池化层：奖励模型由**同等大小Qwen模型+池化层**得来，用特殊的句子结束标记经过池化层的映射值作为模型奖励值(_incorporated a pooling layer into the original QWEN model to extract the reward for a sentence based on a specific end token)。_

**微调奖励模型：**

1. 分类系统：为确保prompt具备一定的多样性和复杂性，创建了一个包含约6600详细标签的分类系统。
2. 平衡采样：并采用了一种平衡采样算法，以在选择提示时兼顾多样性和复杂性。
3. 多花样采样：为了生成多样的回复，实验过程使用了**不同规模和采样策略的 Qwen 模型**。因为多样化的回复有助于降低标注难度并提高奖励模型的性能。

模型在训练过程中，学习率恒为3e−6，批次大小为64，最大长度为2048，训练一个epoch。


#### 3.2.2 强化学习Proximal Policy **Optimization**（PPO）

PPO阶段共包含四个模型：policy模型、value模型、reference模型、reward模型。在开始PPO流程之前，暂停policy模型的更新，先对value模型训练50步预热，从而确保value模型能够有效地适应不同的reward模型。

在PPO过程中，policy模型和value模型的学习率分别为1e−6和5e−6。为了增强训练的稳定性，裁剪值设置为0.15。

在进行推理时，生成策略的top-p值设置为0.9。研究结果表明，虽然熵值略低于 top-p=1.0时的值，但奖励的增加速度更快，最终在类似条件下始终能够获得较高的评估奖励。

此外，Qwen还采用了预训练梯度来减缓所谓的对齐税(alignment tax)。研究表明，在这种特定的奖励模型下，KL 惩罚在非严格代码或数学性质的基准测试（如常识性知识和阅读理解测试）中足以弥补对齐税。与PPO数据相比，预训练梯度必须使用更大量的预训练数据，以确保预训练梯度的有效性。此外，实证研究表明，过大的系数值会大大阻碍与奖励模型的匹配，最终影响匹配，而过小的系数值对减缓对齐税的效果微不足道。 

> ##### 对齐税
>
> 对齐模型相比未对齐模型在某些能力上表现下降，需要更多计算资源来弥补性能

四、部署实践
------------

### 4.1 部署示例

##### 模型要求

* python 3.8及以上版本
* pytorch 1.12及以上版本，推荐2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）

##### 快速使用

我们提供简单的示例来说明如何利用🤖 ModelScope和🤗 Transformers快速使用Qwen-7B和Qwen-7B-Chat。

在开始前，请确保你已经配置好环境并安装好相关的代码包。最重要的是，确保你满足上述要求，然后安装相关的依赖库。

```bash
git clone https://github.com/QwenLM/Qwenpip install -r requirements.txt
```

如果你的显卡支持fp16或bf16精度，我们还推荐安装[flash-attention](https://github.com/Dao-AILab/flash-attention "flash-attention")来提高你的运行效率以及降低显存占用。(**flash-attention只是可选项，不安装也可正常运行该项目**)

```bash
git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attentioncd flash-attention && pip install .# 下方安装可选，安装可能比较缓慢。# pip install csrc/layer_norm# pip install csrc/rotary
```

接下来你可以开始使用Transformers或者ModelScope来使用我们的模型。

##### 🤗 Transformers安装方式

如希望使用Qwen-chat进行推理，所需要写的只是如下所示的数行代码。**请确保你使用的是最新代码，并指定正确的模型名称和路径，如`Qwen/Qwen-7B-Chat`和`Qwen/Qwen-14B-Chat`**

```python
from transformers import AutoModelForCausalLM, AutoTokenizerfrom transformers.generation import GenerationConfig # 可选的模型包括: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()# 使用CPU进行推理，需要约32GB内存# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()# 默认使用自动模式，根据设备自动选择精度model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval() # 可指定不同的生成长度、top_p等相关超参model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 第一轮对话response, history = model.chat(tokenizer, "你好", history=None)print(response)# 你好！很高兴为你提供帮助。 # 第二轮对话response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)print(response)# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。 # 第三轮对话response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)print(response)# 《奋斗创业：一个年轻人的成功之路》
```

##### 🤖 ModelScope安装方式

魔搭（ModelScope）是开源的模型即服务共享平台，为泛AI开发者提供灵活、易用、低成本的一站式模型服务产品。使用ModelScope同样非常简单，代码如下所示：

```python
from modelscope import AutoModelForCausalLM, AutoTokenizerfrom modelscope import GenerationConfig # 可选的模型包括: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", revision='v1.0.5', trust_remote_code=True)model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", revision='v1.0.5', device_map="auto", trust_remote_code=True, fp16=True).eval()model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", revision='v1.0.5', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参 response, history = model.chat(tokenizer, "你好", history=None)print(response)response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) print(response)response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)print(response)
```

### 4.3 量化

量化方案为基于[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ "AutoGPTQ")的量化，提供Int4量化模型，包括Qwen-7B-Chat [Click here](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4 "Click here")和Qwen-14B-Chat [Click here](https://huggingface.co/Qwen/Qwen-14B-Chat-Int4 "Click here")。该方案在模型评测效果几乎无损，且存储需求更低，推理速度更优。

以下示例说明如何使用Int4量化模型。在开始使用前，请先保证满足要求（如torch 2.0及以上，transformers版本为4.32.0及以上，等等），并安装所需安装包：

```bash
pip install auto-gptq optimum
```

如安装`auto-gptq`遇到问题，建议到官方[repo](https://github.com/PanQiWei/AutoGPTQ "repo")搜索合适的wheel。

随后即可使用和上述一致的用法调用量化模型：

```bash
# 可选模型包括："Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"model = AutoModelForCausalLM.from_pretrained(    "Qwen/Qwen-7B-Chat-Int4",    device_map="auto",    trust_remote_code=True).eval()response, history = model.chat(tokenizer, "Hi", history=None)
```

参考资料：

[https://zhuanlan.zhihu.com/p/658392609](https://zhuanlan.zhihu.com/p/658392609 "https://zhuanlan.zhihu.com/p/658392609")

[从零手搓大模型之路（一、学习Qwen模型架构）\_qwen2模型结构-CSDN博客](https://blog.csdn.net/qq_37021523/article/details/138901191 "从零手搓大模型之路（一、学习Qwen模型架构）_qwen2模型结构-CSDN博客")

[LLM系列 | 26：阿里千问Qwen模型解读、本地部署-CSDN博客](https://blog.csdn.net/ljp1919/article/details/134220489 "LLM系列 | 26：阿里千问Qwen模型解读、本地部署-CSDN博客")

[Qwen（通义千问）安装部署 - 开源中文大语言模型 - 梯子教程网](https://www.tizi365.com/topic/9020.html "Qwen（通义千问）安装部署 - 开源中文大语言模型 - 梯子教程网")
