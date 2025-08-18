1. **FIM (Fill-in-the-Middle)**

   参考资料：

   https://juejin.cn/post/7307544166447317004

   https://zhuanlan.zhihu.com/p/652855450

   GPT简介：

   * 一种代码大模型的训练方式。
   * 传统的代码补全是 **从左到右** 的（prefix → completion），但很多开发场景中需要在代码中间插入或修改。
   * FIM 训练方法会把代码拆成三段：
     * 前缀（prefix）
     * 缺失片段（middle / masked part）
     * 后缀（suffix）
   * 模型通过学习在前后文中补全“中间缺失部分”，因此对 IDE 代码补全、交互式修改非常有用。

   <br/>document→(prefix,middle,suffix)→(prefix,suffix,middle)

llm语言模型的自回归训练适合代码提示生成任务，但不支持代码补全。

代码补全就是根据代码的上下文预测代码缺失的部分，比如在IDE中，对鼠标位置的代码自动完成，文档自动生成等。

将训练的文本序列一部分移动到结尾，然后自回归重新排序进行训练。

策略参考论文[[Efficient Training of Language Models to Fill in the Middle](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2207.14255.pdf)]，将文本分成<prefix>,<middle>和<suffix>三部分(前，中，后)，然后按2种方式排列。

PSM：即prefix,suffix,middle的顺序，结构如下图：

![](https://pica.zhimg.com/v2-bc79c3eebc3d946c312240fa7668d29e_r.jpg)

SPM:即suffix,prefix,middle的顺序，类似上图，前2部分顺序反转。

训练时样本一半按PSM格式，一半按SPM格式。

2. **LCFT (Long Context Fine-Tuning)**

   * 指 **长上下文微调** ，让模型能处理比默认更长的上下文窗口。
   * 大模型通常在预训练时有一个固定的上下文长度限制，比如 4k、8k、16k tokens。
   * 通过 LCFT，可以让模型支持 **更长的输入上下文** （比如几十 k 的代码文件），对代码理解和大规模 refactoring 特别有帮助。
   * 似乎只是应用了ROPE没什么特别  训练时样本一半按PSM格式，一半按SPM格式。

处理长文本是大语言模型的基础能力，为了支持，支持更长的上下文，常见的做法就是外推。

llama2训练时token长度为4096，codellama则增加到16384。

为了将训练成本限制在微调，参考RoPE线性插值的思路，只不过这里没有采用插值，而是修改注意力的衰减周期。参数从10000改成1000000，然后进行微调训练。