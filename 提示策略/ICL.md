**In-Context Learning 上下文学习**

参考链接：

https://zhuanlan.zhihu.com/p/611217770

https://www.cnblogs.com/ting1/p/18254665

# 1 ICL介绍

自GPT-3首次提出了In-Context Learning(ICL)的概念而来，ICL目前已经变成了一种经典的LLMs使用方法。ICL，即In-Context Learning，是一种让大型语言模型（LLMs）通过少量标注样本在特定任务上进行学习的方法。这种方法的核心思想是，通过设计任务相关的指令形成提示模板，利用少量标注样本作为提示，引导模型在新的测试数据上生成预测结果。

* ICL主要思路是：给出少量的标注样本，设计任务相关的指令形成提示模板，用于指导待测试样本生成相应的结果。
* ICL的过程：并不涉及到梯度的更新，因为整个过程不属于fine-tuning范畴。而是将一些带有标签的样本拼接起来，作为prompt的一部分，引导模型在新的测试数据输入上生成预测结果。
* ICL方法：表现大幅度超越了Zero-Shot-Learning，为少样本学习提供了新的研究思路。

![](https://pic4.zhimg.com/v2-76557778e35baee4266048d08f667cc5_r.jpg)

In Context Learning（ICL）的关键思想是从类比中学习。上图给出了一个描述语言模型如何使用 ICL 进行决策的例子。首先，ICL 需要一些示例来形成一个演示上下文。这些示例通常是用自然语言模板编写的。然后 ICL 将查询的问题（即你需要预测标签的 input）和一个上下文演示（一些相关的 cases）连接在一起，形成带有提示的输入，并将其输入到语言模型中进行预测。

值得注意的是，与需要使用反向梯度更新模型参数的训练阶段的监督学习不同，**ICL 不需要参数更新，并直接对预先训练好的语言模型进行预测（这是与 prompt，传统 demonstration learning 不同的地方，ICL 不需要在下游 P-tuning 或 Fine-tuning）** 。我们希望该模型学习隐藏在演示中的模式，并据此做出正确的预测。

Few-shot 和 ICL 的区别

| 方面               | Few-shot Learning | ICL                        |
| ------------------ | ----------------- | -------------------------- |
| **是否调参** | ✅ 需要参数更新   | ❌ 不需要，直接推理        |
| **数据用法** | 样本用于微调模型  | 样本作为 prompt 的一部分   |
| **学习方式** | 模型真正学会任务  | 模型在上下文里“临时模仿” |
| **应用场景** | 数据有限但能微调  | 只调用 API，不想训练       |

2 形式化定义
------------

给出少量任务相关的模型输入输出示例(demonstration)，如$k$个示例$D\_k={f(x\_1,y\_1),...,f(x\_k,y\_k)}$，其中$f(x\_k,y\_k)$是一个预定义的关于Prompt的函数（文本格式），用于将$k$个任务相关的示例，转换成自然语言Prompt。

给出任务定义$I$，示例$D\_k$，以及一个新的输入$x\_{k+1}$，我们的目的是通过LLM生成输出$\\hat{y}\_k+1$。公式化为：

![](https://ai-studio-static-online.cdn.bcebos.com/64ca2a77b6b244bd87c5aba1995408ade6c8cd2c8f9b429fa0e52c54df1a8e39)

# 3 实例理解

以一个分类任务进行举例，从训练集中抽取了$k=3$个包含输入输出的实例，使用换行符"\\n"来区分输入和输出。

在预测时，可以更换测试样本输入（绿色部分），并在末尾留出空间让LLM生成。

![](https://ai-studio-static-online.cdn.bcebos.com/4f5b87e30c8c434b9e35ecfa14022e28e659d641083e400ba1ff2840146fcf06)

# 4 ICL实例设计

参考https://kcnd4kn8i6ap.feishu.cn/wiki/D106wyTlEizwX9kO6u5cp2W5nXe
