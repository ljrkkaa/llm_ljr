# 1 RLAIF

参考链接：

https://zhuanlan.zhihu.com/p/682888070


总体来说，RLAIF和RLHF的训练过程基本一样，唯一的不同点只是在于reward model训练阶段的数据收集，它使用了现有的LLM对(x,1y1,y2)进行评价

![](assets/20250827_125135_image.png)


Prompt由以下四部分构成：

1. Preamble（前言）
2. Few-shot exemplars (optional)（示例）
3. Sample to annotate（要标注的样本）
4. Ending (e.g. preferred Response=)（结束语）

![](https://pic2.zhimg.com/v2-c07429b4d7b134b59f230d43277b153b_r.jpg)

尝试从AI标注器中引出思维链 (COT) 推理，以提高与人类偏好的一致性。我们将标准提示的结尾(i.e. “Preferred Summary=”) 替换为 “Consider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better. Rationale:”，然后解码LLM的响应。最后，我们将原始提示、响应和原始结尾字符串“Preferred Summary=”连接在一起

## 两种形式

文中提出了两种形式的RLAIF，分别是蒸馏式的RLAIF（Distilled RLAIF）和直接式RLAIF（Direct RLAIF），主要区别仅在于训练RL的时候reward来自 reward model 还是直接来自现有LLM。

蒸馏式的RLAIF其实就是我们经典的训练步骤

直接式RLAIF不再收集偏好数据训练奖励模型，而是在RL阶段中直接给生成的answer打分，具体的打分方式如下：

![](assets/20250827_125740_image.png)

## 总结

RLAIF 达到了与 RLHF 相当的性能。首先，我们观察到 RLAIF 和 RLHF 策略分别在 71% 和 73% 的样例中比有监督微调 (SFT) 基线更受人类青睐，并且两种获胜率在统计上没有显着差异。其次，当被要求直接比较 RLAIF 和 RLHF 的生成时，人们对两者都具有较高偏好（即 50% 的胜率）。这些结果表明，RLAIF 是 RLHF 的可行替代方案，它不依赖于人工标注，并提供有吸引力的缩放特性

从文中提到的相关实验结果来看，本文提出的RLAIF方式确实有不小的吸引力，特别是对于收集数据很困难的团队来说，可以极大降低相关成本。（其实我觉得只是把成本从人工转到了AI计算资源上，不过方便点）

但是另一方面，它的**性能还是有上限，无法继续迭代升级数据质量来进一步提升模型性能**。因此，我认为更适合作为初始模型版本训练的数据收集器，后续进一步提高性能还是需要人工来精标偏好数据集。

# 2
