参考链接：
https://zhuanlan.zhihu.com/p/1932829167801574272
https://www.bilibili.com/video/BV14VhLzmEjg/?spm_id_from=333.337.search-card.all.click&vd_source=98e4164f2cbf19bf54ad534e5875112c
zotero论文

# 1 引言

在更大的语言模型上使用 GRPO 时，会出现训练不稳定的情况。在这篇论文，作者认为这种现象源于 GRPO 的重要性权重的设计错误，GRPO 对于 next-token 的重要性权重，容易引入高方差的噪声，在 response 的长度的增加和裁剪机制的作用下，最终导致训练崩溃。

为了解决这一问题，论文提出 GSPO (Group Sequence Policy Optimization)，将针对 token 分布的权重改换为针对 sequence 的重要性权重，并且从 sequence 的维度来计算梯度，而不是 token 的维度，和 reward 本身的定义保持一致。

最终，GSPO 在 **MoE 模型的** RL 训练上解决了稳定性问题，从而不必单独设计复杂的 trick 来维持稳定，**简化了 RL 架构**

# 2 GRPO的缺陷

## 重要性采样

RL 阶段，我们首先会采样一个 large rollout batch，为了提高采样效率，通常我们会将其切分成几个 mini-batches 来进行梯度更新，这一过程无可避免地会导致off-policy场景的出现，同时这也一定程度上说明了PPO和 GRPO 的 clip 机制可以防止那些过度 off-policy 的样本参与梯度计算

它的优化目标本质上是病态的（不适定的，ill-posed）这种病态源于对重要性采样权重的错误应用。

传统的重要性采样：

![](assets/20250827_172241_image.png)

依赖多次的采样估计去除分布不匹配，但是GRPO**只采样一次**

GRPO的单条轨迹虽然是从同一个旧分布采样得到的，但 token-wise 的重要性权重意味着这条轨迹中每输出一个 token 旧分布的输入条件 y_{i,<t} 这部分就变一次，导致无法在同一个条件分布下多次采样并加权平均得到新分布下策略梯度的无偏估计（相当于预测i love **u/him** 不同的输出u/him对应不同词的重要性加权 ），而且还引入了高方差。

## 优化目标的单位应与奖励的单位不匹配

优化目标Advantage是sequence单元的  但是优化时在token上优化，也就是reward是token单元的

![](assets/20250827_173051_image.png)

# 3 GSPO设计

GSPO追求优化目标的单位应与奖励的单位匹配，重要性采样改为在sequence单元上实现

![](assets/20250827_173342_image.png)

Si使用序列的重要性采样的几何平均

这很自然地和 sequence-level reward 定义一致，也让 clip 的机制意义更明确 (筛去过度 off-policy 的 sequence 的梯度)。

不然某些token被裁减掉，其他不变会让句子奇怪 （？

# 4 GSPO-token

![](assets/20250827_174004_image.png)

推导下来与sequence level的基本一致

![](assets/20250827_174348_image.png)

# 实验

## 路由重放策略 Routing Replay

![](assets/20250828_105013_image.png)

## MoE 训练中效果

**背景** . MoE 模型的训练中，使用 GRPO 算法，专家激活的不稳定性可能会导致强化学习训练无法正常收敛，进行一次梯度更新后，即使对于相同的 response，所激活的专家也可能发生显著变化。这一不稳定要素，导致 token 级别的重要性权重波动更大，从而如之前讨论的，最后导致模型崩溃。

相比于稠密模型的RL训练，MoE模型的稀疏激活特性引入了独特的稳定性挑战。特别地，我们发现当采用GRPO算法时，MoE模型的专家激活波动率可以防止RL训练正常收敛

网友评论：其实本质上来说还是方差（不过GRPO这个importance ratio的估计倒确实是biased），moe主要可能直接出现expert变换导致token-level的ratio估计发生巨大变化，dense相对来说就还好

![](https://pic4.zhimg.com/v2-6954b5e14cc70e636cc0e6d26d93acb9_r.jpg)

# 总结

GSPO专攻MOE下的优化 对dense效果不大

对rL架构的优化没看明白

![](assets/20250828_110523_image.png)

https://github.com/volcengine/verl/pull/2775


GSPO-token s_i 代码

![](assets/20250828_111551_image.png)
