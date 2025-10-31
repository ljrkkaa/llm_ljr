参考链接：

https://zhuanlan.zhihu.com/p/703680298



本文是讨论在LLM中进行多轮对话的SFT情况下是否有必要使用[Packing技术](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=Packing%E6%8A%80%E6%9C%AF&zhida_source=entity)系列的第一篇，主要是经验之谈，没有具体的理论分析和实验验证，只作为本系列的第一篇供大家讨论交流。

---

TL;DR：可以，但是不一定有必要

---

1.什么是Packing？
-----------------

这里参考

[@swtheking](//www.zhihu.com/people/7b689f2300f55bd187d09ba5e3a9cfd7)

学长的回答：

> [SFT packing](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=SFT+packing&zhida_source=entity)指的是在训练sft的过程中，将多个sft数据pack到一个样本内进行训练的方式，这种方式会加快模型训练速度，原因是如果不进行SFT packing，那么对于短文本sft，需要padding到一个batch的最长长度，那么会浪费很多计算token。

1\. Packing的好处
-----------------

一言蔽之：将Packing拉满到max\_length最直接的好处就是适合压榨GPU算力，可以节省部分时间。

除此之外，也有大佬补充，Non-Packing的方式会影响模型续写的效果，因此会影响一些benchmark效果。

2\. Packing的问题
-----------------

### 2.1 短文本受损

因为我没有特别做过关于文本长短的对比实验（没有合适的数据集以及大量的计算资源），所以这里继续引用

[@swtheking](//www.zhihu.com/people/7b689f2300f55bd187d09ba5e3a9cfd7)

学长的回答：

> SFT Packing以后其实是削弱了模型对难的短query和短答案的拟合。在无sft packing得情况下，假设batch\_size = 1，那么如果有个短query和短答案在这个batch里，其余补充padding，那么这个batch的gradient全是这个短文本的gradient，模型对这个query的拟合能力会变强。但是如果SFT packing以后，多个短文本在一个样本中，这个batch的gradient会被稀释，短文本的拟合就不会特别强。

当然，学长的回答里也提到了：

> 但拟合能力似乎和泛化不可以挂钩，初步观察SFT Packing和Non SFT Packing的效果差不了很多。另外，在数据量小，或者特定困难的数据上，SFT Packing是有损泛化效果的，但在大批量数据上是无损泛化效果的。

2.2 对多轮对话有影响
--------------------

如果说短文本受损是可接受的（梯度只是被稀疏，并不是没有没有梯度），那么Packing对于多轮对话的影响则是致命的。这里的讨论主要是受到了群友

[@swtheking](//www.zhihu.com/people/7b689f2300f55bd187d09ba5e3a9cfd7)

和

[@郑楚杰](//www.zhihu.com/people/ad6c5a94fa1647ac9bd040a2e999536f)

的讨论的启发，我沿着他们的思路进行了更深入一点的思考来抛砖引玉，希望大家指点。

主要原因是，目前没有使用[Block Attention](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=Block+Attention&zhida_source=entity)的微调框架对于Packing的实现方式往往是在两个conversation之间用Padding Token或者是EOS Token进行隔断（大多数情况下这两个特殊的token共享一个id），这种方式无法避免Seq和Seq之间的影响，并且，这种数据格式和多轮对话的数据格式是一致的。

参考

[@红雨瓢泼](//www.zhihu.com/people/35338267a3d8b34814c67994323fb393)

在多轮对话微调中的介绍：

[![](https://pic1.zhimg.com/v2-2dd4c5ba2ce7186512483cba6d4cc3bf.jpg?source=7e7ef6e2&needBackground=1)红雨瓢泼：一文看懂：如何充分高效训练多轮对话大模型288 赞同 · 26 评论 文章](https://zhuanlan.zhihu.com/p/645517143)

> 训练的时候，需要在每个Assistant的回复后面都添加`</s>`，作为此轮对话生成结束的标识符。否则推理的时候，模型很难采样到`</s>`，从而无法结束生成。

如果在多轮对话中使用EOS Token作为当前轮对话结束的标识，那么我们就会发现，模型仅能依赖BOS Token来作为当前Seq的标识符（即BOS Token前面的是历史Seq，BOS Token后面的才是模型需要的历史信息），实际上这一点对于现在的LLM通常来说是难以做到的，因为在没有Block Attention的情况下，Seq和Seq之间的Attention系数会导致模型总是无法避免的依赖于之前的Seq的内容，因此，如果一个数据集中既出现了多轮对话又使用了Packing的技术，会导致模型无法区分历史对话和当前对话。如果强行使用Packing技术，那么甚至会存在算力浪费，参考群内大佬给出的答案：

> 因为当前的样本不能Attention其他样本的，这部分要在Self-Attention里面mask掉。这样的话这个效率不好说了，我不用Packing，每次动态Padding 也浪费不了多少计算量。

3\. 实验
--------

为了讨论动态Padding和Packing，以及其他的优化（雕花）技术可以分别带来多少计算量的优化，我们选择[llamafactory/alpaca\_en](http://huggingface.co/datasets/llamafactory/alpaca_en)作为基准数据集，使用[llama2](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=llama2&zhida_source=entity)的[tokenizer](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=tokenizer&zhida_source=entity)进行分词，计算不同的方法分别会带来多少的计算量增加，在我们的实验中我们采用padding的数量作为计算量的评估标准。

实验方法如下：

1. Packing：以[LLaMa-Factory](https://zhida.zhihu.com/search?content_id=244503925&content_type=Article&match_order=1&q=LLaMa-Factory&zhida_source=entity)为例，我们使用其中对应的处理函数preprocess\_packed\_supervised\_dataset，在[LLaMA-Factory](http://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processors/supervised.py#L116)中，我们遵循常规实验设置cutoff\_len = 4096，计算最后的数据集中Padding Token的个数并且减去samples个数（llama2的tokenizer中EOS Token和Padding Token共享id，因此需要去除每个sample最后的EOS Token）。
2. 静态Padding：我们统计出数据集中最长的序列编码后的长度dataset\_max\_len=1511，将每个序列Padding到dataset\_max\_len的长度，计算这个时候的Padding Token的个数。
3. 动态Padding：我们遵循常规实验设置batch\_size=128，我们统计出每个bacth中最长的序列编码后的长度bacth\_max\_len=1511，将这个batch中的所有序列Padding到bacth\_max\_len的长度，计算这个时候的Padding Token的个数。
4. 优化动态Padding：我们首先将数据集按照编码后的序列长度进行排序再划分batch，尽量保证每个bacth中的序列数据长度接近，之后计算Padding Token的个数和上一个方法一样。

| 方法            | Padding总数 | 平均Padding | 增幅   |
| --------------- | ----------- | ----------- | ------ |
| Packing         | 114800      | 2.22        | 0      |
| 静态Padding     | 73526672    | 1,422.20    | 640.63 |
| 动态Padding     | 18470778    | 357.28      | 160.94 |
| 优化动态Padding | 131600      | 2.5455      | 1.15   |

3\. 总结
--------

总体上来说，Packing与否对于下游任务的影响主要依赖于评估数据集的具体特性，因此简单来说是有好（模型续写）有坏（短文本受损，对多轮对话有影响），但是考虑到Packing本身就需要在数据集加载阶段额外进行特殊处理，并且优化动态Padding的方法可以最大限度的减少计算量，因此，个人建议，最好不要使用Packing这种方法，当然，Packing还有一个小问题就是会让batch\_sum变化，进而导致我之前优美的超参数用不上了（笑
