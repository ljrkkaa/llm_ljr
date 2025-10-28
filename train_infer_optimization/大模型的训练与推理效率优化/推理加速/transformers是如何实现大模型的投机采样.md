参考链接：

https://zhuanlan.zhihu.com/p/653935025

良睦路程序员~

**投机性解码原理**
--------------

简单的来说：

1. 就是使用一个小模型来做草稿，然后使用大模型做纠正检查。
2. 小模型的参数量要远小于原模型参数量一个级别才效果明显。
3. 小模型和原模型的tokenizer最好一模一样，不然会增加额外的解码、编码时间。

**具体流程如下：**
--------------

假设，让小模型预测5步

**第一步**
------

原始token序列为蓝色序列

![](https://pic2.zhimg.com/v2-32bb853502e3d67f027500cc833912bb_r.jpg)

**第二步**
------

使用辅助模型生成5个新的token序列（红色）

![](https://pica.zhimg.com/v2-515662fa3d019e3095e7ab3beaef520c_r.jpg)

流程是这样的

![](https://pic2.zhimg.com/v2-4ed5cc1c8d5d719caaf0fd8622657265_r.jpg)

**第三步**
------

将蓝色序列和红色序列拼接在一起，放入原始模型中，并且生成一个绿色结果

![](https://pica.zhimg.com/v2-63c4ad4984c4f6e25fcb76848ba96b0e_r.jpg)

**第四步**
------

考虑到大模型的最后一步都是使用矩阵计算概率，那么在第三步中，看似生成了一个绿色结果。实际上，利用了矩阵计算的并行性：一步计算，就可以验证小模型生成的5个结果对不对。

![](https://picx.zhimg.com/v2-45e77624197f2bf5aa71da4ae2374f15_r.jpg)

**第五步**
------

看第四步生成的结果，可以发现：

1. 小模型生成的第一个token是466，但是大模型生成的第一个token是651。
2. 小模型生成的序列中，只要有一个错了，那后面就不能要了。
3. 因此不用小模型结果，使用大模型的第一个结果651（绿色小块）作为本轮的结果。
4. 以此循环，直到遇到结束符才停止。

第四步扩充
----------

可能还是有人没搞懂：明明大模型利用蓝色和红色块推理得到了绿色338，但是没怎么用到338。而且怎么一次性把绿色块的651、428、287、475、340计算出来的。

**这个疑惑的产生，主要是因为没有把[casualLM](https://zhida.zhihu.com/search?content_id=233447506&content_type=Article&match_order=1&q=casualLM&zhida_source=entity)类型的模型搞懂。casualLM类型的模型，都是预测下一个token的概率。而且使用的是矩阵计算，所以一次性把所有token对应的下一个位置，都预测出来了。**

第四步里面的绿色的651是怎么产生的：

![](https://pic3.zhimg.com/v2-0b10852a907dfc754b03a25122f61c5e_r.jpg)

第四步里面的绿色的428是怎么生成的：

![](https://picx.zhimg.com/v2-6b11bbf7c49706ea1b9b80eb50fb051f_r.jpg)

第四步里面的绿色的287是怎么生成的：

![](https://pica.zhimg.com/v2-cf6e57765007b84c659ac5493a03bd70_r.jpg)

第四步里面的绿色的475是怎么生成的：

![](https://pica.zhimg.com/v2-0f8382caf107c499f1a8b3f4895d71dc_r.jpg)

第四步里面的绿色的340怎么生成的：

![](https://picx.zhimg.com/v2-61302c678c9480f949f20be0b5dcd51d_r.jpg)

第四步里面的绿色的338是怎么生成的：

![](https://pica.zhimg.com/v2-6861861e126890b1750d0034a931a752_r.jpg)

这个时候，你在回看第四步的图（这里复制了一份），发现:

1. **小模型预测的第一个tokenid就已经错了（红色的466），大模型预测的是（绿色的651）。**
2. **这种自回归序列模型，一步错，步步错。因此，虽然后面的小模型预测了一个340（红色）和大模型预测的340（绿色是一样的），但是完全不能用。**
3. **考虑到大模型基于原始的token list（蓝色），预测了651（绿色）。那就把651拿来用了。**
4. **虽然小模型预测了5步，全都错了，但是也不亏。因为小模型的5步的计算时间，远远小于大模型一次预测的时间。**

![](https://pica.zhimg.com/v2-453137cc223c24fe533fe36f0b211bf0_r.jpg)

**解释**
----

可以想象一下：

1. 大模型直接利用原始token（蓝色序列），要预测新的5个token，计算起码需要跑5次。
2. 先使用小模型预测5个试一试，然后大模型借助矩阵计算的并行特性，一次性就可以验证这5个中，前几个都是对的。
3. 如果有对的，那节约的时间可不是一点点（因为小模型远小于大模型，所以小模型消耗的时间基本可以忽略不记）。
4. 这个思想很简单，举个例子：**树上全是枣子，旁边又有竹竿，那你肯定拿起竹竿，在空中挥了一挥。能打到枣子，算走运了，没打到枣子，也不亏。** 那么这个投机生成也是这个道理。

**代码部分**
--------

1. 虽然这个逻辑很简单，而且也说了，是使用了矩阵的并行性，但是在刚开始，我并不知道他在代码里面是如何实现的。
2. 虽然投机采样也就是在8月31号才火起来，但是代码在4月份的时候，就已经在transformers包里面实现了。

代码链接为：[https://github.com/huggingface/transformers/blob/4b796978656e461177a83d58ec3c2b06152c63db/src/transformers/generation/utils.py#L4269](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/4b796978656e461177a83d58ec3c2b06152c63db/src/transformers/generation/utils.py%23L4269)

**如何使用**
--------

在huggingface的transformers包里面，已经给到一个使用案例了，代码如下：

```python3
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
assistant_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id
input_prompt = "It might be possible to"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
outputs = model.assisted_decoding(
    input_ids,
    assistant_model=assistant_model,
    logits_processor=logits_processor,
    stopping_criteria=stopping_criteria,
)
tokenizer.batch_decode(outputs, skip_special_tokens=True)

```

**源码解读**
--------

是怎么的出来，是使用矩阵的并行原理，可以在这里看到：

```python3
            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Run a forward pass on the candidate sequence
            if "past_key_values" in model_kwargs:
            ...
            else:
                if self.config.is_encoder_decoder:
                ...
                else:
                    outputs = self(
                        candidate_input_ids,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        use_cache=True,
                    )

```

在这里，把原始的token和辅助模型（在huggingface的代码里面，叫辅助模型，但是和上面的小模型是一回事，叫法不一样）生成的token绑定在一起，然后放入原始模型做推理。

在下面的代码块中，把大模型批量处理的结果提取出来，和辅助模型生成的结果做比对。

对每一个生成的token进行检查

```python3

# ==============================
# 2.2. 处理新产生的 logits（模型输出的下一步预测分布）
# ==============================

# 从模型输出中取出最后 candidate_length+1 个 token 的 logits。
# -1 是因为生成时通常需要排除输入 prompt 的部分，只保留新增的部分。
new_logits = outputs.logits[:, -candidate_length - 1 :]  # 排除输入提示词部分

# 如果存在 logits_processor（例如强制约束、temperature scaling、重复惩罚等），逐步处理每个时间步的 logits。
if len(logits_processor) > 0:
    for i in range(candidate_length):
        # 对当前 step 的 logits 应用 processor（比如温度、top_k 等）
        new_logits[:, i, :] = logits_processor(
            candidate_input_ids[:, : cur_len + i],  # 当前输入 token 序列
            new_logits[:, i, :]                     # 当前 step 的 logits
        )

# 如果存在 logits_warper（例如 top-p、top-k 策略），再进行一次变换。
if len(logits_warper) > 0:
    for i in range(candidate_length):
        new_logits[:, i, :] = logits_warper(
            candidate_input_ids[:, : cur_len + i],
            new_logits[:, i, :]
        )


# ==============================
# 3. 从处理后的 logits 中选出下一个 token
# ==============================

if do_sample:
    # 采样模式：使用 softmax 得到概率分布，再随机采样生成下一个 token。
    probs = new_logits[:, -candidate_length - 1 :, :].softmax(dim=-1)
    # 从一组概率分布中随机抽取样本（根据概率加权）
    selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
else:
    # 贪心模式：直接取最大概率对应的 token（argmax）。
    selected_tokens = new_logits[:, -candidate_length - 1 :, :].argmax(dim=-1)


# ==============================
# 4. 比较预测结果（用于验证或拒绝采样机制）
# ==============================

# 从候选输入中取出模型预测的新 token 序列。
candidate_new_tokens = candidate_input_ids[:, -candidate_length:]

# 计算与主模型预测的 token 一致的数量：
#  - candidate_new_tokens: 辅助模型（或提前预测）的 token
#  - selected_tokens: 当前模型真正输出的 token
# 逐元素比较，统计连续匹配的 token 数量。
n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
# ↑ 含义：
#   - candidate_new_tokens == selected_tokens[:, :-1]  → True 表示匹配
#   - 取反 ~ → False 表示不匹配
#   - cumsum < 1 表示在第一个不匹配之前的所有位置
#   - sum() 统计匹配的 token 个数



```

**误区**
----

其实，刚开始，还以为是把辅助模型生成的新token逐步复制，从 `batchsize=1`变成 `batchsize=5`。但在debug的时候，解除了我的困惑。

因为数据上显示 `batchsize=1`，一直没有变化，就是取了logits的不同位置。

![](https://picx.zhimg.com/v2-b73587d2ee5b8a8b7f31160fb4cc8327_r.jpg)

然后想起来 `CasualLM`类型的模型，最后一层都是 `nn.Linear`结构(将 `hidden_states`转换成 `logits`概率)，然后就想到了是使用矩阵的并行原理实现的。

![](https://pic3.zhimg.com/v2-d7c3a6265489ea0a08b43f8b6ed7a81c_r.jpg)

**参考链接**
--------

1. Niels Rogge的推特链接:[https://twitter.com/NielsRogge/status/1697335383166472294](https://link.zhihu.com/?target=https%3A//twitter.com/NielsRogge/status/1697335383166472294)
2. 新智元的那个文章的知乎链接:[https://zhuanlan.zhihu.com/p/653729679](https://zhuanlan.zhihu.com/p/653729679)
3. huggingface的辅助生产文章链接:[https://huggingface.co/blog/zh/assisted-generation](https://link.zhihu.com/?target=https%3A//huggingface.co/blog/zh/assisted-generation)

**最后**
----

本文图解投机采样的原理，并且介绍了其代码实现，如果有写错的地方，大佬们多多指导~
