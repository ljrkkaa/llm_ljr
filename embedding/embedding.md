# 静态编码
word2vec   https://zhuanlan.zhihu.com/p/114538417
           https://blog.csdn.net/weixin_41885239/article/details/121387608

fastText   https://zhuanlan.zhihu.com/p/598176213

fastText 是一种文本分类模型，它的目标是根据输入文本（通常是一个句子或文档）来预测其所属的类别标签。它的输入是一个句子，输出是一个类别，例如情感分类中的“正面”或“负面”。

而 CBOW（Continuous Bag of Words）模型是 Word2Vec 中的一种词向量训练方法，它的目标是根据上下文词来预测中间的目标词。

# 动态编码

静态编码的每个单词都只能学出一个词向量，但在np工作中，单词再不同上下文中更可能有不同的意义。这就需要动态编码，也就是一个单词可以学出多个词向量。以下方法预训练阶段是无监督的，下游任务一般是有监督的。

ELmo(Embedding from Language Models)    https://zhuanlan.zhihu.com/p/51679783
                                        https://www.bilibili.com/video/BV1384y1J7fh/?vd_source=5bf2abd640b441eb6f95f5cd173690fa
                            


GPT(Generative Pre-Training),从名字看其含义是指的生成式的预训练。

GPT也采用两阶段过程：
第一个阶段：利用语言模型进行预训练；
第二个阶段：通过 Fine-tuning 的模式解决下游任务。

https://www.cnblogs.com/nickchen121/p/16470569.html
下游微调  https://blog.csdn.net/weixin_60734652/article/details/132470086

BERT

https://ar5iv.labs.arxiv.org/html/1810.04805?_immersive_translate_auto_translate=1
https://www.cnblogs.com/nickchen121/p/16470569.html
源码级别  https://zhuanlan.zhihu.com/p/103226488
参数计算  https://blog.csdn.net/weixin_44402973/article/details/126405946
