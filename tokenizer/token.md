# 1 学习链接
https://huggingface.co/docs/transformers/tokenizer_summary#byte-pair-encoding-bpe

https://zhuanlan.zhihu.com/p/652520262

LLM中的Tokenizers.pdf


# 2 Byte-Pair Encoding (BPE)
预分词后，已创建一组唯一词汇，并确定了每个词汇在训练数据中出现的频率。接下来，BPE 创建一个包含所有在唯一词汇集中出现的符号的基础词汇表，并学习合并规则，将基础词汇表中的两个符号合并成一个新的符号。它持续这样做，直到词汇表达到所需的词汇量。请注意，所需的词汇量是在训练分词器之前定义的超参数。

# 3 WordPiece
WordPiece 首先初始化词汇表以包含训练数据中出现的每个字符，并逐步学习给定数量的合并规则。与 BPE 相比，WordPiece 不会选择最频繁的符号对，而是选择在添加到词汇表后最大化训练数据可能性的符号对。

# 4 Unigram
与 BPE 或 WordPiece 相比，Unigram 将其基础词汇表初始化为大量符号，并逐步减少每个符号以获得更小的词汇表。基础词汇表可以对应于所有预分词的单词和最常见的子字符串。Unigram 并不直接用于 transformers 中的任何模型，但它与 SentencePiece 一起使用。

# 5 SentencePiece
上面提到的所有分词算法都存在同一个问题：这些分词方式都是建立在原文本通过空格来分隔词语的。但是并不是所有语言词和词之间都是有空格的。还有就是，将分词结果解码到原来的句子中时，会在不同的词之间添加空格，这样解码出来的标点符号就不会和词语连着。

Raw text: Hello world.
你好，世界。

Tokenized: [Hello] [world] [.]
已分词：[Hello] [world] [。]

Decoded text: Hello world .
解码文本：Hello world 。

​ 这就是编码解码出现的歧义性，因此需要特别定义规则来实现互逆。还有一个例子是，在解码阶段，欧洲语言词之间要添加空格，而中文等语言则不应添加空格。一种解决方案是，给不同的语言定制不同的预分词算法和解码算法，比如 XLM 就在中文、日文和泰文上做了尝试。

​ 为了能够更好地解决这个问题，谷歌在 SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018)中提出，把将所有的字符都转化成 Unicode 编码，空格用_来代替，然后进行分词操作。这样空格也不需要特别定义规则了，然后在解码结束后恢复即可。然后还是用 BPE 或者 ULM 算法来构建词表。

​ SentencePiece 集成了BPE、ULM 算法，除此之外，它也能支持字符和词级别的分词。使用 SentencePiece + ULM 的模型有 ALBERT、XLNet、T5。

​