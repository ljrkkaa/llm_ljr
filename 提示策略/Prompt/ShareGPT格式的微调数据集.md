参考链接：
https://www.cnblogs.com/zhiyong-ITNote

### 概述

ShareGPT格式的数据集中，一般是如下格式：

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "I saw a dress that I liked. It was originally priced at $200 but it's on sale for 20% off. Can you tell me how much it will cost after the discount?"
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"calculate_discount\", \"arguments\": {\"original_price\": 200, \"discount_percentage\": 20}}"
      },
      {
        "from": "observation",
        "value": "{\"discounted_price\": 160}"
      },
      {
        "from": "gpt",
        "value": "The dress will cost you $160 after the 20% discount."
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "[{\"name\": \"calculate_discount\", \"description\": \"Calculate the discounted price\", \"parameters\": {\"type\": \"object\", \"properties\": {\"original_price\": {\"type\": \"number\", \"description\": \"The original price of the item\"}, \"discount_percentage\": {\"type\": \"number\", \"description\": \"The percentage of discount\"}}, \"required\": [\"original_price\", \"discount_percentage\"]}}]"
  }
]
```

`function_call`表示函数调用，什么是函数调用？其作用是什么？  
由于大模型的数据一般都是截止于某个时间点之前的数据，不具备实时性。比如，我要问今天的天气，正常来说，由于模型参数的局限性，是不会知道的。但基于函数调用的功能，就解决了这个问题。  
所谓的`function_call`，在某个程度来说，可以理解为API调用，这个API就是一个function，提供了某种功能。  
`observation`表示观测结果，即`function_call`的执行结果。  
`tools`表示工具，即对`function_call`的总结描述。

> observation并不是新词汇，对于HMM模型如果有了解的话，在其模型算法的表述中，也有着observation的相关引用。

ShareGPT格式简单明了而且结构强大，不仅仅轻易的支持单轮对话、多轮对话；还引入了强大的函数调用，支持功能扩展。

### 扩展

`function_call`的设计引申出来，可以对应到业务开发中的规则引擎、脚本引擎等设计。譬如，支持在json参数的格式中，传入JS脚本参数，做一些强大的运算等。在原有的参数格式中，引入强大的函数调用支持。  
[如何在聊天模型中调用函数（Function Calling)--金融大模型知识库实战（十六）](https://mp.weixin.qq.com/s/l19a-PqFvmukG1YEGygTCg)  
[大模型开发 - 一文搞懂 Function Calling（函数调用）](https://mp.weixin.qq.com/s/wAWsXI5jWa_rtSKw5nJ_tw)

> 转载请注明住处：[https://www.cnblogs.com/zhiyong-ITNote](https://www.cnblogs.com/zhiyong-ITNote)
