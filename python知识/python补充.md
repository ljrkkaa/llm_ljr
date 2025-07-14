# 2025.07.09

- [X] 1 collection

> Q: vocab = collections.defaultdict(int)
> A: 参考链接 https://blog.csdn.net/weixin_44799217/article/details/124380270

- [X] 2 re库是什么

> Q: bigram = re.escape(' '.join(pair))
> A: re 模块使 Python 语言拥有全部的正则表达式功能。
> 正则表达式学习：https://github.com/ziishaned/learn-regex/blob/master/translations/README-cn.md
> 简单的re函数：https://www.runoob.com/python3/python3-reg-expressions.html

- [X] 3 正则规则

> Q: 自定义一个正则规则, (?<!\S)h\ e(?!\S) 只有前面、后面不是非空白字符(\S)(意思前后得是没东西的)，才匹配h\ e，这样就可以把Th\ e<\w>排除在外
> p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
> A: 同问题2，但是还记不住

- [X] 4 函数前@ 跟装饰器什么关系

> Q: @staticmethod @dataclass
> A: @staticmethod 装饰器可以定义静态方法，将其放在方法定义的开头。
> @dataclass 装饰器可以定义数据类，将其放在类定义的开头。
> @classmethod 用于修饰类方法（Class Method）。类方法是绑定到类不是实例的方法，因此可以通过类名直接调用，而不需要先创建实例


| 类型       | 装饰器          | 第一个参数 | 作用                   |
| ------------ | ----------------- | ------------ | ------------------------ |
| 实例方法   | （无）          | `self`     | 操作对象（实例）       |
| 静态类方法 | `@classmethod`  | `cls`      | 操作类（不是实例）     |
| 静态方法   | `@staticmethod` | （无）     | 无隐式参数，纯工具方法 |

# 2025.07.11

- [X] 1 pytorch基础

> Q: 常见的函数来自nn还是torch等等
> A: 参考链接

# 2025.07.14

- [X] 1 有偏估计和无偏估计

> Q: defaultdict
> A: 参考链接 https://www.cnblogs.com/HumbleHater/p/16094151.html#1%E5%8C%BA%E5%88%AB%E4%B8%8E%E4%BA%A7%E7%94%9F%E7%9A%84%E5%8E%9F%E5%9B%A0

无偏估计
无偏估计是用样本统计量来估计总体参数时的一种无偏推断。估计量的数学期望等于被估计参数的真实值，则称此估计量为被估计参数的无偏估计，即具有无偏性，是一种用于评价估计量优良性的准则。无偏估计的意义是：在多次重复下，它们的平均数接近所估计的参数真值。

有偏估计
有偏估计（biased estimate）是指由样本值求得的估计值与待估参数的真值之间有系统误差，其期望值不是待估参数的真值。在统计学中，估计量的偏差（或偏差函数）是此估计量的期望值与估计参数的真值之差。偏差为零的估计量或决策规则称为无偏的。否则该估计量是有偏的。

* [X] 2 Pytorch中train和eval模式的区别

> Q:
> A: 参考链接：
>
> 1. https://blog.csdn.net/qq_39463274/article/details/105347842(简单介绍，主要是bn和dropout)
> 2. https://zhuanlan.zhihu.com/p/668433409
> 3. https://www.bilibili.com/video/BV1n14y1v7ap?spm_id_from=333.788.videopod.sections&vd_source=e6a26642f7f1d14e5b11a109a4dfffe9
