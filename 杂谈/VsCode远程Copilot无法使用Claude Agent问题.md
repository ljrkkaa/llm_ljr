 

最近我突然发现vscode Copilot中Claude模型突然没了，我刚充的钱啊！没有Claude我还用啥Copilot

很多小伙伴知道要开代理，开完代理后确实Claude会出来，本地使用是没有任何问题的，但是如果使用远程ssh的话，会出现访问异常，连接不上的情况。这时候很多小伙伴就在网上寻找方法，在vscode setting中添加这么一段代码。[可以看看这篇博客](https://blog.csdn.net/qq_32057921/article/details/145764603 "可以看看这篇博客")

```XML
"http.proxy": "http://127.0.0.1:1082",    "remote.extensionKind": {                 "GitHub.copilot": [            "ui"         ],        "GitHub.copilot-chat": [            "ui"        ],        "pub.name": [            "ui"        ]    }
```

代理设置在本地回还地址，然后强制copilot也在本地上运行，这时候你会发现Claude模型出来了，ask也可以正常询问，但是Agent模式无法正常编辑，会有一段这么个提醒：

copilotAllow edits to sensitive files?The model wants to edit files outside of your workspace

这是提醒你工作区错误，但是明明发现我的工作区没有错。这其中的问题就出在了"GitHub.copilot": \["ui" \],  
    "GitHub.copilot-chat": \["ui"\],这两行代码会强制你的copilot在本地运行，然后你的远程路径在本地是无法识别的，所以就告诉你Workspace异常。

**那么如何解决呢？**

首先我们的代理地址（我这里的端口是1082，根据自己的情况来）是需要的，只是加错了位置。我们先把本地的配置文件里面的这些代码注释掉

![](https://i-blog.csdnimg.cn/direct/d76c0acee89c4a0e885faff2722bc645.png)

![](https://i-blog.csdnimg.cn/direct/cfc0cb4975584b468050519734c24a8e.png)

然后，我们打开ssh的配置文件，加入这么一段

![](https://i-blog.csdnimg.cn/direct/de2c42de10cb47d197f9dcc81ca122de.png)

将本地代理的端口，在远程穿透回来，此时远程服务器也用上了代理

然后再在远程的settings.json中配置代理端口![](https://i-blog.csdnimg.cn/direct/024e9b66e1834b198ec8918e0a82ebf0.png)

```XML
{    "http.proxy": "http://127.0.0.1:1082",    "http.proxyStrictSSL": false,    "remote.extensionKind": {         "pub.name": [            "ui"        ]    }}
```

重启vscode，你会发现Claude又回来了QAQ![](https://i-blog.csdnimg.cn/direct/b3f7911e20a3428b92303ea6fffbc056.png)

并且Agent模型也可以正常使用了，因为工作区一直在远程上，不会出现问题

本文转自 <https://blog.csdn.net/qq_40620465/article/details/152000104>，如有侵权，请联系删除。