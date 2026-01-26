`Python` 包管理生态中存在多种工具，如 `pip`、`pip-tools`、`poetry`、`conda` 等，各自具备一定功能。

而今天介绍的`uv` 是 `Astral` 公司推出的一款基于 `Rust` 编写的 `Python` 包管理工具，旨在成为 “**Python 的 Cargo**”。

它提供了快速、可靠且易用的包管理体验，在性能、兼容性和功能上都有出色表现，为 `Python` 项目的开发和管理带来了新的选择。

1\. 为什么用uv
==========

与其他`Python`中的包管理工具相比，`uv`更像是一个全能选手，它的优势在于：

1.  **速度快**：得益于`Rust`，`uv`工具的速度让人惊艳，比如安装依赖，速度比其他工具快很多
2.  **功能全面**：`uv` 是“**一站式服务**”的工具，从安装 Python、管理虚拟环境，到安装和管理包，再到管理项目依赖，它统统都能处理得很好
3.  **前景光明**：背后有风投公司`Astral`支持，且采用了`MIT`许可，即使未来出现问题，社区也有应对的办法

使用`uv`，也可以像`NodeJS`或者`Rust`项目那样方便的管理依赖。

2\. 如何安装
========

安装 `uv` 非常简单，可以使用官方提供的安装脚本，也可以通过`pip`来安装。

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
pip install uv
```

安装之后，可以通过`uv help`命令检查是否安装成功：

![](https://img2024.cnblogs.com/blog/83005/202412/83005-20241227125034737-1020856408.png)

3\. 如何使用
========

下面演示如何使用`uv`来管理`Python`项目。

使用`uv`之前，创建一个`Python`项目对我来说就是创建一个文件夹而已。

使用`uv`之后，终于有了一些项目的感觉，对于`uv`，我使用时间也不长，疏漏或错误的地方欢迎指正！

接下来，从创建一个项目开始，演示我使用`uv`时常用的一些功能。

首先，介绍`uv`工具主要使用的两个文件：

*   `pyproject.toml`：定义项目的主要依赖，包括项目名称、版本、描述、支持的 `Python` 版本等信息
*   `uv.lock`：记录项目的所有依赖，包括依赖的依赖，且跨平台，确保在不同环境下安装的一致性。这个文件由 `uv` 自动管理，不要手动编辑

3.1. 创建项目
---------

接下来，创建一个项目，使用`uv init <project dir>`命令。

```bash
$  uv init myproject
Initialized project `myproject` at `D:\projects\python\myproject`

$  cd .\myproject\

$  ls


    目录: D:\projects\python\myproject


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        2024/12/27  12:06:08            109 .gitignore
-a----        2024/12/27  12:06:08              5 .python-version
-a----        2024/12/27  12:06:08             87 hello.py
-a----        2024/12/27  12:06:08            155 pyproject.toml
-a----        2024/12/27  12:06:08              0 README.md
```

通过`init`创建项目之后，`uv`工具贴心地帮助我们生成了一些默认文件。

其中 `hello.py` 只是一段演示用的代码，

随后我们可以根据实际的项目需要删除这个代码文件，换成自己的实际代码。

```bash
$  cat .\hello.py
def main():
    print("Hello from myproject!")


if __name__ == "__main__":
    main()
```

`pyproject.toml`中是一些项目信息：

```bash
$  cat .\pyproject.toml
[project]
name = "myproject"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []
```

**注意**，`uv init` 创建项目之后，会自动将项目使用`Git`来管理。

3.2. 操作环境
---------

创建项目之后，我们进入项目根文件夹的第一件事就是同步项目依赖。

```bash
$  uv sync
Using CPython 3.12.4 interpreter at: D:\miniconda3\envs\databook\python.exe
Creating virtual environment at: .venv
Resolved 1 package in 15ms
Audited in 0.05ms
```

同步之后，会自动查找或下载合适的 `Python` 版本，创建并设置项目的虚拟环境，构建完整的依赖列表并写入

`uv.lock` 文件，最后将依赖同步到虚拟环境中。

我们这个是新创建的项目，没有什么依赖，所以`uv.lock` 文件中的内容也比较简单。

```bash
$  ls


    目录: D:\projects\python\myproject


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        2024/12/27  12:12:39                .venv
-a----        2024/12/27  12:06:08            109 .gitignore
-a----        2024/12/27  12:06:08              5 .python-version
-a----        2024/12/27  12:06:08             87 hello.py
-a----        2024/12/27  12:06:08            155 pyproject.toml
-a----        2024/12/27  12:06:08              0 README.md
-a----        2024/12/27  12:12:39            116 uv.lock

$  cat .\uv.lock
version = 1
requires-python = ">=3.12"

[[package]]
name = "myproject"
version = "0.1.0"
source = { virtual = "." }
```

`uv sync`同步之后，就可以运行项目的代码了。

既然使用`uv`管理项目的话，我们就使用`uv`的命令来运行代码，不要像以前那样使用`python xxx.py`来运行。

我们可以试着运行项目创建时自动生成的代码。

```bash
$  uv run .\hello.py
Hello from myproject!
```

3.3. 管理依赖
---------

管理依赖是我使用`uv`工具的主要目的，使用`uv`添加依赖非常简单，和`npm`和`cargo`差不多。

```bash
$  uv add pandas
Resolved 7 packages in 3.41s
Prepared 6 packages in 4.63s
Installed 6 packages in 1.80s
 + numpy==2.2.1
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2024.2
 + six==1.17.0
 + tzdata==2024.2
```

尝试安装了一个`pandas`依赖（pandas依赖的包也自动安装了），从上面日志可以看出速度非常快。

这时再看看`uv.lock` 文件的变化。

```bash
$  cat .\uv.lock
version = 1
requires-python = ">=3.12"

[[package]]
name = "myproject"
version = "0.1.0"
source = { virtual = "." }
dependencies = [
    { name = "pandas" },
]

[package.metadata]
requires-dist = [{ name = "pandas", specifier = ">=2.2.3" }]

[[package]]
name = "pandas"
version = "2.2.3"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "numpy" },
    { name = "python-dateutil" },
    { name = "pytz" },
    { name = "tzdata" },
]

[[package]]
name = "pytz"
version = "2024.2"
source = { registry = "https://pypi.org/simple" }

```

上面的日志中我删除了很多内容，因为整体内容太多，详细记录了每个包以及它依赖的包的情况。

`uv.lock`这个文件我们不要手动去编辑它，使用`uv`工具去管理它。

引入了pandas之后，我们看看是否可以在`hello.py`中使用。

```bash
$  cat .\hello.py
import pandas as pd


def main():
    print("Hello from myproject!")
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    print(df)


if __name__ == "__main__":
    main()

$  uv run .\hello.py
Hello from myproject!
   A  B
0  1  4
1  2  5
2  3  6
```

可以正常使用安装的包`pandas`，下面在试试删除依赖会怎么样。

```bash
$  uv remove pandas
Resolved 1 package in 12ms
Uninstalled 6 packages in 1.18s
 - numpy==2.2.1
 - pandas==2.2.3
 - python-dateutil==2.9.0.post0
 - pytz==2024.2
 - six==1.17.0
 - tzdata==2024.2

$  cat .\uv.lock
version = 1
requires-python = ">=3.12"

[[package]]
name = "myproject"
version = "0.1.0"
source = { virtual = "." }
```

使用`uv remove`命令删除`pandas`包之后，也会自动删除`pandas`依赖的其他包，

我们看到`uv.lock` 文件也恢复到最初的内容。

再试试运行`hello.py`看看。

```bash
$  uv run .\hello.py
Traceback (most recent call last):
  File "D:\projects\python\myproject\hello.py", line 1, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
```

果然，无法运行了。

3.4. 区分开发和生产环境
--------------

还有一个比较常用的功能是区分**开发环境**和**生产环境**的依赖，这个功能在`NodeJS`和`Rust`中很常见。

比如，我们想把`pandas`安装到开发环境中，而把`requests`安装到生产环境中。

```bash
$  uv add --group dev pandas
Resolved 7 packages in 1.72s
Installed 6 packages in 1.39s
 + numpy==2.2.1
 + pandas==2.2.3
 + python-dateutil==2.9.0.post0
 + pytz==2024.2
 + six==1.17.0
 + tzdata==2024.2

$  uv add --group production requests
Resolved 12 packages in 2.72s
Prepared 5 packages in 1.31s
Installed 5 packages in 68ms
 + certifi==2024.12.14
 + charset-normalizer==3.4.1
 + idna==3.10
 + requests==2.32.3
 + urllib3==2.3.0
```

安装之后，`uv.lock` 文件自动添加了各个包及其依赖，这里不再赘述。

从项目的`pyproject.toml`中可以看出不同环境的包依赖。

```bash
$  cat .\pyproject.toml
[project]
name = "myproject"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[dependency-groups]
dev = [
    "pandas>=2.2.3",
]
production = [
    "requests>=2.32.3",
]
```

4\. 未来发展
========

`uv` 也可以构建和发布 `Python` 包到 `PyPi`，具体细节本篇就不展开了。

`uv` 自从发布后，团队一直致力于优先提升其跨平台的兼容性、性能和稳定性，帮助用户顺利将项目过渡到使用`uv`来管理。

长远来看，`uv` 将发展成为一个完整的 `Python` 项目和包管理器，提供一站式的开发体验，涵盖从 `Python` 安装到项目管理的各个环节，进一步简化 `Python` 项目的开发流程，提高开发效率。

本文转自 <https://www.cnblogs.com/wang_yb/p/18635441>，如有侵权，请联系删除。