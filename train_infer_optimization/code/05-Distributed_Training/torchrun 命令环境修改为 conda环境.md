### 1，查看真实路径

```
conda activate env
```

输出：

```
$ which python
/home/metax/miniconda3/envs/cnn/bin/python
```

### 2，编辑 torchrun

```
$ vim ~/.local/bin/torchrun
#!/home/metax/miniconda3/envs/cnn/bin/python 这里修改为上述路径
# -*- coding: utf-8 -*-
import re
import sys
from torch.distributed.run import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

保存，授权，[chmod](https://so.csdn.net/so/search?q=chmod&spm=1001.2101.3001.7020) +x ~/.local/bin/torchrun

执行torchrun -h 验证

本文转自 <https://blog.csdn.net/LABLENET/article/details/143887216>，如有侵权，请联系删除。