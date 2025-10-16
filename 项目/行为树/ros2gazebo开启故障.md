针对nav2的第一个demo
相关讨论在https://github.com/ros-navigation/navigation2/issues/2757


Ubuntu2204 humble在使用launch运行gazebo的时候

```cobol
ros2 launch gazebo_ros gazebo.launch.py 
```

电脑死机，进程结束。

解决方法：

在launch之前先source，设置环境变量

```cobol
source /usr/share/gazebo/setup.bash
```

然后再运行

```cobol
ros2 launch gazebo_ros gazebo.launch.py
```

完美解决问题。

为了以后可以不用每一次[ros2](https://so.csdn.net/so/search?q=ros2&spm=1001.2101.3001.7020) launch之前都source

在主文件夹下按Ctrl +h显示隐藏的文件夹和文件，然后打开.bashrc文件

![](https://i-blog.csdnimg.cn/blog_migrate/5453f0a2a3b39d24ce671a10ec78b8e3.png)

在最后一行加入即可

![](https://i-blog.csdnimg.cn/blog_migrate/7e99ae3f2d1b36d79f5d9135b1a71865.png)

```cobol
source /usr/share/gazebo/setup.bash
```