# 入学篇-Linux

🧐 本次学习由`InternStudio`平台赞助

🧿算力平台：https://studio.intern-ai.org.cn/console/instance

🧿闯关流程：https://aicarrier.feishu.cn/wiki/XBO6wpQcSibO1okrChhcBkQjnsf

🧿闯关文档地址：https://github.com/InternLM/Tutorial/tree/camp3/docs/L0

🧿推荐学习资料：[计算机教育中缺失的一课 · the missing semester of your cs education (missing-semester-cn.github.io)](https://missing-semester-cn.github.io/)

---



##  1. `InternStudio`平台使用教程

平台首页：

![image-20240718101541142](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181015286.png)

创建机器：

![image-20240718101644835](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181016893.png)

选择镜像：

![image-20240718101724488](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181017555.png)

![image-20240718101930310](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181019361.png)

记得修改运行时间：

![image-20240718101803708](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181018764.png)

## 2. Linux 登录方式

### 2.1 添加ssh密钥

### 2.2 本地端创建生成密钥

> 本地端`powershell`输入`ssh-keygen -t rsa`
>
> 输入命令后**一路回车**就可以了，这里的密钥默认情况下是生成在`~/.ssh/`目录下的，`~`表示的是家目录，如果是windows就是`C:\Users\{your_username}\`。
>
> ![](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181058605.png)

1. ### 服务器端添加密钥

    - 方式一：在线添加

    > 
    >
    > ![image-20240718105230406](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181052517.png)
    >
    > ![image-20240718105239565](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181052606.png)
    >
    > ![image-20240718105334657](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181053710.png)
    >
    > 

- 方式二：写入文件



![image-20240718105935169](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181059210.png)

选择登录方式：

![image-20240718102013097](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181020156.png)

### 2.3 在线登录

根据个人爱好选择喜欢的用户界面：

- [ ] JupyterLab

- [ ] CLI

- [x] VsCode (地表最强，不接受反驳)

![image-20240718102243393](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181022470.png)

### 2.4 powershell + ssh登录：

> 复制**登录命令到powershell**，这里的37367是开发机所使用的SSH端口，一般使用的都是22端口，没有这个端口号的话是连不上SSH的，并且每个人的端口都不一样，所以如果大家在连接开发机时出现连不上的情况，那就需要检查一下是不是端口错了。

![image-20240718102606209](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181026249.png)

![image-20240718102850063](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181028096.png)

成功登录

![image-20240718102928978](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181029066.png)

#### 2.3 `VsCode` + `remote SSH` 登录

> 当然也可以使用SSH远程连接软件，例如：**Windterm、Xterminal**等。这里我们使用VScode进行远程连接，使用VScode的好处是，本身它就是代码编辑器，进行代码修改等操作时会非常方便。

![image-20240718110249247](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181102286.png)

![image-20240718110459569](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181104601.png)

输入登录命令：

![image-20240718110608275](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181106308.png)

![image-20240718110626126](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181106161.png)

> 如果将*`StrictHostKeyChecking`*` no`和*`UserKnownHostsFile`*` /dev/null`删除掉会跳出指纹验证的弹窗：





