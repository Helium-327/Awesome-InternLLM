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

##  3. Linux使用

### 3.1 快速认识常用命令

> `ls`: 列出目录中的文件或子目录
>
> `cd `：改变当前工作目录的路径
>
> `pwd`：显示当前工作目录的路径
>
> `cp`：复制文件或目录
>
> `mv`：移动或重命名文件或目录
>
> `rm`：删除文件或目录
>
> `mkdir`：创建一个新的目录
>
> `touch`：创建一个空文件或修改文件的时间戳
>
> `cat`：查看文件内容或合并文件
>
> `more / less`：分页显示文件内容
>
> `head / tail`：显示文件的开头或结尾部分
>
> `grep`：在文件中搜索指定的字符串
>
> `find`：在目录树中搜索文件
>
> `tar`：打包或解压文件
>
> `gzip / gunzip`：压缩或解压缩文件
>
> `chmod`：修改文件或目录的权限
>
> `chown`：改变文件或目录的所有者
>
> `ps`：显示当前运行的进程
>
> `top`：显示实时进程和系统负载信息
>
> `kill`：发送信号到进程
>
> `df`：显示文件系统的磁盘空间使用情况
>
> `du`：显示文件或目录的磁盘使用情况
>
> `free`：显示内存和交换空间的使用情况
>
> `ping`：测试网络连接
>
> `ssh`：安全地访问远程服务器
>
> `scp`：安全地复制文件到远程服务器
>
> `wget`：从网络上下载文件
>
> `curl`：传输数据的工具，支持多种协议
>
> `sudo`：以超级用户身份执行命令

###  3.2 服务器端查看系统信息的命令

`hostname`：开发机名称

`uname -a`：查看开发机内核信息

`lsb_release -a`：查看开发机版本信息

`nvidia-smi`：查看GPU的信息

### 3.3 端口映射

什么是端口映射？

> 端口映射可以将外网中的任意端口映射搭配内网中的相应端口，实现内网与外网之间的通信。通过端口映射可以从外网范文内网中的服务或应用，实现跨越网络的便捷通信。

为什么要进行端口映射？

> 因为开发机Web IDE中运行web_demo时，直接访问开发机内 http/https 服务可能会遇到代理问题，外网链接的**ui资源**没有被加载完全。
>
> 为了解决这个问题，我们需要对运行web_demo的连接进行端口映射，将**外网链接映射到我们本地主机**，

## 4.  `Linux`基础命令 （详解版）

> 在继续阅读之前建议先快速了解Linux常用命令  —>[ 3.1 快速认识常用命令](####3.1 快速认识常用命令)

### 4.1 文件管理

#### 4.1.1 `touch`

![image-20240718122905138](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181229192.png)

#### 4.1.2 `mkdir`

![image-20240718122927764](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181229803.png)

#### 4.1.3 `cd` 

![image-20240718123010163](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181230201.png)

#### 4.1.4 `pwd`

![image-20240718123115840](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181231877.png)

#### 4.1.5 `cat`

> `cat`命令可以查看文件里面的内容，更多的使用命令可以使用`--help`命令查看：

#### 4.1.6 `vi` or `vim`

> 进入编辑模式可以使用`i`，vim的方便之处就是可以在终端进行简单的文件修改。
>
> vi 或者 vim 有三种模式：

![](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181237006.png)

#### 4.1.7  `cp` 和 `ln`

> `cp`:复制文件或文件夹

- 复制文件：`cp 源文件 目标文件`
- 复制目录：`cp -r 源目录 目标目录`

> `ln`：创建类似`windows`的快捷方式，`ln` [参数] [源文件或目录] [目标文件或目录]

- -s：创建软链接（符号链接）也是最常用的；
- -f：强制执行，覆盖已存在的目标文件；
- -i：交互模式，文件存在则提示用户是否覆盖；
- -n：把符号链接视为一般目录；
- -v：显示详细的处理过程。

#### 4.1.8 `mv`和`rm`

> `mv`命令和`rm`命令的使用方式很相似
>
> `mv`命令：移动文件或者目录的，同时还可以进行重命名
>
> `rm`：删除文件或者目录的

`mv`命令常用参数：

- `-i`：交互模式，覆盖前询问。
- `-f`：强制覆盖。
- `-u`：只在源文件比目标文件新时才进行移动。

`rm`命令常用参数：

- `-i`：交互模式，删除前询问。
- `-f`：强制删除，忽略不存在的文件，不提示确认。
- `-r`：递归删除目录及其内容。

#### 4.1.9 `find`

> `find`命令是Linux系统中一个强大的文件搜索工具，它可以在指定的目录及其子目录中查找符合条件的文件或目录，并执行相应的操作。

`find [查找路径] `参数：

- `-name [文件名]` ：按文件名进行查找 
- `-type [文件类型 ]`：按文件类型进行查找，`f`：文件
- `-size [文件大小]` ：按文件大小查找，`+100M`：大于100M的文件
- `-mtime`/ `-atime`/ `-ctime`：按修改时间进行查找。例如，`find /path/to/directory -mtime -7`将查找指定目录及其子目录中在7天内修改过的文件。
- `-pern`：按权限进行查找
- `-user`：按用户进行查找
- `-group`：按组进行查找
- `-exec`：对找到的对象执行操作，例如：`find /path/to/directory -name "*.txt" -exec rm {} \;`将删除找到的所有以`.txt`结尾的文件。

#### 4.1.10 `ls`

> `ls`命令可以用来列出目录的内容以及**详细信息**。

常用参数：

- `-a`：显示所有文件和目录，包括隐藏文件（以`.`开头的文件或目录）。
- `-l`：以长格式显示详细信息，包括文件权限、所有者、大小、修改时间等。
- `-h`：与`-l`结合使用，以人类可读的方式显示文件大小（如`K`、`M`、`G`等）。
- `-R`：递归列出子目录的内容。
- `-t`：按文件修改时间排序显示。

#### 4.1.11 `sed`

> `sed`命令是一种流编辑器，主要用于文本处理，在处理复杂的文件操作时经常用到。

参数说明：

- `-e<script>` 或 `--expression=<script>`：直接在命令行中指定脚本进行文本处理。
- `-f<script文件>` 或 `--file=<script文件>`：从指定的脚本文件中读取脚本进行文本处理。
- `-n` 或 `--quiet` 或 `--silent`：仅打印经过脚本处理后的输出结果，不打印未匹配的行。

动作说明：

- `a`：在当前行的下一行添加指定的文本字符串

- `c`：用指定的文本字符串替换指定范围内的行

- `d`：删除指定的行

- `i`：在当前行的上一行添加指定的文本字符串

- `p`：打印经过选择的行。通常与`-n`参数一起使用，只打印匹配的行

- `s`：使用正则表达式进行文本替换。

    ![](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181336374.png)

#### 4.1.12 `grep`

> 一个强大的文本搜索工具，通常与管道符`>` `<` `|`等连用

参数：

- `-i`：忽略大小写进行搜索。
- `-v`：反转匹配，即显示不匹配的行。
- `-n`：显示行号。
- `-c`：统计匹配的行数。

![image-20240718134011422](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181340477.png)

### 4.2 进程管理

> **进程管理**命令是进行系统监控和进程管理时的重要工具
>
> 常用的进行管理命令：
>
> - `ps`：查看正在运行的进程
>
> - `top`：动态显示正在运行的进程
>
> - `pstree`：树状查找正在运行的进程
>
> - `pgrep`：用于查找进程
>
> - `nice`：更改进程的优先级
>
> - `jobs`：显示进程相关的信息
>
> - `bg`和`fg`：将进程调入后台
> - `kill`：杀死进程

#### 4.2.1 `ps`

> 列出当前系统中的进程。使用不同的选项可以显示不同的进程信息

`ps -aux`：显示系统所有进程的详细信息

![image-20240718134912121](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181349213.png)

#### 4.2.2 `top`

> 动态显示系统中进程的状态。它会实时更新进程列表，显示CPU和内存使用率最高的进程。

`top`：动态显示系统进程信息

![image-20240718135028390](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181350480.png)

#### 4.2.3 `pstree`

> 以树状图的形式显示当前运行的进程及其父子关系。

`pstree`：显示进程树

![image-20240718135607714](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181356793.png)

#### 4.2.4 `pgrep`

> 查找匹配条件的进程。可以根据进程名、用户等条件查找进程

`pgrep -u username`：查找特定用户的所有进程

![image-20240718135821442](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407181358511.png)

#### 4.2.5 `nice`

> 更改进程的优先级。`nice` 值越低，进程优先级越高。

`nice -n 10 long-running-command`：以较低优先级运行一个长时间运行的命令

#### 4.2.6 `jobs`

> 显示当前终端会话中的作业列表，包括后台运行的进程。

#### 4.2.7 `bg`和`fg`：

> `bg` 将挂起的进程放到后台运行，`fg` 将后台进程调回前台运行。

#### 4.2.8 `kill`

> 发送信号到指定的进程，通常用于杀死进程

> `kill` 命令默认发送 `SIGTERM` 信号，如果进程没有响应，可以使用`-9`使用`SIGKILL` 信号强制杀死进程。`SIGTERM`（Signal Termination）信号是Unix和类Unix操作系统中用于请求进程终止的标准信号。当系统或用户想要优雅地关闭一个进程时，通常会发送这个信号。与`SIGKILL`信号不同，`SIGTERM`信号可以被进程捕获并处理，从而允许进程在退出前进行清理工作。（来源于网络）

`kill PID`：杀死指定的进程ID

`kill -9 PID`：强制杀死进程

#### 4.2.9 `nvidia-smi`

`nvidia-smi`：显示`GPU`状态的摘要信息

`nvidia-smi -l 1`：显示详细的`GPU`状态信息

`nvidia-smi -h`：显示`GPU`的帮助信息

`nvidia-smi pmon`：列出所有`GPU`并显示他们的PID和进程名称

`nvidia-smi --id=0 -ex_pid=12345`：强制结束`GPU 0`上的`PID`为`12345`的进程

`nvidia-smi -pm 1`：设置所有`GPU`为性能模式

`nvidia-smi -i 0 -pm 1`：设置`GPU 0 `为性能模式

`nvidia-smi --id=0 -r`：重启`GPU 0`

### 4.3 AI环境配置

#### 4.3.1 `conda`

> Conda 是一个开源的包管理和环境管理系统，可在 Windows、macOS 和 Linux 上运行。它快速安装、运行和更新软件包及其依赖项。使用 Conda，您可以轻松在本地计算机上创建、保存、加载和切换不同的环境。

⏱️相关的笔记：

- [`conda` 笔记](https://uagpgtqm72r.feishu.cn/wiki/TMEewY79Li8XAIkkHYpcvOmRnOb?from=from_copylink)

- [自己写的`wsl 环境配置脚本`，可以在`ubuntu`上运行]([Helium-327/AwesomeTools-wsl4AI: 一些 为wsl自动配置zsh 和 AI环境的脚本 (github.com)](https://github.com/Helium-327/AwesomeTools-wsl4AI))



