# Git

> Git 是一种开源的分布式版本控制系统，广泛应用于软件开发领域，尤其是在协同工作环境中。它为程序员提供了一套必备的工具，使得团队成员能够有效地管理和跟踪代码的历史变更。下面是 Git 的主要功能和作用的规范描述：

---

🧐官网：https://git-scm.com/

🧿官方文档：[Git - Book](https://git-scm.com/book/en/v2)

🧿Git 基础：[Git 基础知识](https://aicarrier.feishu.cn/wiki/YAXRwLZxPi8Hy6k3tOQcuwAHn5g)

🧿活动教程：[Tutorial/docs/L0/Git/readme.md at camp3 · InternLM/Tutorial (github.com)](https://github.com/InternLM/Tutorial/blob/camp3/docs/L0/Git/readme.md)

---



##  `Git`中基本概念

### 工作区、暂存区和 Git 仓库区

#### 工作区

>  当我们在本地创建一个 Git 项目，或者从 GitHub 上 clone 代码到本地后，项目所在的这个目录就是“工作区”。这里是我们对项目文件进行编辑和使用的地方。

#### 暂存区

>  暂存区是 Git 中独有的一个概念，位于 .git 目录中的一个索引文件，记录了下一次提交时将要存入仓库区的文件列表信息。使用 git add 指令可以将工作区的改动放入暂存区。

#### 仓库区/本地仓库（Repository）

> 在项目目录中，.git 隐藏目录不属于工作区，而是 Git 的版本仓库。这个仓库区包含了所有历史版本的完整信息，是 Git 项目的“本体”。

### 文件状态

#### 已跟踪

> 文件已被纳入版本控制，根据其是否被修改，可以进一步分为未修改（Unmodified）、已修改（Modified）或已暂存（Staged）。

#### 未跟踪

> 文件存在于工作目录中，但还没被纳入版本控制，也未处于暂存状态。

### 主要功能

- 代码历史记录跟踪
- 团队协作
- 变更审查
- 实现机制

| 特性       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| 分布式架构 | 与集中式版本控制系统不同，Git 在每个开发者的机器上都存有完整的代码库副本，包括完整的历史记录。这种分布式的特性增强了数据的安全性和获取效率。 |
| 分支管理   | Git 的分支管理功能非常灵活，支持无缝切换到不同的开发线路（分支），并允许独立开发、测试新功能，最终通过合并操作将这些功能稳定地集成到主项目中。 |
| 快照系统   | Git 通过快照而非差异比较来管理数据。每次提交更新时，Git 实际上是在存储一个项目所有文件的快照。如果文件没有变化，Git 只是简单地链接到之前存储的文件快照。 |

## `Git`常用命令

### 基础命令

| 指令             | 描述                                       |
| ---------------- | ------------------------------------------ |
| `git config`     | 配置用户信息和偏好设置                     |
| `git init`       | 初始化一个新的 Git 仓库                    |
| `git clone`      | 克隆一个远程仓库到本地                     |
| `git status`     | 查看仓库当前的状态，显示有变更的文件       |
| `git add`        | 将文件更改添加到暂存区                     |
| `git commit`     | 提交暂存区到仓库区                         |
| `git branch`     | 列出、创建或删除分支                       |
| `git checkout`   | 切换分支或恢复工作树文件                   |
| `git merge`      | 合并两个或更多的开发历史                   |
| `git pull`       | 从另一仓库获取并合并本地的版本             |
| `git push`       | 更新远程引用和相关的对象                   |
| `git remote`     | 管理跟踪远程仓库的命令                     |
| `git fetch`      | 从远程仓库获取数据到本地仓库，但不自动合并 |
| `git command -h` | 命令行的帮助文档                           |

### 进阶指令

| `git stash`       | 暂存当前工作目录的修改，以便可以切换分支             |
| ----------------- | ---------------------------------------------------- |
| `git cherry-pick` | 选择一个提交，将其作为新的提交引入                   |
| `git rebase`      | 将提交从一个分支移动到另一个分支                     |
| `git reset`       | 重设当前 HEAD 到指定状态，可选修改工作区和暂存区     |
| `git revert`      | 通过创建一个新的提交来撤销之前的提交                 |
| `git mv`          | 移动或重命名一个文件、目录或符号链接，并自动更新索引 |
| `git rm`          | 从工作区和索引中删除文件                             |

## 配置`Git`

- **全局设置**：这些设置影响你在该系统上所有没有明确指定其他用户名和电子邮件的 Git 仓库。这是设置默认用户名和电子邮件的好方法。

    ```shell
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    ```

- **本地设置**：这些设置仅适用于特定的 Git 仓库。这对于你需要在不同项目中使用不同身份时很有用，例如区分个人和工作项目。

    ```shell
    git config --local user.name "Your Name"
    git config --local user.email "your.email@example.com"
    ```

- **查看全局配置**

    ```shell
    git config --global --list
    ```

- **查看仓库配置**

    ```shell
    git config --local --list
    ```

- **查看特定配置项**

    ```shell
    git config user.name
    git config user.email
    ```



## 使用`Git`（四部曲）

### `git add`

> 将修改过的文件添加到本地暂存区（Staging Area）。这一步是准备阶段，你可以选择性地添加文件，决定哪些修改应该被包括在即将进行的提交中。

### `git commit -m ""`

> 将暂存区中的更改提交到本地仓库。这一步是将你的更改正式记录下来，每次提交都应附带一个清晰的描述信息，说明这次提交的目的或所解决的问题。

### `git pull`

> 从远程仓库拉取最新的内容到本地仓库，并自动尝试合并到当前分支。这一步是同步的重要环节，确保你的工作基于最新的项目状态进行。在多人协作中，定期拉取可以避免将来的合并冲突。

### `git push`

> 将本地仓库的更改推送到远程仓库。这一步是共享你的工作成果，让团队成员看到你的贡献。

## 开发流程

### `Fork`目标项目

![image-20240719104311554](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191043624.png)

### 获取仓库链接 (`ssh`)

![image-20240719104412988](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191044053.png)

### `git clone `到本地

> 再进行这一步之前需要先将本地的`ssh-key`上传到个人的`github`仓库 ———> [6](##本地`ssh-key`上传到`github`仓库)

![image-20240719104621085](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191046136.png)

### 查看分支

```shell
git branch -a 
```

![image-20240719111116107](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191111166.png)

### 创建并切换分支

```shell
git checkout -b <name-branch>
```



### 查看分支内容

### 修改

### 提交修改到分支

```shell
git push origin <name-branch>
```

### 查看提交

## 本地`ssh-key`上传到`github`仓库

> 本地仓库上传需要通过`ssh-key`进行身份验证，因此为了能`git push`到云端仓库，需要先将`ssh-key`上传到`github`仓库

### 生成主机的公钥

`powershell`终端执行下列代码后，按照提示一路回车

```shell
ssh-keygen -t rsa -C "your_email@example.com"
```

参数含义：

`-t` 指定密钥类型，默认是 `rsa` ，可以省略。
`-C` 设置注释文字，比如邮箱。
`-f` 指定密钥文件存储文件名。

### 获取主机的公钥

> 生成密钥后，需要找到`/path/to/username/.ssh/id_rsa.pub`文件，该文件中存放的是生成的公钥

![image-20240719105743629](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191057679.png)

### 将公钥暴露给`github`

![image-20240719112237940](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191122079.png)

![image-20240719112146630](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191121731.png)

![image-20240719111351359](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191113397.png)

![image-20240719111435747](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407191114791.png)

> 添加完之后，即可回到 —-> [5.开发流程](##开发流程)
