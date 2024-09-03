# `LMDeploy`部署

🏰官方教程文档：https://github.com/InternLM/Tutorial/tree/camp3/docs/L1/Demo

## `Cli Demo`部署 `InternLM2-Chat-1.8B`

### 环境配置：

```shell
# 创建环境
conda create -n demo python=3.10 -y
# 激活环境
conda activate demo
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

```shell
# requirements.txt
transformers==4.38      # 预训练工具
sentencepiece==0.1.99   # 将文本转换为子词或词片段
einops==0.8.0           # 张亮操作的库，重塑、转置、压缩、扩展
protobuf==5.27.2        # 数据序列化协议
accelerate==0.33.0      # 简化训练过程
streamlit==1.37.0       # 快速构建交互式web应用
```

### 编写代码

```python
# cli_demo.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```

### 运行结果

![image-20240728125830755](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281258804.png)

## `Streamlit Web Demo` 部署 `InternLM2-Chat-1.8B` 模型

### 下载仓库

```shell
cd /root/demo
git clone https://github.com/InternLM/Tutorial.git

```

### 开启`streamlit`服务

```shell

## remote 
streamlit run /root/demo/Tutorial/tools/streamlit_demo.py --server.address 127.0.0.1 --server.port 6006

```



![image-20240728125853820](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281258866.png)

### 本地端口映射

```shell
## local
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
```



![image-20240728125904867](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281259902.png)

![image-20240728131400546](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281314640.png)

### 【**TASK**】 生成`300`字的故事

![image-20240728134405214](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281344318.png)

## `LMDeploy` 部署 `InternLM-XComposer2-VL-1.8B` 模型

### `InternLM-XComposer2`

> InternLM-XComposer2 是一款基于 InternLM2 的视觉语言大模型，其擅长自由形式的文本图像合成和理解。其主要特点包括：
>
> - 自由形式的交错文本图像合成：`InternLM-XComposer2` 可以根据大纲、详细文本要求和参考图像等不同输入，生成连贯且上下文相关，具有交错图像和文本的文章，从而实现高度可定制的内容创建。
> - 准确的视觉语言问题解决：InternLM-XComposer2 基于自由形式的指令准确地处理多样化和具有挑战性的视觉语言问答任务，在识别，感知，详细标签，视觉推理等方面表现出色。
> - 令人惊叹的性能：基于 InternLM2-7B 的InternLM-XComposer2 在多个基准测试中位于开源多模态模型第一梯队，而且在部分基准测试中与 GPT-4V 和 Gemini Pro 相当甚至超过它们。

### `LMDeploy`

> LMDeploy 是一个用于压缩、部署和服务 LLM 的工具包，由 MMRazor 和 MMDeploy 团队开发。它具有以下核心功能：
>
> - 高效的推理：LMDeploy 通过引入持久化批处理、块 KV 缓存、动态分割与融合、张量并行、高性能 CUDA 内核等关键技术，提供了比 vLLM 高 1.8 倍的推理性能。
> - 有效的量化：LMDeploy 支持仅权重量化和 k/v 量化，4bit 推理性能是 FP16 的 2.4 倍。量化后模型质量已通过 OpenCompass 评估确认。
> - 轻松的分发：利用请求分发服务，LMDeploy 可以在多台机器和设备上轻松高效地部署多模型服务。
> - 交互式推理模式：通过缓存多轮对话过程中注意力的 k/v，推理引擎记住对话历史，从而避免重复处理历史会话。
> - 优秀的兼容性：LMDeploy支持 KV Cache Quant，AWQ 和自动前缀缓存同时使用。

### 环境安装

```shell
conda activate demo
pip install lmdeploy[all]==0.5.1
pip install timm==1.0.7
```

### 一条命令部署

```shell
lmdeploy serve gradio /share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-1_8b --cache-max-entry-count 0.1
```

![image-20240728135359492](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281353589.png)

### 【**TASK**】图片对话

![image-20240728135341618](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281353755.png)

## `LMDeploy` 部署 `InternVL2-2B` 模型

```shell
lmdeploy serve gradio /share/new_models/OpenGVLab/InternVL2-2B --cache-max-entry-count 0.1
```

### 【**TASK**】图片对话

![image-20240728133740369](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202407281337515.png)

