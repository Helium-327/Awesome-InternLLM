# XTuner 微调个人小助手认知

🔗完整教程：https://github.com/InternLM/Tutorial/tree/camp3/docs/L1/XTuner

 ## 环境配置

### conda 环境

```shell
conda create -n xtuner0121 python=3.10 -y

# 激活虚拟环境（注意：后续的所有操作都需要在这个虚拟环境中进行）
conda activate xtuner0121

## 安装一些必要的库
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
# 安装其他依赖
pip install transformers==4.39.3
pip install streamlit==1.36.0
```

### 安装 XTuner

```shell
mkdir -p /root/InternLM/code

cd /root/InternLM/code

git clone -b v0.1.21  https://github.com/InternLM/XTuner /root/InternLM/code/XTuner

# 进入到源码目录
cd /root/InternLM/code/XTuner
conda activate xtuner0121

# 执行安装
pip install -e '.[deepspeed]'

# 查看版本
xtuner version

# 查看帮助
xtuner help
```

## 指令微调

### 准备数据

```shell
mkdir -p datas
touch datas/assistant.json
```

```python
# xtuner_generate_assistant.py
import json

# 设置用户的名字
name = '伍鲜同志'  # TODO: 修改为自己的名字
# 设置需要重复添加的数据次数
n = 8000

# 初始化数据
data = [
    {"conversation": [{"input": "请介绍一下你自己", "output": "我是{}的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦".format(name)}]},
    {"conversation": [{"input": "你在实战营做什么", "output": "我在这里帮助{}完成XTuner微调个人小助手的任务".format(name)}]}
]

# 通过循环，将初始化的对话数据重复添加到data列表中
for i in range(n):
    data.append(data[0])
    data.append(data[1])

# 将data列表中的数据写入到'datas/assistant.json'文件中
with open('datas/assistant.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(data, f, ensure_ascii=False, indent=4)

```

### 准备配置文件

```python
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, dataset_map_fns, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = '/mnt/d/AI_Research/WS-HUB/WS-InternLM/InterLM_XTuner/XTuner/Shanghai_AI_lab/internlm2-chat-1_8b'		#TODO:change path
use_varlen_attn = False

# Data
alpaca_en_path = '/mnt/d/AI_Research/WS-HUB/WS-InternLM/InterLM_XTuner/XTuner/data/assistant.json'  #TODO:change path
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 4
max_epochs = 3
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = SYSTEM_TEMPLATE.alpaca
evaluation_inputs = [
    '请介绍你自己', 'Please introduce yourself',  #TODO:change your inputs
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
alpaca_en = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,  		#TODO:change 
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

```

### 启动微调

```shell
xtuner train path/to/internlm2_chat_1_8b_qlora_alpaca_e3_copy.py
```

#### ⚠️ 使用 WSL2 进行微调时踩的坑

- 先看报错：

  ```shell
  [2024-09-02 23:42:30,446] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  09/02 23:42:31 - mmengine - WARNING - WARNING: command error: 'CUDA_HOME does not exist, unable to compile CUDA op(s)'!
  09/02 23:42:31 - mmengine - WARNING -
  Arguments received: ['xtuner', 'train', './internlm2_chat_1_8b_qlora_alpaca_e3_copy.py']. xtuner commands use the following syntax:
  
      xtuner MODE MODE_ARGS ARGS
  
      Where   MODE (required) is one of ('list-cfg', 'copy-cfg', 'log-dataset', 'check-custom-dataset', 'train', 'test', 'chat', 'convert', 'preprocess', 'mmbench', 'eval_refcoco')
              MODE_ARG (optional) is the argument for specific mode
              ARGS (optional) are the arguments for specific command
  
  Some usages for xtuner commands: (See more by using -h for specific command!)
  
      1. List all predefined configs:
          xtuner list-cfg
      2. Copy a predefined config to a given path:
          xtuner copy-cfg $CONFIG $SAVE_FILE
      3-1. Fine-tune LLMs by a single GPU:
          xtuner train $CONFIG
      3-2. Fine-tune LLMs by multiple GPUs:
          NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
      4-1. Convert the pth model to HuggingFace's model:
          xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
      4-2. Merge the HuggingFace's adapter to the pretrained base model:
          xtuner convert merge $LLM $ADAPTER $SAVE_PATH
          xtuner convert merge $CLIP $ADAPTER $SAVE_PATH --is-clip
      4-3. Split HuggingFace's LLM to the smallest sharded one:
          xtuner convert split $LLM $SAVE_PATH
      5-1. Chat with LLMs with HuggingFace's model and adapter:
          xtuner chat $LLM --adapter $ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
      5-2. Chat with VLMs with HuggingFace's model and LLaVA:
          xtuner chat $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --image $IMAGE --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
      6-1. Preprocess arxiv dataset:
          xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
      6-2. Preprocess refcoco dataset:
          xtuner preprocess refcoco --ann-path $RefCOCO_ANN_PATH --image-path $COCO_IMAGE_PATH --save-path $SAVE_PATH
      7-1. Log processed dataset:
          xtuner log-dataset $CONFIG
      7-2. Verify the correctness of the config file for the custom dataset:
          xtuner check-custom-dataset $CONFIG
      8. MMBench evaluation:
          xtuner mmbench $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $MMBENCH_DATA_PATH
      9. Refcoco evaluation:
          xtuner eval_refcoco $LLM --llava $LLAVA --visual-encoder $VISUAL_ENCODER --prompt-template $PROMPT_TEMPLATE --data-path $REFCOCO_DATA_PATH
      10. List all dataset formats which are supported in XTuner
  
  Run special commands:
  
      xtuner help
      xtuner version
  
  GitHub: https://github.com/InternLM/xtuner
  ```

- 报错原因：

  ```shell
  command error: 'CUDA_HOME does not exist, unable to compile CUDA op(s)'!
  ```

*为什么？明明我环境里面安装了torch和cuda的，为什么会检测不出CUDA呢？*

> 对于`CUDA_HOME `一开始，我以为是配置文件里面的某个参数。但是没有找到。后来无意间执行了一下`nvcc -V`发现，报错了。这才会想起来，前段时间电脑刚重装了系统，可能Nvidia的驱动都没有装全
>
> 那会不会是`nvcc`出了问题，导致无法训练呢？什么是`nvcc`，和CUDA有什么关系，与nvidia-smi的区别是什么？
>
> - 推荐一篇博文，讲的很详细：[nvidia-smi nvcc -V 及 CUDA、cuDNN 安装-CSDN博客](https://blog.csdn.net/qq_28087491/article/details/132635794)
>
> 简单来说就是，Nvidia的驱动API由两部分组成
>
> - CUDA Driver API
>
>   > 这个其实对应的就是最常用的`nvidia-smi`
>   >
>   > 官网下载链接：https://www.nvidia.cn/geforce/drivers/
>
> - CUDA Runtime API
>
>   > 与之对应的其实是`CUDA Toolkit` 工具包
>
> 什么是nvcc?
>
> > `nvcc` 为 `NVIDIA Cuda compiler driver` 的缩写，是 NVIDIA 的 CUDA 编译器驱动程序，是用于编译 CUDA 代码的工具。它主要用户将包含 CUDA 扩展的 C/C++ 代码编译为能够在 NVIDIA GPU 上运行的可执行文件或库。
> >
> > `nvcc-V` 的结果是对应 CUDA Runtime API，如果`CUDA Runtime API`没有安装，执行`nvcc -V`就会报错。

到此，问题找到，即系统缺少`cuda tookit` 工具包。对于`WSL2`，官网提示需要手动安装`cuda tookit`工具包

- 官网地址：[CUDA Toolkit 12.6 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network)

- 手动安装指令：

  安装之前需要先用`nvidia-smi`看一下当前驱动的最高版本

  ![image-20240903103343298](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409031033382.png)

  ```shell
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-6   #TODO：更改版本号
  ```

### 训练完成

![image-20240903103448609](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409031034682.png)

## 模型合并

### 格式转换

> 模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 HuggingFace 格式文件，那么我们可以通过以下命令来实现一键转换。

使用 `xtuner convert pth_to_hf` 命令来进行模型格式转换。

参数：

- `--fp32`：代表以fp32的精度开启，假如不输入则默认为fp16
- `--max-shard-size {GB}`：代表每个权重文件最大的大小（默认为2GB）

```shell
# 先获取最后保存的一个pth文件
pth_file=`ls -t ./work_dirs/internlm2_chat_1_8b_qlora_alpaca_e3_copy/*.pth | head -n 1`
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm2_chat_1_8b_qlora_alpaca_e3_copy.py ${pth_file} ./hf
```

转换完成后，可以看到模型被转换为 HuggingFace 中常用的 `.bin` 格式文件，这就代表着文件成功被转化为 `HuggingFace` 格式了。

此时，`hf` 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”

> 可以简单理解：LoRA 模型文件 = Adapter

### 模型合并

对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（Adapter），训练完的这个层最终还是要与原模型进行合并才能被正常的使用。

> 全量微调（full）之后，不需要进行合并。

`xtuner`合并指令：`xtuner convert merge`

准备三个路径：

- `LLM` : 源模型路径
- `ADAPTER`：训练好的Adapter层路径（模型格式转换后的）
- `SAVE_PAT`：最终的保存路径

参数：

- `--max-shard-size {GB}`：代表每个权重文件最大的大小（默认为2GB）
- `--device {device_name}`：这里指的就是device的名称，可选择的有cuda、cpu和auto，默认为cuda即使用gpu进行运算
- `--is-clip`：这个参数主要用于确定模型是不是CLIP模型，假如是的话就要加上，不是就不需要添加

### 最终效果

![image-20240903110229613](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409031102692.png)

