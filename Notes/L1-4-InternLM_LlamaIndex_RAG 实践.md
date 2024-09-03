# InternLM + LlamaIndex RAG 实践

## RAG

### 两种给模型注入新知识的方式

- 内部方式——更新模型的权重

  > 对模型进行训练，代价较大

- 外部方式——不改变权重，给模型注入额外的上下文或者外部信息

  > 只给模型引入额外的信息。实现更容易

### RAG原理

> 个人理解：
>
> - 将文本知识库转换向量知识库，并使其能被索引
> - 用户的输入会通过嵌入模型编码成向量，并在上述知识库中检索
> - 输出向量知识库中与输入相关的回答

![image-20240902203229002](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409022032047.png)

### 效果对比

![image-20240902203258160](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409022032189.png)

## 实践开始

### 配置环境

- 安装`Llamaindex`

  ```shell
  conda activate llamaindex
  
  pip install llama-index==0.10.38 llama-index-llms-huggingface==0.2.0 "transformers[torch]==4.41.1" "huggingface_hub[inference]==0.23.1" huggingface_hub==0.23.1 sentence-transformers==2.7.0 sentencepiece==0.2.0
  ```

- 下载 `Sentence Transformer`模型

  ```python
  # download_hf.py
  
  import os
  
  # 设置环境变量
  os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
  
  # 下载模型
  os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
  ```

- 下载`NLTK`相关资源

  ```shell
  git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
  
  cd nltk_data
  
  mv packages/*  ./
  
  cd tokenizers
  
  unzip punkt.zip
  
  cd ../taggers
  
  unzip averaged_perceptron_tagger.zip
  ```

## LlamaIndex HuggingFaceLLM

```python
# llamaindex_internlm.py

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
llm = HuggingFaceLLM(
    model_name="path/to/internlm2-chat-1_8b", # TODO:需要修改路径
    tokenizer_name="path/to/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
print(rsp)
```

## LlamaIndex RAG

- 安装llama库

  ```shell
  pip install llama-index-embeddings-huggingface==0.2.0 llama-index-embeddings-instructor==0.1.3
  ```

- 获取知识库

  ```shell
  mkdir data
  cd data
  git clone https://github.com/InternLM/xtuner.git
  mv xtuner/README_zh-CN.md ./
  ```

- 运行RAG:

  ```python
  # llamaindex_RAG.py
  
  from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
  
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.llms.huggingface import HuggingFaceLLM
  
  #初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
  embed_model = HuggingFaceEmbedding(
  #指定了一个预训练的sentence-transformer模型的路径
      model_name="/root/model/sentence-transformer"
  )
  #将创建的嵌入模型赋值给全局设置的embed_model属性，
  #这样在后续的索引构建过程中就会使用这个模型。
  Settings.embed_model = embed_model
  
  llm = HuggingFaceLLM(
      model_name="path/to/model/internlm2-chat-1_8b", # TODO: 修改路径
      tokenizer_name="path/to/model/internlm2-chat-1_8b",  
      model_kwargs={"trust_remote_code":True},
      tokenizer_kwargs={"trust_remote_code":True}
  )
  #设置全局的llm属性，这样在索引查询时会使用这个模型。
  Settings.llm = llm
  
  #从指定目录读取所有文档，并加载数据到内存中
  documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
  #创建一个VectorStoreIndex，并使用之前加载的文档来构建索引。
  # 此索引将文档转换为向量，并存储这些向量以便于快速检索。
  index = VectorStoreIndex.from_documents(documents)
  # 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
  query_engine = index.as_query_engine()
  response = query_engine.query("xtuner是什么?")
  
  print(response)
  ```

## LlamaIndex Web

- 安装`streamlit`

  ```shell
  pip install streamlit==1.36.0
  ```

- 运行web app：

  ```python
  import streamlit as st
  from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
  from llama_index.embeddings.huggingface import HuggingFaceEmbedding
  from llama_index.llms.huggingface import HuggingFaceLLM
  
  st.set_page_config(page_title="llama_index_demo", page_icon="🦜🔗")
  st.title("llama_index_demo")
  
  # 初始化模型
  @st.cache_resource
  def init_models():
      embed_model = HuggingFaceEmbedding(
          model_name="path/to/model/sentence-transformer" # TODO: 修改地址
      )
      Settings.embed_model = embed_model
  
      llm = HuggingFaceLLM(
          model_name="path/to/model/internlm2-chat-1_8b",				# TODO: 修改地址
          tokenizer_name="path/to/model/internlm2-chat-1_8b",			# TODO: 修改地址
          model_kwargs={"trust_remote_code": True},
          tokenizer_kwargs={"trust_remote_code": True}
      )
      Settings.llm = llm
  
      documents = SimpleDirectoryReader("path/to/llamaindex_demo/data").load_data()   		# TODO: 修改地址
      index = VectorStoreIndex.from_documents(documents)
      query_engine = index.as_query_engine()
  
      return query_engine
  
  # 检查是否需要初始化模型
  if 'query_engine' not in st.session_state:
      st.session_state['query_engine'] = init_models()
  
  def greet2(question):
      response = st.session_state['query_engine'].query(question)
      return response
  
        
  # Store LLM generated responses
  if "messages" not in st.session_state.keys():
      st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]    
  
      # Display or clear chat messages
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.write(message["content"])
  
  def clear_chat_history():
      st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]
  
  st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
  
  # Function for generating LLaMA2 response
  def generate_llama_index_response(prompt_input):
      return greet2(prompt_input)
  
  # User-provided prompt
  if prompt := st.chat_input():
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.write(prompt)
  
  # Gegenerate_llama_index_response last message is not from assistant
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
              response = generate_llama_index_response(prompt)
              placeholder = st.empty()
              placeholder.markdown(response)
      message = {"role": "assistant", "content": response}
      st.session_state.messages.append(message)
  ```

## Task 

- 问题： 请介绍一下中国

- 不用RAG:

  ```
  中国是一个历史悠久、文化灿烂、人口众多的国家，拥有丰富的自然资源和广阔的国土面积。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。
  　　中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。
  　　中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。
  　　中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。
  　　中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。中国是一个统一的多民族国家，拥有56个民族，56个民族共同构成了中华民族大家庭。
  ```

  ![image-20240902212119962](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409022121046.png)

- 使用RAG：

  ```
  中国是一个拥有悠久历史和灿烂文化的国家，也是世界上人口最多的国家。中国有着丰富的自然资源和独特的地理环境，拥有丰富的文化遗产和历史遗迹。中国是一个统一的多民族国家，拥有56个民族，每个民族都有自己独特的文化和传统。中国是一个开放的国家，积极参与全球治理，推动国际合作和交流。中国是一个社会主义国家，坚持中国共产党的领导，实行人民代表大会制度，实行公有制为主体、多种所有制经济共同发展的基本经济制度，实行按劳分配为主体、多种分配方式并存的分配制度。中国是一个和平、合作、负责任的大国，在国际事务中发挥着重要作用。中国是一个负责任的大国，积极参与全球治理，推动国际合作和交流，为世界和平与发展作出了重要贡献。
  ---------------------
  Given the context information and not prior knowledge, answer the query.
  Query: 请介绍一下中国
  Answer: 中国是一个拥有悠久历史和灿烂文化的国家，也是世界上人口最多的国家。中国有着丰富的自然资源和独特的地理环境，拥有丰富的文化遗产和历史遗迹。中国是一个统一的多民族国家，拥有56个民族，每个民族都有自己独特的文化和传统。中国是一个开放的国家，积极参与全球治理，推动国际合作和交流。中国是一个社会主义国家，坚持中国共产党
  ```

  

  ![image-20240902212307618](https://raw.githubusercontent.com/Helium-327/PicGo/main/win/markdown/202409022123662.png)