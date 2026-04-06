---
title: 基于langchain的个人科研小助手
top_img: /img/wallhaven-427p6y_1280x720.png
abbrlink: ca5ce5aa
---

>注：该笔记基于在Datawhale第68期组队学习中记录，完整的教程和代码请参考：[LLM Universe | 动手学大模型应用开发](https://github.com/datawhalechina/llm-universe)，这里只是简单总结了一下我自己的感悟
## 第一章 大模型简介
- 开发框架：langChain框架可以给大语言模型提供通用的接口来简化开发程序
- 本地VSCode连接Codespace
- jupyter内核完成
- 第一章关于环境配置还是比较基础的，对于计算机出身来说没太大难度，但是第一章的基本概念部分，首次认识了langchain，希望之后可以上手试用
## 第二章 使用LLM API开发应用
- 通过这一章内容学会了申请百度千帆大模型的API key，并成功使用python进行调用，对于prompt engineering，我也是采用的千帆进行测试，发现千帆回答的内容确实和教程中gpt的差距很大
 ```python
	import os
	# from openai import OpenAI
	import qianfan
	from dotenv import load_dotenv, find_dotenv
	 # 如果你设置的是全局的环境变量，这行代码则没有任何作用。
	_ = load_dotenv(find_dotenv())
	
	def gen_wenxin_messages(prompt):
	    '''
	    构造文心模型请求参数 messages
	
	    请求参数：
	        prompt: 对应的用户提示词
	    '''
	    messages = [{"role": "user", "content": prompt}]
	    return messages
	# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
	def get_completion(prompt,
	                   model="Yi-34B-Chat",
	                   temperature=0.95
	                   ):
	
	    chat_comp = qianfan.ChatCompletion()
	    message = gen_wenxin_messages(prompt)
	    resp = chat_comp.do(messages=message, 
	                        model=model,
	                        temperature = temperature,
	                        system="你是一名个人助理")
	    return resp["result"]
 ```
## 第三章 搭建知识库
- 词向量和通用文本向量：通用文本向量是对长文本向量化而不再是单词
- 检索增强生成RAG系统搭建方式：公司的嵌入API、本地使用的向量模型
- 向量数据库：通过计算和目标向量之间的余弦距离等方式获取相似度，在海量的向量数据检索上更有优势，向量数据库包括有：Chroma（轻量，适合初学者）、Weaviate（开源）、Qdrant（效率更高，支持三种部署方式）
- LangChain为LLM开发自定义大模型提供了框架，支持很多大模型的Embeddings，这一节中我学习了使用Zhipuai基于Langchain自定义Embedings。下面是代码总结
```python
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
class ZhipuAIEmbeddings(Embeddings):
    def __init__(self):
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key="...")
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(
            model="embedding-3",
            input=texts
        )
        return [embeddings.embedding for embeddings in embeddings.data]
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
        

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))
texts = []
for loader in loaders: texts.extend(loader.load())
text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)
from zhipuai_embedding import ZhipuAIEmbeddings
embedding = ZhipuAIEmbeddings()
persist_directory = './chroma'

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")
#检索
question="输入问题"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
#输出检索到的位置
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
```
## 第四章 构建RAG应用

**基于LangChain调用智谱Ai**（需要自定义一个llm，这里已经封装好了zhipuai_llm.py）
- 构建检索问答链：使用搭建好的向量数据库，对查询问题进行找回，并将召回结果和问题结合起来输入到大模型进行回答
	- 1、加载向量数据库——"第三章“
		```python
		# 加载数据库
		vectordb = Chroma(
		    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
		    embedding_function=embedding
		)
		#通过as.retriever把向量数据库构造成检索器
		retriever=vector.as_retriever(search_kwargs={"k":3})#根据相似性进行检索，返回前k个最相似的文档
		docs=retriever.invoke(question)
		```
	- 2、创建检索链（使用LangChain的LCEL）
		```python
		def combine_docs(docs):
		    return "\n\n".join(doc.page_content for doc in docs)
		
		combiner = RunnableLambda(combine_docs)
		retrieval_chain = retriever | combiner
		
		retrieval_chain.invoke(question)
		```
	- 3、创建LLM
		```python
		import os 
		OPENAI_API_KEY = os.environ["ZHIPUAI_API_KEY"]
		llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=api_key)
		```
	- 4、构建检索问答链
		```python
		#将template通过 PromptTemplate 转为可以在LCEL中使用的类型
		prompt = PromptTemplate(template=template)
		qa_chain = (
		    RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
		    | prompt
		    | llm
		    | StrOutputParser()
		)
		```
	- 智谱AI的回答（教程中的问题）
	![image.png](https://fqtypora-test.oss-cn-chengdu.aliyuncs.com/20260406174218.png)
	- 5、给检索链添加聊天记录（ChatPromptTemplate把先前的对话嵌入到语言模型中，使其具有连续对话的能力）
		```python
		system_prompt = (
		    "你是一个问答任务的助手。 "
		    "请使用检索到的上下文片段回答这个问题。 "
		    "如果你不知道答案就说不知道。 "
		    "请使用简洁的话语回答用户。"
		    "\n\n"
		    "{context}"
		)
		# 制定prompt template
		qa_prompt = ChatPromptTemplate(
		    [
		        ("system", system_prompt),
		        ("placeholder", "{chat_history}"),
		        ("human", "{input}"),
		    ]
		)
		# 无历史记录
		messages = qa_prompt.invoke(
		    {
		        "input": "南瓜书是什么？",
		        "chat_history": [],
		        "context": ""
		    }
		)
		# 有历史记录
		messages = qa_prompt.invoke(
		    {
		        "input": "你可以介绍一下他吗？",
		        "chat_history": [
		            ("human", "西瓜书是什么？"),
		            ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
		        ],
		        "context": ""
		    }
		)
		```
	- 带有信息压缩的检索链（能完善用户的问题）
		```python
		# 压缩问题的系统 prompt
		condense_question_system_template = (
		    "请根据聊天记录完善用户最新的问题，"
		    "如果用户最新的问题不需要完善则返回用户的问题。"
		    )
		# 构造 压缩问题的 prompt template
		condense_question_prompt = ChatPromptTemplate([
		        ("system", condense_question_system_template),
		        ("placeholder", "{chat_history}"),
		        ("human", "{input}"),
		    ])
		# 构造检索文档的链
		# RunnableBranch 会根据条件选择要运行的分支
		retrieve_docs = RunnableBranch(
		    # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
		    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
		    # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
		    condense_question_prompt | llm | StrOutputParser() | retriever,
		)
		# 重新定义 combine_docs
		def combine_docs(docs):
		    return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]
		# 定义问答链
		qa_chain = (
		    RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数整合 qa_prompt 中的 context
		    | qa_prompt # 问答模板
		    | llm
		    | StrOutputParser() # 规定输出的格式为 str
		)
		# 定义带有历史记录的问答链
		qa_history_chain = RunnablePassthrough.assign(
		    context = (lambda x: x) | retrieve_docs # 将查询结果存为 content
		    ).assign(answer=qa_chain) # 将最终结果存为 answer
		```
	- 完成效果：
	  ![image.png](https://fqtypora-test.oss-cn-chengdu.aliyuncs.com/20260406174433.png)

	
- Streamlit:构建数据应用程序的交互界面的强大工具：最终部署的效果
  ![image.png](https://fqtypora-test.oss-cn-chengdu.aliyuncs.com/20260406174529.png)


## 第五章 系统评估和优化
- 大模型评估方法：量化评估、多维评估、构造客观题、计算答案相似度（例：利用nlkt库的bleu打分函数）、使用大模型进行评估、混合评估
### 评估优化生成部分
	- RAG检索生成增强包含两个核心部分：检索+生成
- 提升直观回答质量
```
	template_v1 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。 {context} 问题: {question} """
	template_v2 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答 案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。 {context} 问题: {question} 有用的回答:"""
	template_v3 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。 如果答案有几点，你应该分点标号回答，让答案清晰具体 {context} 问题: {question} 有用的回答:"""
```
- 标明知识来源，提高可信度
```
  template_v4 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答 案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。 如果答案有几点，你应该分点标号回答，让答案清晰具体。 请你附上回答的来源原文，以保证回答的正确性。 {context} 问题: {question} 有用的回答:"""
```
- 构造思维链
```
  template_v4 = """ 请你依次执行以下步骤： ① 使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。 你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。 如果答案有几点，你应该分点标号回答，让答案清晰具体。 上下文： {context} 问题: {question} 有用的回答: ② 基于提供的上下文，反思回答中有没有不正确或不是基于上下文得到的内容，如果有，回答你不知道 确保你执行了每一个步骤，不要跳过任意一个步骤。 """
```
- 增加一个指令解析
### 优化检索的思路
- 知识片段被割裂导致答案丢失
- query提问需要长上下文概括回答
- 关键词误导
- 匹配关系不合理

## 第六章 我的个人科研小助手
>基于Langchain+智谱embedding-3+智谱GLM设计的大模型个人助手
>源教程中使用了ChatGPT大语言模型，但是智谱送token啊，谁让它免费呢！

**关于在远程服务器上运行加载md文件报错：解决load()文件报错zipfile.BadZipFile: File is not a zip file的问题**
- 代码在本地是可以运行的，但是移到了服务器就报错了
- 解决方法：我把本地环境中的nltk_data这个文件夹复制到服务器的conda环境中的lib文件夹就好了
- 原因：可能因为是用某个镜像源导致在安装环境时这个包没有下载下来

>目的：由于写论文特别是综述时需要参考很多文献，构建RAG可以很快提取自己需要的信息，并且避免大语言模型乱编造文献的情况
>方法：我把所有的参考文献放到如图所示的文件夹，利用智谱的embedding-3向量模型和Chroma构造一个自己的私有数据库

<center><font><b>给定论文创建向量数据库的代码截图</b></font></center>

![image.png](https://fqtypora-test.oss-cn-chengdu.aliyuncs.com/20260406174626.png)

- question1：向量库生成之后发现还需要添加数据怎么办？
	```python
	#1、先加载已经生成的向量库
	vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
	#2、扫描新添加文件，按照之前的步骤进行分割
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	new_split_docs = text_splitter.split_documents(new_texts)
	#3、添加到现有的向量库中
	vectordb.add_documents(documents=new_split_docs)
	```

**相关提问以及回答**

**1、 "非接触式检测脉搏/生命体征的方法有哪些"相似性搜索**
```
向量库中存储的数量：3145
检索到的内容数：3
检索到的第0个内容: 
 contactless and non-invasive monitoring of cardiovascular signals, eliminating the need for
physical sensors or devices attached to the body.
-----------------------------------------------------
检索到的第1个内容: 
 37 529–34
Greneker E F 1997 Radar sensing of heartbeat, respiration at a distance with application of the technology 
RADAR 97 150–4
Garbey M, Sun N, Merla A and Pavlidis I 2007 Contact-free measurement of cardiac pulse based on the 
analysis of thermal imagery IEEE Trans. Biomed. Eng. 54 1418–26
Hülsbusch M and Blazek V 2002 Contactless mapping of rhythmical phenomena in tissue perfusion 
using PPGI Proc. SPIE 4683 110–7
-----------------------------------------------------
检索到的第2个内容: 
 can provide new ideas about non-contact pulse wave monitoring and cardiovascular status
analysis, which has good practical prospects in terms of comfort. The user’s pulse wave
can be captured by radar in a non-contact manner without wearing any sensors. This
monitoring method brings the significant benefit of zero interference. Users can complete
pulse wave monitoring in a non-sensory way while sleeping or working.
Radar-based technology has been increasingly used to monitor vital signs such as
-----------------------------------------------------

```
**2、question_1 = "非接触检测脉搏的方法有哪些"；question_2 = "iPPG数据集有哪些？的回答"**
```
大模型+知识库后回答 question_1 的结果：
非接触检测脉搏的方法包括使用雷达技术捕捉脉搏波，以及利用视频成像和盲源分离技术进行自动化心脏脉冲测量。这些方法无需物理传感器或设备接触身体，提供了舒适且无干扰的监测方式。谢谢你的提问！
大模型+知识库后回答 question_2 的结果：
根据提供的上下文，iPPG数据集包括PURE、BSIPL-RPPG、UBFC-rPPG、UBFC-Phys和MMPD。谢谢你的提问！
----------------大模型自己的回答-----------------
（回答内容十分多此处省略）
```
- 总结：我给了下面的提示词，大模型结合数据库后回答的内容基本源自我给的论文信息，和论文贴合度很高，但是大模型自己的回答非常宽泛
	```python
    system_prompt = (
        "你是一个问答任务的助手。 "
        "使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。 "
        "你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。"
        "如果答案有几点，你应该分点标号回答，让答案清晰具体。"
        "{context}"
        "有用的回答: 基于提供的上下文，反思回答中有没有不正确或不是基于上下文得到的内容，如果有，回答你不知道 确保你执行了每一个步骤，不要跳过任意一个步骤。"
    )
	```

**3、有无历史记录关于问题："rPPG的深度学习模型有哪些-->他们的主要框架是什么"的回答**:回答来源自我给的论文，很好地避免了大模型论文编造的错觉

```
向量库中存储的数量：3145
---------不带聊天记录------------------
文中提到的rPPG深度学习模型包括：

1. **卷积神经网络（CNN）**：用于挖掘微妙的rPPG线索，但限于有限的时空感受野。
2. **PhysFormer**：一种基于视频转换器的端到端架构，能自适应地聚合局部和全局时空特征。

此外，还提到了**迁移学习**，利用在大规模图像数据集上预训练的深度学习模型来初始化rPPG任务的模型参数。
-----------------带聊天记录--------------
根据提供的上下文，以下是几种rPPG深度学习模型的主要框架：

1. **Chen et al. (2018) 的模型**：
   - **框架**：卷积神经网络（CNN）结合注意力机制。
   - **功能**：建立视频帧与生理信息（如心率、呼吸率）之间的映射。

2. **ˇSpetlłk et al. (2018) 的模型**：
   - **框架**：两步CNN，包括特征提取器和心率估计器。
   - **功能**：从视频序列中估计心率。

3. **PhysFormer**：
   - **框架**：基于视频转换器（Transformer）的端到端架构。
   - **功能**：自适应地聚合局部和全局的时空特征，用于rPPG表示。
```

**最后把模型部署上**
![gif1.gif](https://fqtypora-test.oss-cn-chengdu.aliyuncs.com/gif1.gif)

相关代码
```python
import os
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from zhipuai_llm import ZhipuaiLLM
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda,RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import streamlit as st
_ = load_dotenv(find_dotenv()) 
api_key = os.environ['ZHIPUAI_API_KEY']
def get_retriever():
    embedding = ZhipuAIEmbeddings()
    persist_directory = '/amax/tyut/user/fq/workspace/datawhale_llm/llm-universe/data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return vectordb.as_retriever()
def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs["context"]) 
    
def get_qa_history_chain():
    # retriever = vectordb.as_retriever(search_kwargs={"k": 3})#检测数据库里最相关的三个内容
    retriever = get_retriever()
    zhipuai_model = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=api_key)
    condense_question_system_template = (
        "请根据聊天记录完善用户最新的问题，"
        "如果用户最新的问题不需要完善则返回用户的问题。"
        )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
    # 构造检索文档的链，RunnableBranch 会根据条件选择要运行的分支
    retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有  则直接使用用户问题查询向量数据库chat_history
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        # 分支 2 : 若聊天记录中有 chat_history 则先让 zhipuai_model 根据聊天记录完善问题再查询向量数据库
        condense_question_prompt | zhipuai_model | StrOutputParser() | retriever,
    )
    # 问答链的系统prompt
    system_prompt = (
        "你是一个问答任务的助手。 "
        "使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。 "
        "你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。"
        "如果答案有几点，你应该分点标号回答，让答案清晰具体。"
        "{context}"
        "有用的回答: 基于提供的上下文，反思回答中有没有不正确或不是基于上下文得到的内容，如果有，回答你不知道 确保你执行了每一个步骤，不要跳过任意一个步骤。"
    )
    # 制定prompt template
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    # 定义chain:模板->模型->解析器
    qa_chain = (RunnablePassthrough.assign(context=combine_docs) 
        | qa_prompt
        | zhipuai_model
        | StrOutputParser()
    )
    # 定义带有历史记录的问答链
    qa_history_chain = RunnablePassthrough.assign(context = (lambda x: x) | retrieve_docs 
        ).assign(answer=qa_chain)
    return qa_history_chain
    
def gen_response(chain,chat_input, chat_history):
    message=chain.stream({
        "input": chat_input,
        "chat_history": chat_history
    })
    for res in message:
        if "answer" in res.keys():
            yield res["answer"]
def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
        # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            chat_input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
if __name__ == "__main__":
    main()
     
# streamlit run "/amax/tyut/user/fq/workspace/datawhale_llm/llm-universe/core/个人助手.py"
```
