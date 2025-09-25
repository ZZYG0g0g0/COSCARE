## <center>COSCARE:代码摘要与RAG案例检索结合的需求-代码跟踪链接恢复方法</center>

### 方法总览

![1755574087379](https://github.com/ZZYG0g0g0/nlpl/blob/main/fig/QQ20250819-125022.png)

现有LLMs的训练语料往往缺乏需求–代码链接恢复任务相关的数据，导致模型难以学习有效的任务表征。此外，尽管LLMs在自然语言理解方面表现优异，但在处理需求描述与程序代码这类跨语言制品时，仍面临语义鸿沟的挑战。针对上述问题，本研究提出COSCARE方法  ，通过融合代码摘要与索引增强的案例检索模块，提升需求-代码链接恢复。代码摘要将代码转换为自然语言表示，从而缩小其与需求之间的语义距离。而基于RAG的案例检索则借助外部知识库引入领域知识，增强模型决策能力。

你可以根据以下指导运行COSCARE

### 项目列表

│  readme.md                                    # 项目描述
│  requirements.txt                             # 环境配置
├─fig                                           # Readme文件中的图片
│      
├─Dataset                                 #数据集
│      SMOS                 
│      ...               
├─RQ1
│      Result          # 模型预测结果
│      RQ1_run.py               #不适用任何策略，仅使用提示学习的方法生成预测结果
│      RQ1_test.py               #根据预测结果计算性能指标
│      model_metrics.xlsx               #各模型性能指标结果
│      
└─RQ2
│      Result          # 模型预测结果
│      instance              #知识库案例
│      knowledge_base               #知识库
│      no_code_abstract_instance               #不进行代码摘要的知识库案例
│      RQ2_run.py		#COSCARE模型运行
│      RQ2_run_with_code_abstract.py		#仅使用代码摘要模块
│      RQ2_run_with_instance.py		#仅使用RAG案例检索模块
│      RQ2_test.py		#根据Result文件夹中的结果进行性能指标计算
│      code_abstract.py		#生成代码摘要
│      knowledge_base_build.py		#构建知识库
    

### 数据集

本研究中经处理后的数据集(将原始的项目转换成了csv文件)已放置于`Dataset`文件夹下，如需获得更多数据集你可以访问[here](https://drive.google.com/drive/folders/1-0MJEreOJr6F5lDQtJnCV5aNjQn_PDJX?dmr=1&ec=wgc-drive-hero-goto)

### 如何使用COSCARE

#### 环境配置

根据如下步骤可以配置COSCARE的运行环境

```shell
conda create -n hg python=3.12
conda activate hg
pip install -r requirements.txt
```

#### 运行COSCARE

首先你可以构建自己的领域知识库，即运行`RQ2/knowledge_base_build.py`，并修改`excel_file`和`save_path`参数，例如：

```python
excel_file = "./no_code_abstract_instance/Seam2.xlsx"  # 输入的Excel文件
save_path = "./knowledge_base/Seam2_no_ca"                   # 知识库存储路径
build_knowledge_base(excel_file, save_path)
```

如果使用自己的项目可以运行`RQ2/code_abstract.py`，并修改`dataset_name`,`input_folder`和`output_folder`例如：

```python
dataset_name = "Groovy" #修改成你自己的项目
input_folder = f"../Instance/{dataset_name}/cc/" #项目的源代码文件夹
output_folder = f"./Instance/{dataset_name}/" #输出路径
```

随后你可以运行`RQ2/RQ2_run.py`，选用合适的聊天模型，向量模型，检索策略和知识库，例如：

```python
# 判定用的对话模型（OpenRouter）
CHAT_CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-xxxx"  # ← 换成你的 OpenRouter Key
)
CHAT_MODEL = "meta-llama/llama-3.3-70b-instruct"
CHAT_TEMPERATURE = 0

# 检索用的 Embedding（阿里云百炼 DashScope 兼容 OpenAI Embeddings）
DASHSCOPE_API_KEY = "sk-xxxx"  # ← 换成你的 DashScope Key
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024

# 多库路径（相对当前脚本）
KB_DIRS = [
    "knowledge_base/Derby_ca",
    "knowledge_base/Dronology_ca",
    "knowledge_base/Drools_ca",
    "knowledge_base/maven_ca",
    "knowledge_base/Groovy_ca",
    "knowledge_base/Seam2_ca",
    "knowledge_base/EasyClinic_ca",
    "knowledge_base/Pig_ca"
]

# 检索策略
SIM_THRESHOLD = 0.6        # 相似度阈值（0~1）
GLOBAL_TOP_K = 3           # 全局最多取 3 条（跨库合并后）
PER_DB_CANDIDATES = 10     # 每库先取若干候选再合并
```

随后跟换主函数中的`input_file`和`output_file`以声明你的输入数据路径以及输出结果路径，例如：

```python
input_file = "../Dataset/SMOS/SMOS_code_abstract.xlsx"
output_file = "./Result/SMOS_predict_with_instance_ca.xlsx"
```

最后你可以使用如下命令运行：

```shell
python RQ2/RQ2_run.py
```







