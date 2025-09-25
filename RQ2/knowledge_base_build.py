"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2025/8/30
Last Updated: 2025/8/30
Version: 1.0.1
"""
import pandas as pd
import tiktoken  # 导入 tiktoken 用于计算 token 数量
from openai import OpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from tqdm import tqdm


# ============ 自定义 DashScope Embeddings 类 ============

class DashScopeEmbeddings(Embeddings):
    def __init__(self,
                 api_key: str = "sk-xxxx",  # 替换为你的API Key
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "text-embedding-v4",
                 dimensions: int = 1024,
                 max_token_limit: int = 8192):  # 最大 token 限制，默认 8192
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self.max_token_limit = max_token_limit
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 使用 tiktoken 加载 tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")  # 适配 GPT-3.5 和 GPT-4

    def embed_documents(self, texts):
        embeddings = []
        skipped_texts = []  # 记录跳过的文本

        for text in tqdm(texts, desc="嵌入文档", unit="文档"):
            # 判断文本是否超长，若超长则跳过
            if self._is_text_too_long(text):
                skipped_texts.append(text[:100])  # 记录跳过的文本前100个字符
                print(f"⚠️ 跳过：文本超长，无法处理该文本：{text[:100]}...")
                continue  # 跳过超长文本

            # 处理正常的文本
            embedding = self._embed(text)
            embeddings.append(embedding)

        # 打印跳过的文本（可选）
        if skipped_texts:
            print(f"共跳过 {len(skipped_texts)} 个超长文本：")
            for skipped in skipped_texts:
                print(f"  - {skipped}")

        return embeddings

    def embed_query(self, text):
        if self._is_text_too_long(text):
            print(f"⚠️ 跳过：查询文本超长，无法处理该查询：{text[:100]}...")
            return None  # 返回 None 表示无法处理该查询

        # 正常处理查询文本
        return self._embed(text)

    def _embed(self, text: str):
        # 使用 OpenAI API 生成嵌入
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return resp.data[0].embedding

    def _is_text_too_long(self, text: str):
        # 计算文本的 token 数量
        token_count = len(self.encoder.encode(text))
        return token_count > self.max_token_limit

# ============ 构建知识库 ============

def build_knowledge_base(excel_file: str, save_path: str, batch_size: int = 100):
    """
    构建或更新知识库，支持从中断处继续，不会重复添加。
    """
    # 1. 初始化 Embedding 模型
    embeddings = DashScopeEmbeddings()

    # 2. 加载或创建数据库
    db = Chroma(persist_directory=save_path, embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"})

    # 3. 获取数据库中已存在的所有文档ID，这是实现断点续传的关键！
    # 使用 set 是为了让后续的查询速度更快 (O(1))
    existing_ids = set(db.get()['ids'])
    print(f"✅ 知识库加载成功。发现已存在 {len(existing_ids)} 个文档。")

    # 4. 读取Excel并筛选出需要新增的文档
    df = pd.read_excel(excel_file)
    if "req_code" not in df.columns:
        raise ValueError("Excel中必须包含 req_code 列！")

    docs_to_add = []
    ids_to_add = []  # 我们需要一个单独的列表来存放与 docs_to_add 对应的ID

    print("🔍 开始检查Excel文件，筛选需要新增的文档...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="筛选新文档", unit="行"):
        req_code = str(row["req_code"]).strip()

        # 如果 req_code 已经存在于数据库中，则跳过
        if req_code in existing_ids:
            continue

        # 如果是新文档，则处理并准备添加
        content_parts = [f"{col}: {row[col]}" for col in df.columns if col != "req_code"]
        content = "\n".join(content_parts)

        # 同样需要检查文档是否超长
        if embeddings._is_text_too_long(content):
            print(f"⚠️ 跳过超长文档: {content[:100]}...")
            continue

        docs_to_add.append(Document(page_content=req_code, metadata={"content": content}))
        ids_to_add.append(req_code)

    # 5. 如果没有新文档需要添加，则直接退出
    if not docs_to_add:
        print("🎉 所有文档均已存在于知识库中，无需新增。")
        return

    print(f"ℹ️ 发现 {len(docs_to_add)} 个新文档需要被添加到知识库。")

    # 6. 分批次将新文档及其唯一ID添加到数据库
    total_docs = len(docs_to_add)
    for i in tqdm(range(0, total_docs, batch_size), desc="嵌入并添加新文档", unit="批"):
        batch_docs = docs_to_add[i:i + batch_size]
        batch_ids = ids_to_add[i:i + batch_size]

        if batch_docs:
            try:
                # 使用 add_documents 时同时传入 documents 和 ids
                db.add_documents(documents=batch_docs, ids=batch_ids)
            except Exception as e:
                print(f"❌ 在处理批次时发生错误: {e}")
                print("程序将停止。已完成的批次已保存，下次重新运行此脚本即可从断点继续。")
                return

    print(f"✅ 成功添加 {len(docs_to_add)} 个新文档到知识库！")

###新增内容进知识库
def add_to_knowledge_base(excel_file: str, save_path: str):
    # 加载现有知识库
    embeddings = DashScopeEmbeddings()
    db = Chroma(persist_directory=save_path, embedding_function=embeddings)

    # 读取新的Excel文件
    df = pd.read_excel(excel_file)
    if "req_code" not in df.columns:
        raise ValueError("Excel中必须包含 req_code 列！")

    new_docs = []
    for _, row in df.iterrows():
        req_code = str(row["req_code"]).strip()
        # 拼接其他列
        content_parts = []
        for col in df.columns:
            if col != "req_code":
                content_parts.append(f"{col}: {row[col]}")
        content = "\n".join(content_parts)
        new_docs.append(Document(page_content=req_code, metadata={"content": content}))

    # 将新文档添加到现有知识库
    db.add_documents(new_docs)
    print(f"✅ 新内容已成功添加到知识库！")


# ============ 检索知识库 ============
def load_knowledge_base(save_path: str):
    embeddings = DashScopeEmbeddings()
    return Chroma(persist_directory=save_path, embedding_function=embeddings)

def search_knowledge_base(query: str, save_path: str, top_k: int = 3):
    db = load_knowledge_base(save_path)
    results = db.similarity_search(query, k=top_k)
    for i, res in enumerate(results, 1):
        print(f"🔎 结果 {i}:")
        print(f"req_code: {res.page_content}")
        print(f"内容: {res.metadata['content']}\n")

# ============ 示例运行 ============
if __name__ == "__main__":
    excel_file = "./no_code_abstract_instance/Seam2.xlsx"  # 输入的Excel文件
    save_path = "./knowledge_base/Seam2_no_ca"                   # 知识库存储路径

    # add_to_knowledge_base(excel_file, save_path)
    # 第一次运行：构建知识库
    build_knowledge_base(excel_file, save_path)

    # 检索示例
    # query = ""
    # search_knowledge_base(query, save_path, top_k=3)
