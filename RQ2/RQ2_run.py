"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2025/8/31
Last Updated: 2025/8/31
Version: 1.0.0
"""
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


# ================== 配置区 ==================
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

# ================== 系统/用户提示词 ==================
SYSTEM_PROMPT = """
Role: Requirement–Code Trace Linkage Judge
Goal: Determine whether the given requirement description and code abstract are any related.
Skill: Compare requirements and code abstract, and determine the trace linkage relationship based on their degree of match. (If relevant cases are provided, please decide whether to reference them; if they are irrelevant, please ignore them.)
Workflow: Parse the input requirement and code abstract, compare them to see if a trace linkage relationship exists, and output “yes” or “no.”
Output format: Output only a single lowercase word yes or no, without any other text, punctuation, or explanation.
Constraints:
1. Do not output anything other than yes or no.
2. Do not make subjective guesses; base the judgment solely on the given input.
"""

# ================== Embeddings 类（与建库一致） ==================
class DashScopeEmbeddings(Embeddings):
    """用 DashScope 兼容 OpenAI Embedding 接口，确保与建库一致。"""
    def __init__(
        self,
        api_key: str = DASHSCOPE_API_KEY,
        base_url: str = DASHSCOPE_BASE_URL,
        model: str = EMBED_MODEL,
        dimensions: int = EMBED_DIM,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        # 保险：空串直接返回零向量，避免 400
        if not text or not str(text).strip():
            return [0.0] * self.dimensions
        resp = self.client.embeddings.create(
            model=self.model,
            input=str(text),
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return resp.data[0].embedding

# ================== 载入多个 Chroma 库 ==================
def load_all_kbs(embedding_fn: Embeddings):
    dbs = []
    for path in KB_DIRS:
        if os.path.isdir(path):
            try:
                db = Chroma(persist_directory=path, embedding_function=embedding_fn)
                dbs.append(db)
            except Exception as e:
                print(f"⚠️ 加载知识库失败：{path} -> {e}")
        else:
            print(f"⚠️ 未找到知识库目录：{path}")
    if not dbs:
        raise RuntimeError("未能成功加载任何知识库，请检查 KB_DIRS 路径。")
    return dbs

# ================== 跨库检索（阈值过滤 + 全局TopK） ==================
def retrieve_cases(requirement: str, code: str, dbs) -> List[Tuple[str, float]]:
    """
    返回 [(case_text, score), ...]，score ∈ [0,1]，只保留 > SIM_THRESHOLD 的前 GLOBAL_TOP_K 条
    """
    query = f"{requirement}\n{code}".strip()
    all_hits: List[Tuple[str, float]] = []

    for db in dbs:
        try:
            # 取每库候选，拿到相似度分数（0~1）
            results = db.similarity_search_with_relevance_scores(query, k=PER_DB_CANDIDATES)
            for doc, score in results:
                if score is None:
                    continue
                if score > SIM_THRESHOLD:
                    # 案例文本：优先包含检索列（page_content），再拼上其他内容（metadata['content']）
                    case_body = str(doc.page_content or "").strip()
                    meta_content = str(doc.metadata.get("content", "")).strip() if doc.metadata else ""
                    if meta_content:
                        case_text = f"{case_body}\n{meta_content}"
                    else:
                        case_text = case_body
                    # 去掉极端长文本，避免对话超限
                    if len(case_text) > 60000:
                        case_text = case_text[:60000] + "…"
                        print("去除极端长文本")
                    all_hits.append((case_text, float(score)))
        except Exception as e:
            # 单库失败不影响整体
            print(f"⚠️ 知识库检索异常（已跳过）：{e}")

    # 全局合并排序，保留分数最高的前 GLOBAL_TOP_K 条
    all_hits.sort(key=lambda x: x[1], reverse=True)
    return all_hits[:GLOBAL_TOP_K]

# ================== 判定函数（把相似案例注入提示） ==================
def predict_linkage(requirement: str, code: str, dbs) -> str:
    # 检索相关案例
    hits = retrieve_cases(requirement, code, dbs)
    print(len(hits))
    if hits:
        cases_block = "\n\n---\n\n".join([h[0] for h in hits])
    else:
        cases_block = "无"

    # 用户提示词 + 上下文
    user_prompt = (
        f"The requirement artifact is: {requirement}, and the code artifact is: {code}. Please determine whether there exists a traceability link between them.\n\n"
        f"Similarity cases:\n{cases_block}"
    )
    resp = CHAT_CLIENT.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=CHAT_TEMPERATURE,
    )
    out = resp.choices[0].message.content.strip().lower()
    print(out)
    # 严格约束输出
    return "yes" if out.startswith("yes") else ("no" if out.startswith("no") else "no")

# ================== 主流程 ==================
if __name__ == '__main__':
    input_file = "../Dataset/SMOS/SMOS_code_abstract.xlsx"
    output_file = "./Result/SMOS_predict_with_instance_ca.xlsx"

    # 载入多库（只初始化一次）
    embedding_fn = DashScopeEmbeddings()
    dbs = load_all_kbs(embedding_fn)

    # 读输入
    df = pd.read_excel(input_file)

    # 断点续跑
    if os.path.exists(output_file):
        df_out = pd.read_excel(output_file)
        processed_pairs = set(zip(df_out["req_name"], df_out["code_name"]))
    else:
        df_out = pd.DataFrame(columns=["req_name", "code_name", "predict"])
        processed_pairs = set()

    # 逐行判定
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        req_name = row["req_name"]
        code_name = row["code_name"]

        if (req_name, code_name) in processed_pairs:
            continue

        requirement = str(row["req"])
        code = str(row["code"])

        try:
            pred = predict_linkage(requirement, code, dbs)  # 'yes' / 'no'
            pred_int = 1 if pred == "yes" else 0
        except Exception as e:
            print(f"Error processing ({req_name}, {code_name}): {e}")
            continue

        df_out = pd.concat(
            [df_out, pd.DataFrame([[req_name, code_name, pred_int]],
                                  columns=["req_name", "code_name", "predict"])],
            ignore_index=True
        )
        df_out.to_excel(output_file, index=False)

    print("预测任务完成，结果已保存到", output_file)
