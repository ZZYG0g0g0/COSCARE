"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2025/8/29
Last Updated: 2025/8/29
Version: 1.0.0
"""
from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import time

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-xxxx"
)

system_prompt = """
Role: Requirement–Code Trace Linkage Judge
Goal: Determine whether the given requirement description and code abstract are related.
Skill: Compare requirements and code abstract, and determine the trace linkage relationship based on their degree of match.
Workflow: Parse the input requirement and code, compare them to see if a trace linkage relationship exists, and output “yes” or “no.”
Output format: Output only a single lowercase word yes or no, without any other text, punctuation, or explanation.
Constraints:
1. Do not output anything other than yes or no.
2. Do not make subjective guesses; base the judgment solely on the given input.
"""

def predict_linkage(requirement: str, code: str) -> str:
    """
    判断需求和代码摘要是否相关，返回 'yes' 或 'no'
    """
    user_input = f"Requirement:\n{requirement}\n\nCode:\n{code}"

    response = client.chat.completions.create(
        # model="meta-llama/llama-3.3-70b-instruct:free",  # 你可以换成其它支持的模型
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_input}
        ],
        temperature=0  # 保证输出稳定
    )

    # 获取模型输出并去掉多余空格/换行
    output = response.choices[0].message.content.strip().lower()

    # 确保只返回 yes/no
    if output not in ("yes", "no"):
        raise ValueError(f"Invalid output from model: {output}")

    return output

if __name__ == '__main__':
    input_file = "../Dataset/eAnci/eAnci_code_abstract.xlsx"
    output_file = "./Result/eAnci_predict.xlsx"

    # 读取输入文件
    df = pd.read_excel(input_file)

    # 如果输出文件存在，则读取已有数据
    if os.path.exists(output_file):
        df_out = pd.read_excel(output_file)
        processed_pairs = set(zip(df_out["req_name"], df_out["code_name"]))
    else:
        df_out = pd.DataFrame(columns=["req_name", "code_name", "predict"])
        processed_pairs = set()

    # 遍历输入文件的每一行（加进度条）
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # time.sleep(1)
        req_name = row["req_name"]
        code_name = row["code_name"]

        # 跳过已经处理过的记录
        if (req_name, code_name) in processed_pairs:
            continue

        requirement = row["req"]
        code = row["code"]

        try:
            pred = predict_linkage(requirement, code)
            pred_int = 1 if pred == "yes" else 0
            print(pred)
        except Exception as e:
            print(f"Error processing ({req_name}, {code_name}): {e}")
            continue

        # 追加结果到 df_out
        df_out = pd.concat(
            [df_out, pd.DataFrame([[req_name, code_name, pred_int]], columns=["req_name", "code_name", "predict"])],
            ignore_index=True
        )

        # 实时保存（防止程序中断丢失）
        df_out.to_excel(output_file, index=False)

    print("预测任务完成，结果已保存到", output_file)