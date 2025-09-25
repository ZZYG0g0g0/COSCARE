"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2025/8/28
Last Updated: 2025/8/28
Version: 1.0.0
"""
from openai import OpenAI
import os
from tqdm import tqdm


client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key="sk-XXXX"
)

system_prompt = """
Role: You are a code analysis tool, and you need to provide a summary of the user’s input code.
Objective: Summarize the main functionality and structure of the code.
Skills: 1. Analyze the overall purpose of the code (in 1–2 sentences).
2. Describe the functions of different parts of the code in bullet points (coarse-grained, not line-by-line).
Output format:
Overall functionality of the code: xxx.
Functions of each part of the code: xxx.
Constraint: You must output in English.
"""

def predict_linkage(code_content: str) -> str:
    """
    生成代码摘要
    """
    user_input = f"The code content is: {code_content}, and you need to analyze it."

    response = client.chat.completions.create(
        model="deepseek-chat",  # 你可以换成其它支持的模型
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_input}
        ],
        temperature=0  # 保证输出稳定
    )

    # 获取模型输出并去掉多余空格/换行
    output = response.choices[0].message.content.strip().lower()

    return output

if __name__ == '__main__':
    # 输入和输出文件夹路径
    dataset_name = "Groovy"
    # dataset_name = "EasyClinic"
    input_folder = f"../Instance/{dataset_name}/cc/"
    output_folder = f"./Instance/{dataset_name}/"

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹下所有文件
    for file_name in tqdm(os.listdir(input_folder), desc="Processing files"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 如果输出文件已存在，跳过处理
        if os.path.exists(output_path):
            print(f"⏭️ Skipping {file_name}, already processed.")
            continue

        # 读取文件内容（假设是文本/代码文件）
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                code_content = f.read()
        except Exception as e:
            print(f"❌ Error reading file {file_name}: {e}")
            continue

        # 调用模型生成摘要
        try:
            summary = predict_linkage(code_content)
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            continue

        # 保存结果到输出文件
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(summary)
        except Exception as e:
            print(f"❌ Error saving output for {file_name}: {e}")
            continue

    print(f"✅ 所有文件处理完成，结果已保存到 {output_folder}")
