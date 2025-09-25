"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2025/8/12
Last Updated: 2025/8/12
Version: 1.0.0
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    # 输入文件路径
    dataset_file = "../Dataset/SMOS/SMOS.xlsx"
    # predict_file = "../RQ1/Result/SMOS_predict.xlsx"
    # predict_file = "./Result/SMOS_predict_with_instance_ca.xlsx"
    # predict_file = "./Result/SMOS_predict_with_ca.xlsx"
    predict_file = "../RQ3/Result/SMOS_predict1.xlsx"

    # 读取数据集真值
    df_truth = pd.read_excel(dataset_file)
    # 只保留必要列
    df_truth = df_truth[["req_name", "code_name", "label"]]

    # 读取预测文件的所有 sheet
    sheets = pd.read_excel(predict_file, sheet_name=None)

    # 存储结果
    results = []

    for model_name, df_pred in sheets.items():
        # 保留必要列
        df_pred = df_pred[["req_name", "code_name", "predict"]]

        # 按 (req_name, code_name) 进行合并
        df_merged = pd.merge(
            df_truth,
            df_pred,
            on=["req_name", "code_name"],
            how="inner",
            suffixes=("_true", "_pred")
        )

        # 实际上是判断 label 和 predict 是否相等，label 是真值类别
        y_true = df_merged["label"]
        y_pred = df_merged["predict"]

        # 如果是多类别，用 macro 平均
        P = precision_score(y_true, y_pred, average="binary", zero_division=0)
        R = recall_score(y_true, y_pred, average="binary", zero_division=0)
        F1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        results.append({"model": model_name, "precision": P, "recall": R, "f1": F1})

    # 输出结果表
    df_results = pd.DataFrame(results)
    print(df_results)
    # df_results.to_excel("model_metrics1.xlsx", index=False)
