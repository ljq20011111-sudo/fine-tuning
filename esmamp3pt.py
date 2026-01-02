# amp_3_final.py

import pandas as pd
import numpy as np
import xgboost as xgb
import os


def read_amp_format(filename):
    """
    读取 AMP 特征文件，并确保：
      - 第一列作为 name
      - 最后一列为 cate (标签)
      - 中间列是特征
    返回:
      X, y, names
    """
    df = pd.read_excel(filename)

    # 第一列是 name（数字）
    names = df.iloc[:, 0].astype(str).tolist()

    # 最后一列标签
    y = df.iloc[:, -1].astype(int).to_numpy()

    # 中间列 → 特征
    X = df.iloc[:, 1:-1].astype(float).to_numpy()

    return X, y, names



def ampRunmodel(trainfilename, testfilename,
                scale_pos_weight=2.84,
                outname="amp_scores.xlsx"):
    """
    输出格式与 MIC / SPEC / TOX 完全一致：
        name | score | pred
    """

    # 读取训练/测试
    X_train, y_train, name_train = read_amp_format(trainfilename)


    X_test,  y_test,  name_test  = read_amp_format(testfilename)

    print("训练集 labels 分布：", np.unique(y_train, return_counts=True))

    # 训练 XGBoost 模型
    clf = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 0/1 预测
    preds = clf.predict(X_test)

    # 概率得分
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_test)[:, 1]
    else:
        scores = preds.astype(float)

    # 输出文件格式（与 MIC 完全一致）
    out_df = pd.DataFrame({
        "name": name_test,
        "score": scores,
        "pred": preds
    })

    # 覆盖旧文件
    if os.path.exists(outname):
        os.remove(outname)

    out_df.to_excel(outname, index=False)
    print(f"✔ 已保存 AMP 预测结果 → {outname}")

    return name_test, scores, preds



# ================= 主函数 ================

if __name__ == "__main__":
    train_file = "./esm_amp_train.xlsx"
    test_file  = "./esm_amp_test.xlsx"

    names, scores, preds = ampRunmodel(train_file, test_file)

    print("\n前 10 条预测:")
    for i in range(10):
        print(names[i], scores[i], preds[i])
