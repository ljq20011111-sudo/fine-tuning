import pandas as pd
import numpy as np
import xgboost as xgb


#########################################
# 正确读取 Excel：与 AMP/MIC 相同格式
#########################################
def read_our(filename):
    """
    读取 esm_spectrum_train/test.xlsx
    格式：
    col0 = name
    col1 = y（一般不用）
    col2 ~ col_last-1 = 特征
    col_last = cate（0/1 标签）
    """
    df = pd.read_excel(filename, sheet_name="res")

    names = df.iloc[:, 0].astype(str).values       # 序列 ID
    y = df.iloc[:, -1].astype(int).values          # 最后一列 = cate = 标签
    X = df.iloc[:, 2:-1].astype(float).values      # 特征 = 中间所有列

    return X, y, names


#########################################
# Spectrum 分类模型（XGBoost）
#########################################
def spectrumRunmodel(train_file, test_file, scale_pos_weight=0.22):

    # 训练集
    X_train, y_train, _ = read_our(train_file)

    # 测试集
    X_test, _, names_test = read_our(test_file)

    print("训练集标签分布：", np.unique(y_train, return_counts=True))

    # 定义 XGBoost 分类器
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        n_jobs=-1,
        use_label_encoder=False
    )

    # 训练模型
    clf.fit(X_train, y_train)

    # 分类预测（0/1）
    preds = clf.predict(X_test)

    # 概率分数（score）
    scores = clf.predict_proba(X_test)[:, 1]

    return names_test, preds, scores


#########################################
# 主函数：输出最终 score
#########################################
if __name__ == "__main__":

    train_file = "esm_spectrum_train.xlsx"
    test_file = "esm_spectrum_test.xlsx"

    names, preds, scores = spectrumRunmodel(train_file, test_file)

    # 保存结果
    df = pd.DataFrame({
        "name": names,
        "pred": preds,
        "score": scores
    })

    df.to_excel("esm_spectrum_scores.xlsx", index=False)

    print("已完成 → 结果保存在 esm_spectrum_scores.xlsx")
