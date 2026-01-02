import pandas as pd
import numpy as np
import xgboost as xgb


def read_our(filename):
    df = pd.read_excel(filename, sheet_name="res")

    # 正确读取：根据你截图结构
    # col 0 = ID
    # col 1 = y (mic)
    # col 2 ~ -2 = feature
    # col -1 = cate (0/1)
    names = df.iloc[:, 0].astype(str).values
    y = df.iloc[:, -1].astype(int).values      # 标签 = 最后一列
    X = df.iloc[:, 2:-1].astype(float).values  # 特征 = 中间的所有列

    return X, y, names


def micRunmodel(train_file, test_file, scale_pos_weight=1.0):
    X_train, y_train, name_train = read_our(train_file)
    X_test, y_test, name_test = read_our(test_file)

    print("训练集标签分布：", np.unique(y_train, return_counts=True))

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    # 得分 = predict_proba 的阳性概率
    scores = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    return name_test, scores, preds


if __name__ == "__main__":
    train_file = "esm_mic_train.xlsx"
    test_file = "esm_mic_test.xlsx"

    names, scores, preds = micRunmodel(train_file, test_file, scale_pos_weight=12.16)

    df = pd.DataFrame({"name": names, "score": scores, "pred": preds})
    df.to_excel("esm_mic_scores.xlsx", index=False)

    print("已完成 → 已生成 esm_mic_scores.xlsx")
