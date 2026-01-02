# esm_toxicity3pt.py
import pandas as pd
import numpy as np
import xgboost as xgb

def read_our(filename):
    """
    读取 Excel，返回 X, y, names
    期望表格格式（与其它模块保持一致）：
      col0: name (ID)
      col1: y (可忽略)
      col2 ... colN-1: features
      colN: cate (0/1 标签，1=有毒, 0=无毒)
    """
    df = pd.read_excel(filename, sheet_name="res")
    names = df.iloc[:, 0].astype(str).values
    y = df.iloc[:, -1].astype(int).values
    X = df.iloc[:, 2:-1].astype(float).values
    return X, y, names

def toxicityRunModel(train_file, test_file):
    # 读取数据
    X_train, y_train, _ = read_our(train_file)
    X_test, _, names_test = read_our(test_file)

    # 打印训练标签分布以便检查
    uniq, counts = np.unique(y_train, return_counts=True)
    print("训练集标签分布：", dict(zip(uniq.tolist(), counts.tolist())))

    # 计算 scale_pos_weight = (#neg / #pos) 以处理不平衡（防止手动设错）
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        scale_pos_weight = 1.0
        print("注意：训练集中没有正样本 (toxicity=1)。使用 scale_pos_weight=1.0")
    else:
        scale_pos_weight = neg / pos
        print(f"自动设置 scale_pos_weight = neg/pos = {neg}/{pos} = {scale_pos_weight:.3f}")

    # XGBoost 分类器（默认 objective binary:logistic）
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=0,
        n_jobs=-1,
        use_label_encoder=False
    )

    clf.fit(X_train, y_train)

    # toxic probability and compute non-toxic score
    toxic_prob = clf.predict_proba(X_test)[:, 1]   # P(toxic)
    non_toxic_score = 1.0 - toxic_prob            # P(non-toxic)

    # pred: 1 if non_toxic_score > 0.5, else 0
    preds = (non_toxic_score > 0.5).astype(int)

    return names_test, preds, non_toxic_score

if __name__ == "__main__":
    train_file = "esm_toxicity_train.xlsx"
    test_file  = "esm_toxicity_test.xlsx"

    print("Running toxicity model (output non-toxic score + pred)...")
    names, preds, scores = toxicityRunModel(train_file, test_file)

    out_df = pd.DataFrame({
        "name": names,
        "pred": preds,
        "score": scores
    })

    out_file = "esm_toxicity_scores.xlsx"
    out_df.to_excel(out_file, index=False)
    print(f"Done. Results saved to {out_file}")
    # show top 5
    print(out_df.head())
