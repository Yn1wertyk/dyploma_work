import os, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from features import build_features, compute_user_stats

RANDOM_STATE, N_FOLDS = 42, 5
BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = f"{BASE}/data/synthetic_fraud_dataset.csv"
MODELS = f"{BASE}/models"

def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    return float(th[np.argmax(2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-8))])

def plot_importance(model, names, top=20):
    i = np.argsort(model.feature_importances_)[::-1][:top]
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(i)), model.feature_importances_[i])
    plt.xticks(range(len(i)), [names[x] for x in i], rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(MODELS, exist_ok=True)
    plt.savefig(f"{MODELS}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

def train(path=DATA, target="is_fraud"):
    df = pd.read_csv(path)
    if target not in df: raise ValueError(f"Немає '{target}'")

    y = df[target].values
    fraud_rate = y.mean()
    print(f"Fraud rate: {fraud_rate:.4f}")

    params = dict(n_estimators=1000, learning_rate=.05, num_leaves=31, max_depth=6, min_child_samples=20, subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1., scale_pos_weight=int((1 - fraud_rate) / (fraud_rate + 1e-8)), random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aps, ths, models = [], [], []

    for f, (tr, vl) in enumerate(skf.split(df, y), 1):
        print(f"\nFold {f}/{N_FOLDS}")

        train_df, val_df = df.iloc[tr], df.iloc[vl]
        stats = compute_user_stats(train_df)

        Xtr, ytr = build_features(train_df)
        Xvl, yvl = build_features(val_df, user_stats=stats)

        model = LGBMClassifier(**params)
        model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], callbacks=[early_stopping(50, verbose=False), log_evaluation(200)])

        p = model.predict_proba(Xvl)[:, 1]
        ap = average_precision_score(yvl, p)
        th = best_threshold(yvl.values, p)

        aps += [ap]
        ths += [th]
        models += [model]

        print(f"PR-AUC: {ap:.4f} | Threshold: {th:.4f}")
        print(classification_report(yvl, (p >= th).astype(int),
                                    target_names=["Normal", "Fraud"]))

    i = int(np.argmax(aps))
    best = models[i]

    os.makedirs(MODELS, exist_ok=True)
    out = f"{MODELS}/fraud_model.pkl"

    joblib.dump({
        "model": best,
        "features": list(Xtr.columns),
        "threshold": ths[i],
        "user_stats": compute_user_stats(df),
        "pr_auc_scores": aps,
        "best_fold": i,
        "feature_importance": dict(zip(Xtr.columns, best.feature_importances_))
    }, out)

    print(f"\nSaved: {out}")
    print(f"PR-AUC: {np.mean(aps):.4f} ± {np.std(aps):.4f}")

    plot_importance(best, Xtr.columns)
    return best, aps

if __name__ == "__main__":
    train()