import pandas as pd
import numpy as np

TRANSACTION_TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
MERCHANT_CATEGORIES = ["Clothing", "Electronics", "Food", "Gambling", "Grocery", "Travel", "Utilities", "Other"]
HIGH_RISK_COUNTRIES = {"TR", "NG", "IN", "RU", "CN", "PK"}

FEATURE_COLS = [
    "amount", "amount_log", "hour", "is_night",
    "is_business_hours", "is_high_risk_country",
    "device_risk_score", "ip_risk_score",
    "combined_risk_score",
    *[f"tx_type_{x.lower()}" for x in TRANSACTION_TYPES],
    *[f"merchant_{x.lower()}" for x in MERCHANT_CATEGORIES]
]


def _base(df):
    df = df.copy()

    df["amount_log"] = np.log1p(df["amount"])
    df["is_night"] = df["hour"].between(22, 23) | df["hour"].between(0, 6)
    df["is_business_hours"] = df["hour"].between(9, 17)
    df["combined_risk_score"] = df["device_risk_score"] * df["ip_risk_score"]
    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES)

    for x in TRANSACTION_TYPES:
        df[f"tx_type_{x.lower()}"] = df["transaction_type"].eq(x)

    for x in MERCHANT_CATEGORIES:
        df[f"merchant_{x.lower()}"] = df["merchant_category"].eq(x)

    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def compute_user_stats(df):
    stats = df.groupby("user_id").agg(
        user_amount_mean=("amount", "mean"),
        user_amount_std=("amount", "std"),
        user_tx_count=("amount", "count"),
        user_device_risk_mean=("device_risk_score", "mean")
    ).reset_index()

    stats["user_amount_std"] = stats["user_amount_std"].fillna(0)
    return stats


def build_features(df, single=False, user_stats=None):
    df = _base(df).reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(str)

    stats = compute_user_stats(df) if user_stats is None else user_stats.copy()
    stats["user_id"] = stats["user_id"].astype(str)

    df = df.merge(stats, on="user_id", how="left").fillna({
        "user_tx_count": 1,
        "user_amount_std": 0
    })

    df["user_amount_mean"] = df["user_amount_mean"].fillna(df["amount"])
    df["user_device_risk_mean"] = df["user_device_risk_mean"].fillna(df["device_risk_score"])

    df["amount_vs_user_mean"] = (
        (df["amount"] - df["user_amount_mean"]) /
        (df["user_amount_std"] + 1e-8)
    )

    cols = FEATURE_COLS + [
        "user_tx_count",
        "user_amount_mean",
        "user_amount_std",
        "user_device_risk_mean",
        "amount_vs_user_mean"
    ]

    X = df[cols].fillna(0)
    y = df["is_fraud"] if "is_fraud" in df and not single else None

    return X, y