import pandas as pd
import numpy as np
from typing import Tuple, Optional

TRANSACTION_TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
MERCHANT_CATEGORIES = ["Clothing", "Electronics", "Food", "Gambling", "Grocery", "Travel", "Utilities", "Other"]
HIGH_RISK_COUNTRIES = {"TR", "NG", "IN", "RU", "CN", "PK"}
FEATURE_COLS = ["amount", "amount_log", "hour", "is_night", "is_business_hours", "is_high_risk_country", "device_risk_score", "ip_risk_score", "combined_risk_score", *[f"tx_type_{t.lower()}" for t in TRANSACTION_TYPES], *[f"merchant_{c.lower()}" for c in MERCHANT_CATEGORIES]]


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_log"] = np.log1p(df["amount"].astype(float))
    df["hour"] = df["hour"].astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
    df["device_risk_score"] = df["device_risk_score"].astype(float)
    df["ip_risk_score"] = df["ip_risk_score"].astype(float)
    df["combined_risk_score"] = df["device_risk_score"] * df["ip_risk_score"]
    df["is_high_risk_country"] = df["country"].isin(HIGH_RISK_COUNTRIES).astype(int)

    for tx_type in TRANSACTION_TYPES:
        df[f"tx_type_{tx_type.lower()}"] = (df["transaction_type"] == tx_type).astype(int)

    for cat in MERCHANT_CATEGORIES:
        df[f"merchant_{cat.lower()}"] = (df["merchant_category"] == cat).astype(int)

    return df


def build_features(df: pd.DataFrame, single: bool = False, user_stats: Optional[pd.DataFrame] = None,) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = _base_features(df).reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(str)

    if user_stats is None:
        stats = (df.groupby("user_id").agg(user_amount_mean=("amount", "mean"), user_amount_std=("amount", "std"), user_tx_count=("amount", "count"), user_device_risk_mean=("device_risk_score", "mean")).reset_index())
        stats["user_amount_std"] = stats["user_amount_std"].fillna(0)
        stats["user_id"] = stats["user_id"].astype(str)
        df = df.merge(stats, on="user_id", how="left")
    else:
        user_stats = user_stats.copy()
        user_stats["user_id"] = user_stats["user_id"].astype(str)
        df = df.merge(user_stats, on="user_id", how="left")

    df["user_tx_count"] = df["user_tx_count"].fillna(1)
    df["user_amount_mean"] = df["user_amount_mean"].fillna(df["amount"])
    df["user_amount_std"] = df["user_amount_std"].fillna(0)
    df["user_device_risk_mean"] = df["user_device_risk_mean"].fillna(df["device_risk_score"])
    df["amount_vs_user_mean"] = (df["amount"] - df["user_amount_mean"]) / (df["user_amount_std"] + 1e-8)

    feature_cols = FEATURE_COLS + ["user_tx_count", "user_amount_mean", "user_amount_std", "user_device_risk_mean", "amount_vs_user_mean"]

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df["is_fraud"] if ("is_fraud" in df.columns and not single) else None

    return X, y


def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (df.groupby("user_id").agg(user_amount_mean=("amount", "mean"), user_amount_std=("amount", "std"), user_tx_count=("amount", "count"), user_device_risk_mean=("device_risk_score", "mean")).reset_index())
    stats["user_amount_std"] = stats["user_amount_std"].fillna(0)
    stats["user_id"] = stats["user_id"].astype(str)
    return stats
