import os
import joblib
import shap
import pandas as pd
from features import build_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")


class FraudDetector:
    def __init__(self, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        data = joblib.load(path)

        self.model = data["model"]
        self.features = data["features"]
        self.threshold = data.get("threshold", 0.5)
        self.user_stats = data.get("user_stats")
        self.explainer = shap.TreeExplainer(self.model)

    def _prepare(self, data):
        X, _ = build_features(
            pd.DataFrame(data if isinstance(data, list) else [data]),
            single=True,
            user_stats=self.user_stats
        )
        missing_cols = [col for col in self.features if col not in X.columns]
        for col in missing_cols:
            X[col] = 0
        X = X[self.features]
        proba = self.model.predict_proba(X)
        proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        return X, proba, shap_values

    def _decision(self, p):
        if p >= self.threshold:
            return "BLOCK", "HIGH"
        if p >= self.threshold * 0.6:
            return "REVIEW", "MEDIUM"
        return "ALLOW", "LOW"

    def _top(self, vals):
        return dict(sorted(
            zip(self.features, vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10])

    def _explain(self, top, p):
        if p < .3:
            return "Транзакція виглядає нормально."

        text = []

        if p >= .8:
            text.append("ВИСОКА ЙМОВІРНІСТЬ МАХІНАЦІЇ!")
        elif p >= .5:
            text.append("Потребує перевірки.")

        for k, v in list(top.items())[:3]:
            if abs(v) > .01:
                text.append(f"{'Підозрілий' if v > 0 else 'Нормальний'}: {k}")

        return " ".join(text)

    def predict_single(self, tx):
        try:
            _, p, shap_vals = self._prepare(tx)

            p = float(p[0])
            top = self._top(shap_vals[0])

            d, r = self._decision(p)

            return {
                "fraud_probability": p,
                "decision": d,
                "risk_level": r,
                "top_features": top,
                "explanation": self._explain(top, p)
            }

        except Exception as e:
            return self._error(e)

    def predict_batch(self, txs):
        try:
            _, probs, shap_vals = self._prepare(txs)

            return [{
                "fraud_probability": float(p),
                "decision": d,
                "risk_level": r,
                "top_features": top,
                "explanation": self._explain(top, p)
            } for p, s in zip(probs, shap_vals)
              for d, r in [self._decision(p)]
              for top in [self._top(s)]]

        except Exception as e:
            return [self._error(e) for _ in txs]

    @staticmethod
    def _error(e):
        return {
            "fraud_probability": 0,
            "decision": "ERROR",
            "risk_level": "UNKNOWN",
            "top_features": {},
            "explanation": str(e)
        }


_detector = None


def get_detector():
    global _detector
    _detector = _detector or FraudDetector()
    return _detector