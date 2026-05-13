import joblib
import pandas as pd
import shap
import os
from typing import Dict, Any, Optional
from features import build_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")


class FraudDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не знайдена: {model_path}")

        try:
            self.model_data = joblib.load(model_path)
            self.model = self.model_data["model"]
            self.features = self.model_data["features"]
            self.threshold = self.model_data.get("threshold", 0.5)
            self.user_stats = self.model_data.get("user_stats", None)
            self.explainer = shap.TreeExplainer(self.model)
            print(f"Модель завантажена: {model_path}\nКількість ознак: {len(self.features)}\nПоріг прийняття рішення: {self.threshold:.4f}")
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження моделі: {e}")

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        try:
            X, _ = build_features(pd.DataFrame([transaction]), single=True, user_stats=self.user_stats)

            for col in self.features:
                if col not in X.columns:
                    X[col] = 0
            X_features = X[self.features]

            proba_array = self.model.predict_proba(X_features)
            fraud_proba = float(proba_array[0, 1]) if proba_array.shape[1] > 1 else float(proba_array[0, 0])

            shap_values = self.explainer.shap_values(X_features)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                feature_shap = shap_values[1][0]
            elif isinstance(shap_values, list):
                feature_shap = shap_values[0][0]
            else:
                feature_shap = shap_values[0]

            feature_contributions = dict(zip(self.features, feature_shap))
            top_contributions = dict(
                sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )

        except Exception as e:
            return {
                "fraud_probability": 0.0,
                "decision": "ERROR",
                "risk_level": "UNKNOWN",
                "top_features": {},
                "explanation": f"Помилка обробки транзакції: {e}"
            }

        high_threshold = self.threshold
        review_threshold = self.threshold * 0.6

        if fraud_proba >= high_threshold:
            decision, risk_level = "BLOCK", "HIGH"
        elif fraud_proba >= review_threshold:
            decision, risk_level = "REVIEW", "MEDIUM"
        else:
            decision, risk_level = "ALLOW", "LOW"

        return {
            "fraud_probability": fraud_proba, "decision": decision, "risk_level": risk_level, "top_features": top_contributions,
            "explanation": self._generate_explanation(top_contributions, fraud_proba)
        }

    def predict_batch(self, transactions: list) -> list:
        try:
            batch_df = pd.DataFrame(transactions)
            X, _ = build_features(batch_df, single=True, user_stats=self.user_stats)

            for col in self.features:
                if col not in X.columns:
                    X[col] = 0
            X_features = X[self.features]

            proba_array = self.model.predict_proba(X_features)
            fraud_probas = proba_array[:, 1] if proba_array.shape[1] > 1 else proba_array[:, 0]

            shap_values = self.explainer.shap_values(X_features)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                all_shap = shap_values[1]
            elif isinstance(shap_values, list):
                all_shap = shap_values[0]
            else:
                all_shap = shap_values

        except Exception as e:
            return [
                {
                    "fraud_probability": 0.0,
                    "decision": "ERROR",
                    "risk_level": "UNKNOWN",
                    "top_features": {},
                    "explanation": f"Помилка пакетної обробки: {e}"
                }
                for _ in transactions
            ]

        results = []
        high_threshold = self.threshold
        review_threshold = self.threshold * 0.6

        for i, fraud_proba in enumerate(fraud_probas):
            fraud_proba = float(fraud_proba)
            feature_shap = all_shap[i] if all_shap.ndim > 1 else all_shap
            contributions = dict(zip(self.features, feature_shap))
            top_contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )

            if fraud_proba >= high_threshold:
                decision, risk_level = "BLOCK", "HIGH"
            elif fraud_proba >= review_threshold:
                decision, risk_level = "REVIEW", "MEDIUM"
            else:
                decision, risk_level = "ALLOW", "LOW"

            results.append({
                "fraud_probability": fraud_proba,
                "decision": decision,
                "risk_level": risk_level,
                "top_features": top_contributions,
                "explanation": self._generate_explanation(top_contributions, fraud_proba)
            })

        return results

    def _generate_explanation(self, contributions: Dict[str, float], fraud_proba: float) -> str:
        if fraud_proba < 0.3:
            return "Транзакція виглядає нормально. Всі показники в межах норми."

        explanations = []
        if fraud_proba >= 0.8:
            explanations.append("ВИСОКА ЙМОВІРНІСТЬ МАХІНАЦІЇ!")
        elif fraud_proba >= 0.5:
            explanations.append("Потребує додаткової перевірки.")

        for feature, contribution in list(contributions.items())[:3]:
            if abs(contribution) > 0.01:
                label = "Підозрілий" if contribution > 0 else "Нормальний"
                explanations.append(f"{label} показник: {feature}")

        return " ".join(explanations)


_detector: Optional[FraudDetector] = None


def get_detector() -> FraudDetector:
    global _detector
    if _detector is None:
        _detector = FraudDetector()
    return _detector
