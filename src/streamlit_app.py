import streamlit as st, pandas as pd, requests, numpy as np
from datetime import datetime
import plotly.express as px, plotly.graph_objects as go

st.set_page_config("Fraud Detection System", "🔍", layout="wide")
st.title("Система виявлення фінансових махінацій")
st.markdown("---")

API = "http://localhost:8000"
TYPES = ["ATM", "Online", "POS", "QR", "Transfer"]
CATS = ["Clothing", "Electronics", "Food", "Gambling", "Grocery", "Travel", "Utilities", "Other"]
COUNTRIES = ["UA", "US", "GB", "DE", "FR", "PL", "IT", "TR", "NG", "IN", "RU", "CN", "PK", "Other"]
REQ = ["user_id", "amount", "transaction_type", "merchant_category",
       "country", "hour", "device_risk_score", "ip_risk_score"]

def api(ep, data=None, method="post"):
    try:
        r = getattr(requests, method)(f"{API}{ep}", json=data, timeout=30)
        return r.json() if r.ok else st.error(f"{r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"API недоступний: {e}")

page = st.sidebar.selectbox("Навігація",
                            ["Перевірка транзакції", "Пакетна обробка", "Аналітика"])

if page == "Перевірка транзакції":
    st.header("Перевірка транзакції")

    h = api("/health", method="get")
    if h and h.get("status") == "healthy":
        st.success("API працює")
    else:
        st.error("API недоступний")

    c1, c2 = st.columns(2)
    with c1:
        data = dict(
            user_id=st.text_input("ID", "363"),
            amount=st.number_input("Сума", .01, value=100.),
            transaction_type=st.selectbox("Тип", TYPES),
            merchant_category=st.selectbox("Категорія", CATS)
        )
    with c2:
        data |= dict(
            country=st.selectbox("Країна", COUNTRIES),
            hour=st.slider("Година", 0, 23, datetime.now().hour),
            device_risk_score=st.slider("Ризик пристрою", 0., 1., .1),
            ip_risk_score=st.slider("Ризик IP", 0., 1., .1)
        )

    st.json(data)

    if st.button("Перевірити", type="primary"):
        r = api("/score", data)
        if r:
            a, b, c = st.columns(3)
            a.metric("Ймовірність", f'{r.get("fraud_probability",0):.2%}')
            b.markdown(f'**Рішення:** {r.get("decision","?")}')
            c.markdown(f'**Ризик:** {r.get("risk_level","?")}')

            st.info(r.get("explanation", "Немає пояснення"))

            if f := r.get("top_features"):
                df = pd.DataFrame(f.items(), columns=["Ознака", "SHAP"])
                st.plotly_chart(px.bar(df, x="SHAP", y="Ознака", orientation="h"),
                                use_container_width=True)

elif page == "Пакетна обробка":
    st.header("Пакетна обробка")
    st.info(f"CSV колонки: `{', '.join(REQ)}`")

    tpl = pd.DataFrame({
        "user_id": ["363"],
        "amount": [100],
        "transaction_type": ["ATM"],
        "merchant_category": ["Food"],
        "country": ["UA"],
        "hour": [12],
        "device_risk_score": [.1],
        "ip_risk_score": [.1]
    }).to_csv(index=False)

    st.download_button("Шаблон CSV", tpl, "template.csv")

    if f := st.file_uploader("CSV", ["csv"]):
        try:
            df = pd.read_csv(f)
            miss = [c for c in REQ if c not in df]
            if miss:
                st.error(f"Немає: {', '.join(miss)}")
            else:
                st.dataframe(df.head())

                if st.button("Обробити", type="primary"):
                    d = df[REQ].astype({
                        "user_id": str, "amount": float, "hour": int,
                        "device_risk_score": float, "ip_risk_score": float
                    })

                    r = api("/batch_score", d.to_dict("records"))
                    if r and (res := pd.DataFrame(r.get("results", []))).size:
                        cols = st.columns(4)
                        for i, k in enumerate(["BLOCK", "REVIEW", "ALLOW"]):
                            cols[i+1].metric(k, (res.decision == k).sum())
                        cols[0].metric("Всього", len(res))

                        st.plotly_chart(
                            px.pie(names=res.decision.value_counts().index,
                                   values=res.decision.value_counts().values),
                            use_container_width=True
                        )

                        st.dataframe(res)
                        st.download_button(
                            "Завантажити",
                            res.to_csv(index=False),
                            f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
                        )
        except Exception as e:
            st.error(e)

else:
    st.header("Аналітика")

    h = api("/health", method="get")
    if h and h.get("status") == "healthy":
        st.success("API працює")
    else:
        st.warning("API недоступний")

    x = pd.date_range("2025-01-01", periods=30)
    y = np.random.default_rng(42).uniform(.01, .05, 30)

    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(title="Динаміка махінацій",
                      xaxis_title="Дата", yaxis_title="Частота")
    st.plotly_chart(fig, use_container_width=True)