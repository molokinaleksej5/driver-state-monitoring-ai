from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DriverStateNN", layout="wide")
st.title("Оценка состояния водителя на основе нейронных сетей")
st.caption("Дипломный программный модуль: анализ сессий, риск, тревожные события")

log_dir = Path("artifacts/logs")
logs = sorted(log_dir.glob("*.csv")) if log_dir.exists() else []
if not logs:
    st.warning("Пока нет CSV-логов. Запустите scripts/analyze_video.py или scripts/monitor_webcam.py")
    st.stop()

selected = st.selectbox("Выберите сессию", logs, format_func=lambda p: p.name)
df = pd.read_csv(selected)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Кадров", len(df))
c2.metric("Лицо найдено, %", round(df["face_detected"].mean() * 100, 2))
c3.metric("Средний риск", round(df["risk_score"].mean(), 3))
c4.metric("Тревог", int(df["warning"].sum()))

st.subheader("Динамика риска")
st.line_chart(df.set_index("time_sec")[["risk_score"]])

st.subheader("Распределение состояний")
st.bar_chart(df["state"].value_counts())

st.subheader("Журнал кадров")
st.dataframe(df, width="stretch")
