
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
import numpy as np
import os

# ▶ 폰트 설정 (없으면 fallback)
font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
else:
    plt.rc("font", family="DejaVu Sans")

plt.rcParams["axes.unicode_minus"] = False

@st.cache_data
def load_data():
    return pd.read_csv("Realfinaldata.csv")

df = load_data()
st.title("📰 설문 기반 미디어 성향 시각화 대시보드")

# -----------------------------
# 언론사별 정치 성향 평균 점수 (10_*)
# -----------------------------
st.subheader("언론사별 정치 성향 평균 점수")
media_cols = [col for col in df.columns if col.startswith("10_")]
df[media_cols] = df[media_cols].apply(pd.to_numeric, errors="coerce")
media_means = df[media_cols].mean(numeric_only=True).round(2)
labels = [col.split("_", 2)[-1] for col in media_cols]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(labels, media_means, marker='o', linewidth=2)
ax.set_ylim(1, 5)
ax.set_ylabel("성향 점수 (1=진보, 5=보수)", fontproperties=font_prop)
ax.set_title("언론사별 정치 성향 평균", fontproperties=font_prop)
for i, score in enumerate(media_means):
    ax.text(i, score + 0.1, str(score), ha='center', fontproperties=font_prop)
st.pyplot(fig)

# -----------------------------
# 선호 언론사 순위 (6_Most_Favored)
# -----------------------------
if '6_Most_Favored' in df.columns:
    st.subheader("가장 선호하는 언론사 Top 10")
    st.bar_chart(df['6_Most_Favored'].value_counts().head(10))
