
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
import numpy as np
import os
import matplotlib

# ✅ 폰트 처리
font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
else:
    font_prop = fm.FontProperties()
    plt.rc("font", family="DejaVu Sans")

matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ 데이터 로딩
@st.cache_data
def load_data():
    df = pd.read_csv("Realfinaldata.csv")
    return df

df = load_data()

st.title("📰 설문 기반 미디어 성향 시각화 대시보드")

# ✅ 정치 성향 그룹 숫자로 변환
df["4_Polictial"] = pd.to_numeric(df["4_Polictial"], errors="coerce")

# ✅ 1. 언론사별 정치 성향 평균 점수
st.subheader("1️⃣ 언론사별 정치 성향 평균 점수")
media_cols = [col for col in df.columns if col.startswith("10_")]
df[media_cols] = df[media_cols].apply(pd.to_numeric, errors="coerce")
media_means = df[media_cols].mean(numeric_only=True).round(2)
labels = [col.split("_", 2)[-1] for col in media_cols]

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(labels, media_means, marker='o')
ax1.set_ylim(1, 5)
ax1.set_ylabel("성향 점수 (1=진보, 5=보수)", fontproperties=font_prop)
ax1.set_title("언론사별 정치 성향 평균", fontproperties=font_prop)
for i, score in enumerate(media_means):
    ax1.text(i, score + 0.1, str(score), ha='center', fontproperties=font_prop)
st.pyplot(fig1)

# ✅ 2. 정치 성향 그룹별 언론사 선호도
st.subheader("2️⃣ 정치 성향 그룹별 언론사 선호도")

def map_group(x):
    if x in [1, 2]: return "진보"
    elif x == 3: return "중도"
    elif x in [4, 5]: return "보수"
    else: return "기타"

df["정치성향그룹"] = df["4_Polictial"].map(map_group)

pref_cols = [col for col in df.columns if col.startswith("5_")]
df[pref_cols] = df[pref_cols].apply(pd.to_numeric, errors="coerce")
group_pref = df.groupby("정치성향그룹")[pref_cols].mean().T.round(2)
group_pref.index = [col.split("_", 2)[-1] for col in group_pref.index]

st.dataframe(group_pref)

fig2, ax2 = plt.subplots(figsize=(10, 5))
group_pref.plot(kind="bar", ax=ax2)
ax2.set_title("정치성향별 언론사 선호도", fontproperties=font_prop)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
st.pyplot(fig2)

# ✅ 3. 가장 선호 / 비선호 언론사
st.subheader("3️⃣ 가장 선호 / 비선호 언론사")
most_counts = df['6_Most_Favored'].value_counts().head(10)
least_counts = df['8_Least_Favored'].value_counts().head(10)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👍 선호 언론사")
    st.bar_chart(most_counts)
with col2:
    st.markdown("#### 👎 비선호 언론사")
    st.bar_chart(least_counts)

# ✅ 4. 선호/비선호 이유 워드클라우드
st.subheader("4️⃣ 선호 / 비선호 이유 워드클라우드")
text1 = ' '.join(df['7_Reason'].dropna().astype(str))
text2 = ' '.join(df['9_Reason'].dropna().astype(str))

wc1 = WordCloud(font_path=font_path, background_color="white", width=600, height=300).generate(text1)
wc2 = WordCloud(font_path=font_path, background_color="white", width=600, height=300).generate(text2)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👍 선호 이유")
    st.image(wc1.to_array())
with col2:
    st.markdown("#### 👎 비선호 이유")
    st.image(wc2.to_array())

# ✅ 5. 뉴스 읽는 시간 분석
st.subheader("5️⃣ 뉴스 읽는 시간 분석")
read_time = df['3_Readingtime'].value_counts().sort_index()
mean_time = df['3_Readingtime'].dropna().str.extract(r'(\d+)')[0].astype(float).mean()
st.bar_chart(read_time)
st.markdown(f"**📊 평균 읽는 시간**: 약 {round(mean_time)}분")
