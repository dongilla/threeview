# streamlit 앱 전체 기능 통합 버전 (설문 기반 미디어 성향 시각화)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import matplotlib
import os

# 기본 폰트 설정 (NanumGothic이 없을 경우 시스템 기본으로 fallback)
font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
else:
    font_prop = fm.FontProperties()
    plt.rc("font", family=font_prop.get_name())

matplotlib.rcParams['axes.unicode_minus'] = False

# CSV 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("Realfinaldata.csv")
    return df

df = load_data()

st.title("📰 설문 기반 미디어 성향 시각화 대시보드")

# -----------------------------
# 1️⃣ 이슈별 뉴스 정렬 (진보-중도-복수) + 네이버 뉴스 기반 크롤링
# -----------------------------
st.subheader("1️⃣ 이슈별 뉴스 정렬 (진보-중도-복수) + 자동 크롤링")

issue_articles = {
    "후쿠시마 원전 오염수": {
        "진보": "https://www.yna.co.kr/view/AKR20250410114400073?input=1195m",
        "중도": "https://www.newsis.com/view/NISX20250409_0003131627",
        "보수": "https://news.tvchosun.com/site/data/html_dir/2025/04/10/2025041090170.html?_gl=1*inoomr*_ga*MTM2ODA4ODk0Ni4xNzQ0MjY0ODc2*_ga_D5GZR50LJV*MTc0NDM0NzExNC4zLjEuMTc0NDM0NzEyMC41NC4wLjA."
    },
    "이재명 체포동의안": {
        "진보": "https://www.hani.co.kr/arti/politics/polibar/1190906.html",
        "중도": "https://www.newsis.com/view/NISX20250407_0003128998",
        "보수": "https://news.tvchosun.com/site/data/html_dir/2025/03/28/2025032890266.html"
    },
    "윤석열 지지율": {
        "진보": "https://www.hani.co.kr/arti/politics/politics_general/1191261.html",
        "중도": "https://www.newsis.com/view/NISX20250406_0003128466",
        "보수": "http://monthly.chosun.com/client/mdaily/daily_view.asp?idx=21684&Newsnumb=20250321684"
    }
}

issue = st.selectbox("이슈를 선택하세요", list(issue_articles.keys()))

if issue:
    for category, url in issue_articles[issue].items():
        st.markdown(f"### 📰 {category} 언론")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')

            title = soup.select_one("h2.media_end_headline") or soup.select_one(".tit") or soup.select_one("h1")
            if title:
                st.write(f"**{title.text.strip()}**")

            body_tag = (
                soup.select_one("#newsct_article") or
                soup.select_one("#dic_area") or
                soup.select_one("#articleBodyContents") or
                soup.find("article") or
                soup.select_one(".newsct_article") or
                soup.select_one(".article_body") or
                soup.select_one("#articeBody") or
                soup.select_one(".article_view") or
                soup.find("div", class_="article_txt")
            )

            if body_tag:
                content = body_tag.get_text(strip=True)
                st.write(content)
            else:
                st.write("부문을 찾을 수 없습니다.")

            st.markdown(f"[원문 보기]({url})")
        except Exception as e:
            st.error(f"❌ {category} 기사 발표에 실패했습니다: {e}")
# -----------------------------
# 2️⃣ 언론사별 정치 성향 평균 점수
# -----------------------------
media_cols = [col for col in df.columns if col.startswith("10_") and col.count("_") >= 2]
df[media_cols] = df[media_cols].apply(pd.to_numeric, errors="coerce")
media_means = df[media_cols].mean().round(2)
labels = [col.split('_', 2)[-1] for col in media_cols]

st.subheader("2️⃣ 언론사별 정치 성향 평균 점수")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(labels, media_means, marker='o', linewidth=2)
ax1.set_ylim(1, 5)
ax1.set_ylabel("성향 점수 (1=진보적, 5=보수적)", fontproperties=font_prop)
ax1.set_title("언론사별 정치 성향 평균 점수", fontproperties=font_prop)
for i, score in enumerate(media_means):
    ax1.text(i, score + 0.1, str(score), ha='center', fontproperties=font_prop)
ax1.text(-0.5, 4.9, "⬆ 보수적", fontproperties=font_prop, fontsize=10, color='gray')
st.pyplot(fig1)

# -----------------------------
# 3️⃣ 정치 성향 그룹별 언론사 선호도
# -----------------------------
def map_group(x):
    if x in [1, 2]: return '진보'
    elif x == 3: return '중도'
    elif x in [4, 5]: return '보수'
    else: return None

df['정치성향그룹'] = df['4_Polictial'].map(map_group)
preference_cols = [col for col in df.columns if col.startswith("5_") and col.count("_") >= 2]
df[preference_cols] = df[preference_cols].apply(pd.to_numeric, errors='coerce')
group_pref = df.groupby("정치성향그룹")[preference_cols].mean().T.round(2)
group_pref.index = [col.split('_', 2)[-1] for col in group_pref.index]

st.subheader("3️⃣ 정치 성향 그룹별 언론사 선호도")
st.dataframe(group_pref)
fig2, ax2 = plt.subplots(figsize=(10, 5))
group_pref.plot(kind='bar', ax=ax2)
ax2.set_title("정치성향별 언론사 선호도 평균", fontproperties=font_prop)
ax2.set_ylabel("평균 점수", fontproperties=font_prop)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
st.pyplot(fig2)

# -----------------------------
# 4️⃣ 가장 선호 / 비선호 언론사
# -----------------------------
most_counts = df['6_Most_Favored'].value_counts().head(10)
least_counts = df['8_Least_Favored'].value_counts().head(10)

st.subheader("4️⃣ 가장 선호 / 비선호 언론사")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👍 선호 언론사")
    st.bar_chart(most_counts)
with col2:
    st.markdown("#### 👎 비선호 언론사")
    st.bar_chart(least_counts)

# -----------------------------
# 5️⃣ 선호 / 비선호 이유 워드클라우드
# -----------------------------
text1 = ' '.join(df['7_Reason'].dropna().astype(str))
text2 = ' '.join(df['9_Reason'].dropna().astype(str))

wc1 = WordCloud(font_path=font_path, background_color="white", width=600, height=300).generate(text1)
wc2 = WordCloud(font_path=font_path, background_color="white", width=600, height=300).generate(text2)

st.subheader("5️⃣ 선호/비선호 이유 워드클라우드")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 👍 선호 이유")
    st.image(wc1.to_array())
with col2:
    st.markdown("#### 👎 비선호 이유")
    st.image(wc2.to_array())

# -----------------------------
# 6️⃣ 뉴스 읽는 시간 분석
# -----------------------------
read_time = df['3_Readingtime'].value_counts().sort_index()
mean_time = df['3_Readingtime'].dropna().astype(str).str.extract(r'(\d+)')[0].astype(float).mean()

st.subheader("6️⃣ 뉴스 읽는 시간 분석")
st.bar_chart(read_time)
st.markdown(f"**📊 평균 읽는 시간**: 약 {round(mean_time)}분")

# -----------------------------
# 7️⃣ 언론사별 정치 성향 위치 시각화 (중도 기준선 포함)
# -----------------------------
st.subheader("7️⃣ 언론사별 정치 성향 위치 (중도 기준)")
columns_10 = [col for col in df.columns if col.startswith('10_') and col.count('_') >= 2]
mean_scores = df[columns_10].mean(numeric_only=True)
labels = [col.split('_', 2)[-1] for col in mean_scores.index]
mean_scores.index = labels

fig7, ax7 = plt.subplots(figsize=(12, 4))
ax7.axvline(x=3, color='gray', linestyle='--')
for media, score in mean_scores.items():
    ax7.plot(score, 0, 'o', markersize=10)
    if media == 'DongA':
        ax7.text(score, 0.15, media, ha='center', fontproperties=font_prop)
    elif media == 'JoongAng':
        ax7.text(score, 0.25, media, ha='center', fontproperties=font_prop)
    else:
        ax7.text(score, 0.1, media, ha='center', fontproperties=font_prop)
ax7.set_title('언론사별 정치 성향 위치 (중도 기준)', fontproperties=font_prop, fontsize=14)
ax7.set_xlabel('← 진보적        정치 성향 점수        보수적 →', fontproperties=font_prop)
ax7.set_yticks([])
ax7.set_ylim(-0.1, 0.4)
ax7.set_xlim(1, 5)
ax7.grid(axis='x', linestyle=':')
st.pyplot(fig7)

