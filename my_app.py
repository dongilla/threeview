
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

# 폰트 설정: NanumGothic 없으면 fallback
font_path = "NanumGothic.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
else:
    plt.rc("font", family="DejaVu Sans")

matplotlib.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data():
    df = pd.read_csv("Realfinaldata.csv")
    return df

df = load_data()

st.title("📰 설문 기반 미디어 성향 시각화 대시보드")

# -----------------------------
# 1️⃣ 이슈별 뉴스 정렬 + 크롤링
# -----------------------------
st.subheader("1️⃣ 이슈별 뉴스 정렬 (진보-중도-보수) + 자동 크롤링")

issue_articles = {
    "후쿠시마 원전 오염수": {
        "진보": "https://www.yna.co.kr/view/AKR20250410114400073?input=1195m",
        "중도": "https://www.newsis.com/view/NISX20250409_0003131627",
        "보수": "https://news.tvchosun.com/site/data/html_dir/2025/04/10/2025041090170.html"
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
                st.write("본문을 찾을 수 없습니다.")

            st.markdown(f"[원문 보기]({url})")
        except Exception as e:
            st.error(f"❌ {category} 기사 실패: {e}")

# -----------------------------
# 2️⃣ 언론사별 정치 성향 평균 점수
# -----------------------------
st.subheader("2️⃣ 언론사별 정치 성향 평균 점수")
media_cols = [col for col in df.columns if col.startswith("10_") and col.count("_") >= 2]
df[media_cols] = df[media_cols].apply(pd.to_numeric, errors="coerce")
media_means = df[media_cols].mean(numeric_only=True).round(2)
labels = [col.split('_', 2)[-1] for col in media_cols]

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(labels, media_means, marker='o', linewidth=2)
ax1.set_ylim(1, 5)
ax1.set_ylabel("성향 점수 (1=진보적, 5=보수적)", fontproperties=font_prop)
ax1.set_title("언론사별 정치 성향 평균 점수", fontproperties=font_prop)
for i, score in enumerate(media_means):
    ax1.text(i, score + 0.1, str(score), ha='center', fontproperties=font_prop)
ax1.text(-0.5, 4.9, "⬆ 보수적", fontproperties=font_prop, fontsize=10, color='gray')
st.pyplot(fig1)
