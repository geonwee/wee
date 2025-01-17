import pandas as pd
import streamlit as st

# 데이터 로드
df = pd.read_csv("teams_train.csv")  # 파일 경로 확인
st.write("데이터프레임", df.head())  # 데이터 일부 출력
