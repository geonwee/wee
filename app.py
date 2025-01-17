import matplotlib.pyplot as plt

# 간단한 시각화 예제
fig, ax = plt.subplots()
ax.plot(df['date'], df['gold'])  # 'date'와 'gold'는 데이터프레임 컬럼 이름
ax.set_title("Gold Over Time")

# Streamlit에서 그래프 표시
st.pyplot(fig)
