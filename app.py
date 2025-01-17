import streamlit as st
import matplotlib.pyplot as plt

# 간단한 그래프 그리기
st.title("Streamlit Matplotlib Test")

# 그래프 생성
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], label="Example Line")
ax.set_title("Example Matplotlib Plot")
ax.legend()

# Streamlit에서 그래프 렌더링
st.pyplot(fig)
