import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

st.title("GitHub에서 CSV 데이터 불러오기")

# GitHub RAW URL
url = "https://raw.githubusercontent.com/geonwee/wee/main/train_data.csv"

try:
    # CSV 파일 읽기
    df = pd.read_csv(url)
    st.write("데이터 미리보기:")
    st.dataframe(df)
except Exception as e:
    st.error(f"CSV 파일을 불러오는 데 실패했습니다: {e}")
# # 데이터 로드
# df = pd.read_csv(r"C:\Users\USER\Desktop\data\train_data.csv")
# st.write("데이터프레임 미리보기:")
# st.dataframe(df.head()) 

# 한글 폰트 설정 
rc('font', family='Malgun Gothic')  
plt.rcParams['axes.unicode_minus'] = False  

# 팀 이름 정리 (중복 이름 통합)
team_name_mapping = {
    'OKSavingsBank Brion': 'OKSavingsBank BRION',
}
df['teamname'] = df['teamname'].replace(team_name_mapping)

# LCK 데이터 필터링
if 'league' in df.columns:
    lck_teams = df[df['league'] == 'LCK']  # LCK 리그 데이터만 선택
    st.write("LCK 데이터프레임 미리보기:")
    st.dataframe(lck_teams.head())

#평균 경기시간
    # 'gamelength_minutes' 계산
    if 'gamelength' in lck_teams.columns:
        # 초를 분으로 변환
        lck_teams['gamelength_minutes'] = lck_teams['gamelength'] / 60

        # 연도와 팀별 평균 계산
        if 'year' in lck_teams.columns and 'teamname' in lck_teams.columns:
            grouped_data = (
                lck_teams.groupby(['year', 'teamname'])['gamelength_minutes']
                .mean()
                .reset_index()
            )

            # 연도 선택
            available_years = sorted(grouped_data['year'].unique())
            selected_year = st.selectbox("연도를 선택하세요:", available_years)

            # 선택한 연도의 데이터 필터링
            yearly_data = grouped_data[grouped_data['year'] == selected_year]
            yearly_data = yearly_data.sort_values(by='gamelength_minutes')

            # 시각화
            st.subheader(f"LCK 팀별 Game Length 평균 (Year: {selected_year})")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(yearly_data['teamname'], yearly_data['gamelength_minutes'], color='skyblue')
            ax.set_title(f"LCK 팀별 평균 Game Length (Year: {selected_year})", fontsize=16)
            ax.set_xlabel('Game Length (분)', fontsize=12)
            ax.set_ylabel('Team', fontsize=12)
            st.pyplot(fig)

        else:
            st.error("'year' 또는 'teamname' 열이 데이터에 없습니다.")
    else:
        st.error("'gamelength' 열이 데이터에 없습니다.")
else:
    st.error("'league' 열이 데이터에 없습니다.")

st.title("LCK 팀별 Gold Spent 평균 (연도별)")

# LCK 팀 데이터 필터링
if 'league' in df.columns:
    lck_teams = df[df['league'] == 'LCK']  # LCK 리그 데이터만 선택

    # 'goldspent' 열 확인
    if 'goldspent' in lck_teams.columns:
        # 'year'와 'teamname'이 데이터에 있는지 확인
        if 'year' in lck_teams.columns and 'teamname' in lck_teams.columns:
            # 'year'와 'teamname'별로 그룹화하여 'goldspent' 평균 계산
            grouped_data = (
                lck_teams.groupby(['year', 'teamname'])['goldspent']
                .mean()
                .reset_index()
            )

            # 사용 가능한 연도 선택 (key를 추가하여 고유 ID를 부여)
            available_years = sorted(grouped_data['year'].unique())
            selected_year = st.selectbox("연도를 선택하세요:", available_years, key="lck_year_selectbox")

            # 선택된 연도의 데이터 필터링
            yearly_data = grouped_data[grouped_data['year'] == selected_year]
            yearly_data = yearly_data.sort_values(by='goldspent')

            # 시각화
            st.subheader(f"LCK 팀별 Gold Spent 평균 (Year: {selected_year})")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(yearly_data['teamname'], yearly_data['goldspent'], color='skyblue')
            ax.set_title(f"LCK 팀별 Gold Spent 평균 (Year: {selected_year})", fontsize=16)
            ax.set_xlabel('Gold Spent', fontsize=12)
            ax.set_ylabel('Team', fontsize=12)
            st.pyplot(fig)
        else:
            st.error("'year' 또는 'teamname' 열이 데이터에 없습니다.")
    else:
        st.error("'goldspent' 열이 데이터에 없습니다.")
else:
    st.error("'league' 열이 데이터에 없습니다.")

# LCK 팀 데이터 필터링
if 'league' in df.columns:
    lck_teams = df[df['league'] == 'LCK']  # LCK 리그 데이터만 선택

    # 'kills', 'deaths', 'assists' 열 확인
    if {'kills', 'deaths', 'assists'}.issubset(lck_teams.columns):
        # 'year', 'teamname'별로 kills, deaths, assists 평균 계산
        if 'year' in lck_teams.columns and 'teamname' in lck_teams.columns:
            lck_grouped = (
                lck_teams.groupby(['year', 'teamname'])[['kills', 'deaths', 'assists']]
                .mean()
                .reset_index()
            )

            # KDA 계산
            lck_grouped['KDA'] = (lck_grouped['kills'] + lck_grouped['assists']) / (lck_grouped['deaths'] + 1)

            # 사용 가능한 연도 선택
            available_years = sorted(lck_grouped['year'].unique())
            selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_kda")

            # 선택된 연도의 데이터 필터링
            yearly_data = lck_grouped[lck_grouped['year'] == selected_year]
            yearly_data = yearly_data.sort_values(by='KDA')

            # 시각화
            st.subheader(f"LCK 팀별 KDA 평균 (Year: {selected_year})")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(yearly_data['teamname'], yearly_data['KDA'], color='skyblue')
            ax.set_title(f"LCK 팀별 KDA 평균 (Year: {selected_year})", fontsize=16)
            ax.set_xlabel('KDA', fontsize=12)
            ax.set_ylabel('Team', fontsize=12)
            st.pyplot(fig)
        else:
            st.error("'year' 또는 'teamname' 열이 데이터에 없습니다.")
    else:
        st.error("데이터에 'kills', 'deaths', 'assists' 열이 없습니다.")
else:
    st.error("'league' 열이 데이터에 없습니다.")
# LCK 팀 데이터 필터링
if 'league' in df.columns:
    lck_teams = df[df['league'] == 'LCK']  # LCK 리그 데이터만 선택

    # 'golddiffat15' 열 확인
    if 'golddiffat15' in lck_teams.columns:
        # 'year'와 'teamname'별로 그룹화하여 'golddiffat15' 평균 계산
        if 'year' in lck_teams.columns and 'teamname' in lck_teams.columns:
            lck_grouped = (
                lck_teams.groupby(['year', 'teamname'])['golddiffat15']
                .mean()
                .reset_index()
            )

            # 사용 가능한 연도 선택
            available_years = sorted(lck_grouped['year'].unique())
            selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_gold")

            # 선택된 연도의 데이터 필터링
            yearly_data = lck_grouped[lck_grouped['year'] == selected_year]
            yearly_data = yearly_data.sort_values(by='golddiffat15')

            # 시각화
            st.subheader(f"LCK 팀별 Gold Difference at 15 Minutes 평균 (Year: {selected_year})")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(yearly_data['teamname'], yearly_data['golddiffat15'], color='skyblue')
            ax.set_title(f"LCK 팀별 Gold Difference at 15 Minutes 평균 (Year: {selected_year})", fontsize=16)
            ax.set_xlabel('Gold Difference at 15 Minutes', fontsize=12)
            ax.set_ylabel('Team', fontsize=12)
            st.pyplot(fig)
        else:
            st.error("'year' 또는 'teamname' 열이 데이터에 없습니다.")
    else:
        st.error("'golddiffat15' 열이 데이터에 없습니다.")
else:
    st.error("'league' 열이 데이터에 없습니다.")

#LCK 팀별 First Dragon 및 Dragons 평균 (연도별)
st.title("LCK 팀별 First Dragon 및 Dragons 평균 (연도별)")

# 연도 선택
available_years = sorted(df['year'].unique())
selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_dragon")

# 해당 연도의 LCK 데이터 필터링
lck_year = df[(df['year'] == selected_year) & (df['league'] == 'LCK')]

# 팀별 승리 횟수 계산 및 드래곤 통계
if {'teamname', 'result', 'firstdragon', 'dragons'}.issubset(lck_year.columns):
    # 팀별 승리 횟수 계산
    win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

    # 팀별 첫 번째 드래곤 및 드래곤 획득 수 평균 계산
    dragon_stats = lck_year.groupby('teamname')[['firstdragon', 'dragons']].mean()

    # 승리 횟수 기준으로 데이터 정렬
    dragon_stats = dragon_stats.loc[win_counts.sort_values(ascending=False).index]

    # 시각화
    st.subheader(f"{selected_year}년 LCK 팀별 First Dragon 및 Dragons 평균 (승리 횟수 순)")
    fig, ax = plt.subplots(figsize=(12, 6))
    dragon_stats.plot(kind='bar', ax=ax, legend=True)
    ax.set_title(f"{selected_year}년 LCK 팀별 First Dragon 및 Dragons 평균 (승리 횟수 순)", fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Average Count', fontsize=12)
    ax.legend(['First Dragon', 'Dragons'], loc='best')
    plt.xticks(rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error(f"데이터에 'teamname', 'result', 'firstdragon', 'dragons' 열이 없습니다.")

#LCK 팀 골드 차이
st.title("LCK 팀별 Gold Difference (10분 및 15분) 평균 (연도별)")

# 연도 선택
available_years = sorted(df['year'].unique())
selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_gold_diff")

# 해당 연도의 LCK 데이터 필터링
lck_year = df[(df['year'] == selected_year) & (df['league'] == 'LCK')]

# 골드 차이 통계 및 시각화
if {'teamname', 'result', 'golddiffat10', 'golddiffat15'}.issubset(lck_year.columns):
    # 팀별 승리 횟수 계산
    win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

    # 팀별 골드 차이 평균 계산
    gold_diff_stats = lck_year.groupby('teamname')[['golddiffat10', 'golddiffat15']].mean()

    # 승리 횟수 기준으로 데이터 정렬
    gold_diff_stats = gold_diff_stats.loc[win_counts.sort_values(ascending=False).index]

    # 시각화
    st.subheader(f"{selected_year}년 LCK 팀별 Gold Difference (10분 및 15분) 평균 (승리 횟수 순)")
    fig, ax = plt.subplots(figsize=(12, 6))
    gold_diff_stats.plot(kind='bar', ax=ax, legend=True)
    ax.set_title(f"{selected_year}년 LCK 팀별 Gold Difference (10분 및 15분) 평균 (승리 횟수 순)", fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Average Gold Difference', fontsize=12)
    ax.legend(['Gold Difference at 10', 'Gold Difference at 15'], loc='best')
    plt.xticks(rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error("데이터에 'teamname', 'result', 'golddiffat10', 'golddiffat15' 열이 없습니다.")

#연도별 LCK 팀 XP Difference 시각화
st.title("LCK 팀별 XP Difference (10분 및 15분) 평균 (연도별)")

# 연도 선택
available_years = sorted(df['year'].unique())
selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_xp_diff")

# 해당 연도의 LCK 데이터 필터링
lck_year = df[(df['year'] == selected_year) & (df['league'] == 'LCK')]

# XP Difference 통계 및 시각화
if {'teamname', 'result', 'xpdiffat10', 'xpdiffat15'}.issubset(lck_year.columns):
    # 팀별 승리 횟수 계산
    win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

    # 팀별 경험치 차이 평균 계산
    xp_diff_stats = lck_year.groupby('teamname')[['xpdiffat10', 'xpdiffat15']].mean()

    # 승리 횟수 기준으로 데이터 정렬
    xp_diff_stats = xp_diff_stats.loc[win_counts.sort_values(ascending=False).index]

    # 시각화
    st.subheader(f"{selected_year}년 LCK 팀별 XP Difference (10분 및 15분) 평균 (승리 횟수 순)")
    fig, ax = plt.subplots(figsize=(12, 6))
    xp_diff_stats.plot(kind='bar', ax=ax, legend=True)
    ax.set_title(f"{selected_year}년 LCK 팀별 XP Difference (10분 및 15분) 평균 (승리 횟수 순)", fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Average XP Difference', fontsize=12)
    ax.legend(['XP Difference at 10', 'XP Difference at 15'], loc='best')
    plt.xticks(rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error("데이터에 'teamname', 'result', 'xpdiffat10', 'xpdiffat15' 열이 없습니다.")

#LCK 팀별 DPM 평균
selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_dpm")

# 해당 연도의 LCK 데이터 필터링
lck_year = df[(df['year'] == selected_year) & (df['league'] == 'LCK')]

# DPM 통계 및 시각화
if {'teamname', 'result', 'dpm'}.issubset(lck_year.columns):
    # 팀별 승리 횟수 계산
    win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

    # 팀별 DPM 평균 계산
    dpm_stats = lck_year.groupby('teamname')['dpm'].mean()

    # 승리 횟수 기준으로 데이터 정렬
    dpm_stats = dpm_stats.loc[win_counts.sort_values(ascending=False).index]

    # 시각화
    st.subheader(f"{selected_year}년 LCK 팀별 DPM 평균 (승리 횟수 순)")
    fig, ax = plt.subplots(figsize=(12, 6))
    dpm_stats.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(f"{selected_year}년 LCK 팀별 DPM 평균 (승리 횟수 순)", fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Average DPM', fontsize=12)
    plt.xticks(rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error("데이터에 'teamname', 'result', 'dpm' 열이 없습니다.")

#LCK 팀별 VSPM 평균 (연도별)
st.title("LCK 팀별 VSPM 평균 (연도별)")

# 연도 선택
available_years = sorted(df['year'].unique())
selected_year = st.selectbox("연도를 선택하세요:", available_years, key="year_selectbox_vspm")

# 해당 연도의 LCK 데이터 필터링
lck_year = df[(df['year'] == selected_year) & (df['league'] == 'LCK')]

# VSPM 통계 및 시각화
if {'teamname', 'result', 'vspm'}.issubset(lck_year.columns):
    # 팀별 승리 횟수 계산
    win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

    # 팀별 VSPM 평균 계산
    vspm_stats = lck_year.groupby('teamname')['vspm'].mean()

    # 승리 횟수 기준으로 데이터 정렬
    vspm_stats = vspm_stats.loc[win_counts.sort_values(ascending=False).index]

    # 시각화
    st.subheader(f"{selected_year}년 LCK 팀별 VSPM 평균 (승리 횟수 순)")
    fig, ax = plt.subplots(figsize=(12, 6))
    vspm_stats.plot(kind='bar', ax=ax, color='green')
    ax.set_title(f"{selected_year}년 LCK 팀별 VSPM 평균 (승리 횟수 순)", fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Average VSPM', fontsize=12)
    plt.xticks(rotation=90)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.error("데이터에 'teamname', 'result', 'vspm' 열이 없습니다.")

#LCK 시즌별 평균 경기 시간 (Spring vs Summer, 연도별)
st.title("LCK 시즌별 평균 경기 시간 (Spring vs Summer, 연도별)")

# 데이터 필터링
if 'league' in df.columns and 'split' in df.columns:
    lck_data = df[df['league'] == 'LCK']

    # 경기 시간 초 → 분 변환
    lck_data['gamelength_minutes'] = lck_data['gamelength'] / 60

    # 시즌(Spring/Summer)별 평균 경기 시간 계산
    gamelength_by_split = lck_data.groupby(['year', 'split'])['gamelength_minutes'].mean().unstack()

    # 시각화
    st.subheader("LCK 시즌별 평균 경기 시간")
    fig, ax = plt.subplots(figsize=(10, 6))
    gamelength_by_split.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
    ax.set_title('LCK 시즌별 평균 경기 시간 (Spring vs Summer, 연도별)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Game Length (Minutes)', fontsize=12)
    ax.legend(['Spring', 'Summer'], fontsize=10)
    plt.xticks(rotation=0)
    ax.grid(axis='y')
    st.pyplot(fig)
else:
    st.error("데이터에 'league' 또는 'split' 열이 없습니다.")

#LCK 팀별 연도별 레이더 차트
st.title("LCK 팀별 연도별 레이더 차트")

# 설정된 팀, 지표, 스케일링
teams = ['Gen.G', 'T1', 'KT Rolster', 'Dplus KIA']
metrics = ['dpm', 'vspm', 'gpr', 'goldat10', 'team kpm', 'xpat10']
scale_factors = {
    'dpm': 3000,
    'vspm': 12,
    'goldat10': 20000,
    'gpr': 1.2,
    'team kpm': 1,
    'xpat10': 25000
}

# 연도 선택
years = sorted(df['year'].unique())
selected_year = st.selectbox("연도를 선택하세요:", years, key="year_selectbox")

# 팀 선택
selected_team = st.selectbox("팀을 선택하세요:", teams, key="team_selectbox")

# 레이더 차트 생성 함수
def create_radar_chart(teamname, data, metrics, scale_factors, title):
    team_data = data[data['teamname'] == teamname]
    team_metrics = team_data[metrics].mean()

    # 각 지표를 스케일링
    for metric, scale in scale_factors.items():
        team_metrics[metric] = team_metrics[metric] / scale

    # 값 계산 및 닫기
    values = team_metrics.tolist()
    values += [values[0]]  # 시작점으로 돌아가기
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += [angles[0]]

    # 레이더 차트 생성
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=teamname)
    ax.fill(angles, values, alpha=0.25)

    # 레이블 및 눈금 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([])

    # 제목 및 범례 추가
    ax.set_title(title, size=16, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    return fig

# 연도와 팀으로 필터링
filtered_data = df[df['year'] == selected_year]
if filtered_data.empty:
    st.error(f"{selected_year}년도 데이터가 존재하지 않습니다.")
else:
    if selected_team not in filtered_data['teamname'].values:
        st.error(f"{selected_team}의 {selected_year}년도 데이터가 존재하지 않습니다.")
    else:
        # 레이더 차트 생성
        radar_chart = create_radar_chart(
            teamname=selected_team,
            data=filtered_data,
            metrics=metrics,
            scale_factors=scale_factors,
            title=f"{selected_team} 레이더 차트 - {selected_year}"
        )
        st.pyplot(radar_chart)
