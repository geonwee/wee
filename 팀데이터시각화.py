# 로컬 데이터 경로 설정
DATA_PATH = "./data/"  # 스크립트와 같은 폴더 내 data 폴더

SEED = 42

import numpy as np
import random
import os
import pandas as pd  # pandas를 pd로 줄여 사용


# 시드 초기화 함수
def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# 데이터 읽기
train_tr = pd.read_csv(f"{DATA_PATH}/teams_train.csv")  # 학습용 데이터
train_target = pd.read_csv(f"{DATA_PATH}/teams_train_target.csv")  # 학습용 정답
test_tr = pd.read_csv(f"{DATA_PATH}/teams_test.csv")  # 테스트 데이터
test_target = pd.read_csv(f"{DATA_PATH}/teams_test_target.csv")  # 테스트 정답

print(train_tr.shape, test_tr.shape)  # 데이터 확인

drop_list_team = [
    "participantid",
    "playername",
    'position',
    "champion",
    "teamkills",
    "teamdeaths",
    "firstbloodkill",
    "firstbloodassist",
    "firstbloodvictim",
    "dragons (type unknown)",
    "damageshare",
    "earnedgoldshare",
    "total cs",
    "monsterkillsownjungle",
    "monsterkillsenemyjungle",
    "goldat20",
    "xpat20",
    "csat20",
    "opp_goldat20",
    "opp_xpat20",
    "opp_csat20",
    "golddiffat20",
    "xpdiffat20",
    "csdiffat20",
    "killsat20",
    "assistsat20",
    "deathsat20",
    "opp_killsat20",
    "opp_assistsat20",
    "opp_deathsat20",
    "goldat25",
    "xpat25",
    "csat25",
    "opp_goldat25",
    "opp_xpat25",
    "opp_csat25",
    "golddiffat25",
    "xpdiffat25",
    "csdiffat25",
    "killsat25",
    "assistsat25",
    "deathsat25",
    "opp_killsat25",
    "opp_assistsat25",
    "opp_deathsat25"
]

train_ft=train_tr
test_ft=test_tr

lpl_team = [
    "Anyone's Legend",
    "Bilibili Gaming",
    "EDward Gaming",
    "FunPlus Phoenix",
    "Invictus Gaming",
    "JD Gaming",
    "LGD Gaming",
    "LNG Esports",
    "Oh My God",
    "Rare Atom",
    "Royal Never Give Up",
    "Team WE",
    "Top Esports",
    "ThunderTalk Gaming",
    "Ultra Prime",
    "Weibo Gaming",
    "Ninjas in Pyjamas"
]

missing_columns = train_ft.columns[train_ft.isnull().sum() > 0]
null_samples = train_ft[train_ft[missing_columns].isnull().any(axis=1)]

for idx, row in null_samples.iterrows():
    team_history = train_ft[
        (train_ft["teamname"] == row["teamname"])
        & (train_ft["gameid"] < row["gameid"])
    ].sort_values("gameid", ascending=False)

    for col in missing_columns:
        if pd.isnull(row[col]):
            if row["teamname"] in lpl_team:
                # LPL 팀인 경우 LCK 평균으로 채움
                lck_mean = train_ft[train_ft["league"] == "LCK"][col].mean()
                train_ft.loc[idx, col] = lck_mean
            else:
                # LPL 팀이 아닌 경우 해당 팀의 이전 5경기 평균으로 채움
                prev_5_mean = team_history[col].head(5).mean()
                if pd.notnull(prev_5_mean):
                    train_ft.loc[idx, col] = prev_5_mean
                else:
                    # 이전 5경기 데이터가 없는 경우 해당 팀의 전체 평균으로 채움
                    team_mean = train_ft[train_ft["teamname"] == row["teamname"]][col].mean()
                    train_ft.loc[idx, col] = team_mean

train_ft.isnull().sum().sum()

train_ft.shape, test_ft.shape

missing_columns = test_ft.columns[test_ft.isnull().sum() > 0]
missing_columns

#연도
# train_ft 처리
train_ft['year'] = pd.to_datetime(train_ft['date']).dt.year

# test_ft 처리
test_ft['year'] = pd.to_datetime(test_ft['date']).dt.year

# 유효한 픽 데이터를 포함하는 행만 필터링
valid_pick_rows_train_ft = train_ft.dropna(subset=['pick1', 'pick2', 'pick3', 'pick4', 'pick5'], how='all')
valid_pick_rows_test_ft = test_ft.dropna(subset=['pick1', 'pick2', 'pick3', 'pick4', 'pick5'], how='all')

# 필터링된 train_ft 데이터에서 밴 및 픽 데이터를 하나의 열로 변환
champion_data_filtered_train_ft = valid_pick_rows_train_ft.melt(
    id_vars=['patch'],  # 패치 정보를 유지
    value_vars=['ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'pick1', 'pick2', 'pick3', 'pick4', 'pick5'],
    var_name='type',  # 데이터 유형을 나타내는 열 이름 (ban/pick)
    value_name='champion'  # 챔피언 이름을 저장할 열 이름
)

# 변환된 train_ft 데이터에서 NaN(결측값) 제거
champion_data_filtered_train_ft = champion_data_filtered_train_ft.dropna(subset=['champion'])

# train_ft 데이터에서 패치와 챔피언별로 그룹화하여 선택 횟수 계산
champion_counts_filtered_train_ft = champion_data_filtered_train_ft.groupby(['patch', 'champion']).size().reset_index(name='count')

# train_ft 데이터에서 패치별로 챔피언 선택 횟수 기준으로 정렬
top_champions_by_patch_filtered_train_ft = champion_counts_filtered_train_ft.sort_values(['patch', 'count'], ascending=[True, False])

# Train 데이터 병합
# 챔피언별 선택 횟수를 train_ft에 병합
train_ft = train_ft.merge(
    top_champions_by_patch_filtered_train_ft,
    how='left',  # 패치와 챔피언에 대해 일치하는 경우 병합
    left_on=['patch', 'pick1'],  # train_ft에서 병합 기준 열
    right_on=['patch', 'champion']  # top_champions_by_patch_filtered_train_ft에서 병합 기준 열
)

# 'count' 열 이름 변경 (선택 횟수를 의미하는 더 명확한 이름으로)
train_ft.rename(columns={'count': 'pick1_count'}, inplace=True)

# Test 데이터 병합
valid_pick_rows_test_ft = test_ft.dropna(subset=['pick1', 'pick2', 'pick3', 'pick4', 'pick5'], how='all')

champion_data_filtered_test_ft = valid_pick_rows_test_ft.melt(
    id_vars=['patch'],  # 패치 정보를 유지
    value_vars=['ban1', 'ban2', 'ban3', 'ban4', 'ban5', 'pick1', 'pick2', 'pick3', 'pick4', 'pick5'],
    var_name='type',
    value_name='champion'
)

champion_data_filtered_test_ft = champion_data_filtered_test_ft.dropna(subset=['champion'])
champion_counts_filtered_test_ft = champion_data_filtered_test_ft.groupby(['patch', 'champion']).size().reset_index(name='count')
top_champions_by_patch_filtered_test_ft = champion_counts_filtered_test_ft.sort_values(['patch', 'count'], ascending=[True, False])

# Test 데이터 병합
test_ft = test_ft.merge(
    top_champions_by_patch_filtered_test_ft,
    how='left',
    left_on=['patch', 'pick1'],
    right_on=['patch', 'champion']
)

# 'count' 열 이름 변경 (선택 횟수를 의미하는 더 명확한 이름으로)
test_ft.rename(columns={'count': 'pick1_count'}, inplace=True)

train_ft.shape, test_ft.shape

import os
import pandas as pd

# 1. 폴더 생성
data_dir = './data'  # 'data' 폴더 경로
os.makedirs(data_dir, exist_ok=True)  # 폴더가 없으면 생성

# 2. CSV 파일로 저장
train_csv_path = os.path.join(data_dir, 'train_ft_1차_.csv')  # 'data' 폴더에 train 파일 저장 경로
test_csv_path = os.path.join(data_dir, 'test_ft_1차_.csv')   # 'data' 폴더에 test 파일 저장 경로

train_ft.to_csv(train_csv_path, index=False, encoding='utf-8-sig')  # UTF-8로 저장
test_ft.to_csv(test_csv_path, index=False, encoding='utf-8-sig')    # UTF-8로 저장

print(f"Train 파일 저장 경로: {train_csv_path}")
print(f"Test 파일 저장 경로: {test_csv_path}")

# 3. 저장된 데이터 파일 불러오기
train_data_part1 = pd.read_csv(train_csv_path)
test_data_part2 = pd.read_csv(test_csv_path)

# 4. 두 데이터를 합쳐 하나의 train_data로 사용
train_data = pd.concat([train_data_part1, test_data_part2], ignore_index=True)

# 5. 데이터 확인
print(f"합친 데이터 크기: {train_data.shape}")
print(train_data.head())



# 결과 확인
train_data_shape = train_data.shape
train_data_preview = train_data.head()

train_data_shape, train_data_preview

# 팀 이름 정리 (중복 이름 통합)
team_name_mapping = {
    'OKSavingsBank Brion': 'OKSavingsBank BRION',
}



# 팀 이름 통합
train_data['teamname'] = train_data['teamname'].replace(team_name_mapping)

# LCK 팀 데이터 필터링
lck_teams = train_data[train_data['league'] == 'LCK']

# 숫자 데이터만 선택하여 그룹화
numeric_columns = lck_teams.select_dtypes(include=['number'])
lck_grouped = numeric_columns.groupby(lck_teams['teamname']).mean()

import matplotlib.pyplot as plt

# 특정 지표를 선택해 시각화(경기시간)
if 'gamelength' in lck_grouped.columns:
    lck_grouped['gamelength'].sort_values().plot(kind='barh', figsize=(10, 8))
    plt.title('LCK 팀별 Game Length 평균')
    plt.xlabel('Game Length (분)')
    plt.ylabel('Team')
    plt.show()
else:
    print("데이터에 'gamelength' 열이 없습니다. 다른 지표를 사용해주세요.")

# 'year'와 'teamname'별로 그룹화하여 'gamelength' 평균 계산
if 'gamelength' in lck_teams.columns:
    # 초를 분으로 변환
    lck_teams['gamelength_minutes'] = lck_teams['gamelength'] / 60

    lck_grouped = (
        lck_teams.groupby(['year', 'teamname'])['gamelength_minutes']
        .mean()
        .reset_index()
    )

    # 연도별로 시각화
    for year in sorted(lck_grouped['year'].unique()):
        yearly_data = lck_grouped[lck_grouped['year'] == year]
        yearly_data = yearly_data.sort_values(by='gamelength_minutes')
        yearly_data.plot(
            x='teamname', y='gamelength_minutes', kind='barh', figsize=(10, 8), legend=False
        )
        plt.title(f'LCK 팀별 Game Length 평균 (Year: {year})')
        plt.xlabel('Game Length (분)')
        plt.ylabel('Team')
        plt.show()
else:
    print("데이터에 'gamelength' 열이 없습니다.")

"""골드"""

# 'year'와 'teamname'별로 그룹화하여 'goldspent' 평균 계산
if 'goldspent' in lck_teams.columns:
    lck_grouped = (
        lck_teams.groupby(['year', 'teamname'])['goldspent']
        .mean()
        .reset_index()
    )

    # 연도별로 시각화
    for year in sorted(lck_grouped['year'].unique()):
        yearly_data = lck_grouped[lck_grouped['year'] == year]
        yearly_data = yearly_data.sort_values(by='goldspent')
        yearly_data.plot(
            x='teamname', y='goldspent', kind='barh', figsize=(10, 8), legend=False
        )
        plt.title(f'LCK 팀별 Gold Spent 평균 (Year: {year})')
        plt.xlabel('Gold Spent')
        plt.ylabel('Team')
        plt.show()
else:
    print("데이터에 'goldspent' 열이 없습니다.")

"""KDA"""

# 'year', 'teamname'별로 kills, deaths, assists 평균 계산
if {'kills', 'deaths', 'assists'}.issubset(lck_teams.columns):
    lck_grouped = (
        lck_teams.groupby(['year', 'teamname'])[['kills', 'deaths', 'assists']]
        .mean()
        .reset_index()
    )

    # KDA 계산
    lck_grouped['KDA'] = (lck_grouped['kills'] + lck_grouped['assists']) / (lck_grouped['deaths'] + 1)

    # 연도별로 시각화
    for year in sorted(lck_grouped['year'].unique()):
        yearly_data = lck_grouped[lck_grouped['year'] == year]
        yearly_data = yearly_data.sort_values(by='KDA')
        yearly_data.plot(
            x='teamname', y='KDA', kind='barh', figsize=(10, 8), legend=False
        )
        plt.title(f'LCK 팀별 KDA 평균 (Year: {year})')
        plt.xlabel('KDA')
        plt.ylabel('Team')
        plt.show()
else:
    print("데이터에 'kills', 'deaths', 'assists' 열이 없습니다.")

"""시야 점수"""

# 'year'와 'teamname'별로 그룹화하여 'visionscore' 평균 계산
if 'visionscore' in lck_teams.columns:
    lck_grouped = (
        lck_teams.groupby(['year', 'teamname'])['visionscore']
        .mean()
        .reset_index()
    )

    # 연도별로 시각화
    for year in sorted(lck_grouped['year'].unique()):
        yearly_data = lck_grouped[lck_grouped['year'] == year]
        yearly_data = yearly_data.sort_values(by='visionscore')
        yearly_data.plot(
            x='teamname', y='visionscore', kind='barh', figsize=(10, 8), legend=False
        )
        plt.title(f'LCK 팀별 Vision Score 평균 (Year: {year})')
        plt.xlabel('Vision Score')
        plt.ylabel('Team')
        plt.show()
else:
    print("데이터에 'visionscore' 열이 없습니다.")

"""15분 골드 차이"""

# 'year'와 'teamname'별로 그룹화하여 'golddiffat15' 평균 계산
if 'golddiffat15' in lck_teams.columns:
    lck_grouped = (
        lck_teams.groupby(['year', 'teamname'])['golddiffat15']
        .mean()
        .reset_index()
    )

    # 연도별로 시각화
    for year in sorted(lck_grouped['year'].unique()):
        yearly_data = lck_grouped[lck_grouped['year'] == year]
        yearly_data = yearly_data.sort_values(by='golddiffat15')
        yearly_data.plot(
            x='teamname', y='golddiffat15', kind='barh', figsize=(10, 8), legend=False
        )
        plt.title(f'LCK 팀별 Gold Difference at 15 Minutes 평균 (Year: {year})')
        plt.xlabel('Gold Difference at 15 Minutes')
        plt.ylabel('Team')
        plt.show()
else:
    print("데이터에 'golddiffat15' 열이 없습니다.")

"""

#연도별"""

for year in [2022, 2023, 2024]:
    # 해당 연도의 LCK 팀 데이터 필터링
    lck_year = train_data[(train_data['year'] == year) & (train_data['league'] == 'LCK')]

    # 팀별 승리 횟수 계산 및 정렬
    if {'teamname', 'result', 'firstdragon', 'dragons'}.issubset(lck_year.columns):
        win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

        # 팀별 첫 번째 드래곤 및 드래곤 획득 수 평균 계산
        dragon_stats = lck_year.groupby('teamname')[['firstdragon', 'dragons']].mean()

        # 승리 횟수 기준으로 데이터 정렬
        dragon_stats = dragon_stats.loc[win_counts.sort_values(ascending=False).index]

        # 시각화
        dragon_stats.plot(kind='bar', figsize=(12, 6))
        plt.title(f'{year}년 LCK 팀별 First Dragon 및 Dragons 평균 (승리 횟수 순)')
        plt.xlabel('Team')
        plt.ylabel('Average Count')
        plt.legend(['First Dragon', 'Dragons'], loc='best')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()
    else:
        print(f"{year}년 데이터에 'teamname', 'result', 'firstdragon', 'dragons' 열이 없습니다.")

for year in [2022, 2023, 2024]:
    # 해당 연도의 LCK 팀 데이터 필터링
    lck_year = train_data[(train_data['year'] == year) & (train_data['league'] == 'LCK')]

    # 팀별 승리 횟수 계산 및 정렬
    if {'teamname', 'result', 'golddiffat10', 'golddiffat15'}.issubset(lck_year.columns):
        win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

        # 팀별 골드 차이 평균 계산
        gold_diff_stats = lck_year.groupby('teamname')[['golddiffat10', 'golddiffat15']].mean()

        # 승리 횟수 기준으로 데이터 정렬
        gold_diff_stats = gold_diff_stats.loc[win_counts.sort_values(ascending=False).index]

        # 시각화
        gold_diff_stats.plot(kind='bar', figsize=(12, 6))
        plt.title(f'{year}년 LCK 팀별 Gold Difference (10분 및 15분) 평균 (승리 횟수 순)')
        plt.xlabel('Team')
        plt.ylabel('Average Gold Difference')
        plt.legend(['Gold Difference at 10', 'Gold Difference at 15'], loc='best')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()
    else:
        print(f"{year}년 데이터에 'teamname', 'result', 'golddiffat10', 'golddiffat15' 열이 없습니다.")

for year in [2022, 2023, 2024]:
    # 해당 연도의 LCK 팀 데이터 필터링
    lck_year = train_data[(train_data['year'] == year) & (train_data['league'] == 'LCK')]

    # 팀별 승리 횟수 계산 및 정렬
    if {'teamname', 'result', 'xpdiffat10', 'xpdiffat15'}.issubset(lck_year.columns):
        win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

        # 팀별 경험치 차이 평균 계산
        xp_diff_stats = lck_year.groupby('teamname')[['xpdiffat10', 'xpdiffat15']].mean()

        # 승리 횟수 기준으로 데이터 정렬
        xp_diff_stats = xp_diff_stats.loc[win_counts.sort_values(ascending=False).index]

        # 시각화
        xp_diff_stats.plot(kind='bar', figsize=(12, 6))
        plt.title(f'{year}년 LCK 팀별 XP Difference (10분 및 15분) 평균 (승리 횟수 순)')
        plt.xlabel('Team')
        plt.ylabel('Average XP Difference')
        plt.legend(['XP Difference at 10', 'XP Difference at 15'], loc='best')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()
    else:
        print(f"{year}년 데이터에 'teamname', 'result', 'xpdiffat10', 'xpdiffat15' 열이 없습니다.")

for year in [2022, 2023, 2024]:
    # 해당 연도의 LCK 팀 데이터 필터링
    lck_year = train_data[(train_data['year'] == year) & (train_data['league'] == 'LCK')]

    # 팀별 승리 횟수 계산 및 정렬
    if {'teamname', 'result', 'dpm'}.issubset(lck_year.columns):
        win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

        # 팀별 DPM 평균 계산
        dpm_stats = lck_year.groupby('teamname')['dpm'].mean()

        # 승리 횟수 기준으로 데이터 정렬
        dpm_stats = dpm_stats.loc[win_counts.sort_values(ascending=False).index]

        # 시각화
        dpm_stats.plot(kind='bar', figsize=(12, 6), color='skyblue')
        plt.title(f'{year}년 LCK 팀별 DPM 평균 (승리 횟수 순)')
        plt.xlabel('Team')
        plt.ylabel('Average DPM')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()
    else:
        print(f"{year}년 데이터에 'teamname', 'result', 'dpm' 열이 없습니다.")

for year in [2022, 2023, 2024]:
    # 해당 연도의 LCK 팀 데이터 필터링
    lck_year = train_data[(train_data['year'] == year) & (train_data['league'] == 'LCK')]

    # 팀별 승리 횟수 계산 및 정렬
    if {'teamname', 'result', 'vspm'}.issubset(lck_year.columns):
        win_counts = lck_year[lck_year['result'] == 1].groupby('teamname')['result'].count()

        # 팀별 VSPM 평균 계산
        vspm_stats = lck_year.groupby('teamname')['vspm'].mean()

        # 승리 횟수 기준으로 데이터 정렬
        vspm_stats = vspm_stats.loc[win_counts.sort_values(ascending=False).index]

        # 시각화
        vspm_stats.plot(kind='bar', figsize=(12, 6), color='green')
        plt.title(f'{year}년 LCK 팀별 VSPM 평균 (승리 횟수 순)')
        plt.xlabel('Team')
        plt.ylabel('Average VSPM')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.show()
    else:
        print(f"{year}년 데이터에 'teamname', 'result', 'vspm' 열이 없습니다.")

# # 주요 변수와 승리 여부 간의 상관관계 분석
# important_columns = ['result', 'golddiffat15', 'csdiffat10', 'xpdiffat15', 'visionscore', 'kills', 'deaths', 'assists']
# if all(col in lck_2022.columns for col in important_columns):
#     correlation_matrix = lck_2022[important_columns].corr()

#     # 히트맵으로 시각화
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('승리 여부와 주요 변수 간 상관관계 (2022 LCK)')
#     plt.show()
# else:
#     print("데이터에 필요한 열이 없습니다.")

# # 주요 변수와 승리 여부 간의 상관관계 분석 (GPM 추가)
# important_columns = ['result', 'golddiffat15', 'csdiffat10', 'xpdiffat15', 'visionscore', 'kills', 'deaths', 'assists', 'earned gpm']

# if all(col in lck_2022.columns for col in important_columns):
#     correlation_matrix = lck_2022[important_columns].corr()

#     # 히트맵으로 시각화
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('승리 여부와 주요 변수 간 상관관계 (2022 LCK, GPM 포함)')
#     plt.show()
# else:
#     print("데이터에 필요한 열이 없습니다.")

# # 승리 팀과 패배 팀의 주요 변수 평균 비교
# if all(col in lck_2022.columns for col in important_columns):
#     win_stats = lck_2022[lck_2022['result'] == 1][important_columns].mean()
#     lose_stats = lck_2022[lck_2022['result'] == 0][important_columns].mean()

#     comparison = pd.DataFrame({'Win': win_stats, 'Lose': lose_stats})
#     comparison.plot(kind='bar', figsize=(12, 6))
#     plt.title('승리 팀 vs 패배 팀 주요 지표 비교 (2022 LCK)')
#     plt.xlabel('Metrics')
#     plt.ylabel('Average Value')
#     plt.xticks(rotation=45)
#     plt.legend(['Win', 'Lose'])
#     plt.grid(True)
#     plt.show()
# else:
#     print("데이터에 필요한 열이 없습니다.")

# # 골드 차이(golddiffat15)의 승리/패배 분포 확인
# if 'golddiffat15' in lck_2022.columns:
#     import seaborn as sns
#     sns.histplot(data=lck_2022, x='golddiffat15', hue='result', kde=True, bins=30, palette='pastel')
#     plt.title('골드 차이 (15분 기준) 분포 (승리 vs 패배)')
#     plt.xlabel('Gold Difference at 15 Minutes')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()
# else:
#     print("데이터에 'golddiffat15' 열이 없습니다.")


if 'league' in train_data.columns and 'split' in train_data.columns:
    lck_data = train_data[train_data['league'] == 'LCK']

    # 경기 시간 초 → 분 변환
    lck_data['gamelength_minutes'] = lck_data['gamelength'] / 60

    # 시즌(Spring/Summer)별 평균 경기 시간 계산
    gamelength_by_split = lck_data.groupby(['year', 'split'])['gamelength_minutes'].mean().unstack()

    # 시각화
    gamelength_by_split.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange'])
    plt.title('LCK 시즌별 평균 경기 시간 (Spring vs Summer, 연도별)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Average Game Length (Minutes)')
    plt.legend(['Spring', 'Summer'], fontsize=10)
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("데이터에 'league' 또는 'split' 열이 없습니다.")

# 2024년 이전 데이터 필터링
lck_pre_2024 = lck_data[lck_data['year'] < 2024]

# 드래곤과 전령 확보 여부에 따른 승률 분석
if {'firstdragon', 'firstherald', 'result'}.issubset(lck_pre_2024.columns):
    object_win_rates = (
        lck_pre_2024.groupby('firstdragon')['result'].mean()
        .rename('First Dragon Win Rate')
        .to_frame()
    )
    object_win_rates['Herald Win Rate'] = lck_pre_2024.groupby('firstherald')['result'].mean()

    # 시각화
    object_win_rates.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'orange'])
    plt.title('드래곤 및 전령 확보에 따른 승률 (2024년 이전 LCK)', fontsize=16)
    plt.xlabel('Object Secured', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.legend(['First Dragon', 'First Herald'], fontsize=10)
    plt.xticks([0, 1], labels=['No', 'Yes'], rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("데이터에 'firstdragon', 'firstherald', 'result' 열이 필요합니다.")

# 2024년 데이터 필터링
lck_2024 = lck_data[lck_data['year'] == 2024]

# 드래곤과 전령 확보 여부에 따른 승률 분석
if {'firstdragon', 'firstherald', 'result'}.issubset(lck_2024.columns):
    object_win_rates = (
        lck_2024.groupby('firstdragon')['result'].mean()
        .rename('First Dragon Win Rate')
        .to_frame()
    )
    object_win_rates['Herald Win Rate'] = lck_2024.groupby('firstherald')['result'].mean()

    # 시각화
    object_win_rates.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'orange'])
    plt.title('드래곤 및 전령 확보에 따른 승률 (2024년 LCK)', fontsize=16)
    plt.xlabel('Object Secured', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.legend(['First Dragon', 'First Herald'], fontsize=10)
    plt.xticks([0, 1], labels=['No', 'Yes'], rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("데이터에 'firstdragon', 'firstherald', 'result' 열이 필요합니다.")

# # 오브젝트 획득 순서와 승률 분석
# if 'firstdragon' in lck_data.columns and 'firstherald' in lck_data.columns and 'result' in lck_data.columns:
#     lck_data['first_object'] = lck_data.apply(
#         lambda row: 'Dragon' if row['firstdragon'] == 1 else ('Herald' if row['firstherald'] == 1 else 'None'),
#         axis=1
#     )
#     first_object_win_rate = lck_data.groupby('first_object')['result'].mean()

#     # 시각화
#     first_object_win_rate.plot(kind='bar', color='limegreen', figsize=(8, 6))
#     plt.title('첫 번째 오브젝트 확보 유형에 따른 승률 (LCK)', fontsize=16)
#     plt.xlabel('First Object Secured', fontsize=12)
#     plt.ylabel('Win Rate', fontsize=12)
#     plt.xticks(rotation=0)
#     plt.grid(axis='y')
#     plt.show()
# else:
#     print("데이터에 'firstdragon', 'firstherald', 'result' 열이 필요합니다.")

# 오브젝트 획득 순서와 승률 분석 (14 시즌 이전 데이터로)
if {'firstdragon', 'firstherald', 'result', 'patch'}.issubset(lck_data.columns):
    # 14 시즌 이전 데이터 필터링
    pre_patch_14_data = lck_data[lck_data['patch'] < 14]

    # 첫 번째 오브젝트 유형 정의
    pre_patch_14_data['first_object'] = pre_patch_14_data.apply(
        lambda row: 'Dragon' if row['firstdragon'] == 1 else ('Herald' if row['firstherald'] == 1 else 'None'),
        axis=1
    )

    # 첫 번째 오브젝트 확보 유형에 따른 승률 계산
    first_object_win_rate = pre_patch_14_data.groupby('first_object')['result'].mean()

    # 시각화
    first_object_win_rate.plot(kind='bar', color='limegreen', figsize=(8, 6))
    plt.title('14 시즌 이전 첫 번째 오브젝트 확보 유형에 따른 승률 (LCK)', fontsize=16)
    plt.xlabel('First Object Secured', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("데이터에 'firstdragon', 'firstherald', 'result', 'patch' 열이 필요합니다.")

# # LCK 데이터 필터링 및 14시즌 이후 void_grubs와 dragons 비교
# if 'patch' in train_data.columns and 'void_grubs' in train_data.columns and 'dragons' in train_data.columns:
#     # 14 패치 버전 이후 데이터 필터링
#     patch_14_data = train_data[(train_data['patch'] >= 14) & (train_data['league'] == 'LCK')]

#     # void_grubs와 dragons의 승률 비교
#     if 'result' in patch_14_data.columns:
#         void_grubs_win_rate = patch_14_data.groupby('void_grubs')['result'].mean()
#         dragons_win_rate = patch_14_data.groupby('dragons')['result'].mean()

#         # 데이터 정리
#         win_rates = pd.DataFrame({
#             'Void Grubs Win Rate': void_grubs_win_rate,
#             'Dragons Win Rate': dragons_win_rate
#         }).fillna(0)

#         # 시각화
#         win_rates.plot(kind='bar', figsize=(8, 6), color=['purple', 'orange'])
#         plt.title('14 패치 이후 Void Grubs와 Dragons의 승리 기여 비교', fontsize=16)
#         plt.xlabel('Object Secured', fontsize=12)
#         plt.ylabel('Win Rate', fontsize=12)
#         plt.xticks([0, 1], labels=['No', 'Yes'], rotation=0)
#         plt.legend(['Void Grubs', 'Dragons'], fontsize=10)
#         plt.grid(axis='y')
#         plt.show()
#     else:
#         print("데이터에 'result' 열이 없습니다. 승률 분석을 위해 필요합니다.")
# else:
#     print("데이터에 'patch', 'void_grubs', 'dragons' 열이 없습니다. 해당 열이 필요합니다.")

# # 14 시즌 이후의 데이터 필터링 및 드래곤, 전령, Void Grubs 확보 여부에 따른 승률 분석
# if {'firstdragon', 'firstherald', 'void_grubs', 'result', 'patch'}.issubset(lck_data.columns):
#     # 14 패치 이후 데이터 필터링
#     patch_14_data = lck_data[lck_data['patch'] >= 14]

#     # Void Grubs를 4개 이상 확보한 경우를 확보로 간주
#     patch_14_data['void_grubs_secured'] = patch_14_data['void_grubs'] >= 4

#     # 승률 계산
#     object_win_rates = patch_14_data.groupby('firstdragon')['result'].mean().rename('First Dragon Win Rate').to_frame()
#     object_win_rates['Herald Win Rate'] = patch_14_data.groupby('firstherald')['result'].mean()
#     object_win_rates['Void Grubs Win Rate'] = patch_14_data.groupby('void_grubs_secured')['result'].mean()

#     # 시각화
#     object_win_rates.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'orange', 'purple'])
#     plt.title('14 패치 이후 드래곤, 전령 및 Void Grubs 확보에 따른 승률 (LCK)', fontsize=16)
#     plt.xlabel('Object Secured', fontsize=12)
#     plt.ylabel('Win Rate', fontsize=12)
#     plt.legend(['First Dragon', 'First Herald', 'Void Grubs'], fontsize=10)
#     plt.xticks([0, 1], labels=['No', 'Yes'], rotation=0)
#     plt.grid(axis='y')
#     plt.show()
# else:
#     print("데이터에 'firstdragon', 'firstherald', 'void_grubs', 'result', 'patch' 열이 필요합니다.")

# # Void Grubs와 총 골드 획득량의 평균값을 깔끔하게 시각화
# if {'void_grubs', 'totalgold', 'patch'}.issubset(lck_data.columns):
#     # 14 패치 이후 데이터 필터링
#     patch_14_data = lck_data[lck_data['patch'] >= 14]

#     # Void Grubs 확보량별 총 골드 획득량 평균 계산
#     avg_gold_by_void_grubs = patch_14_data.groupby('void_grubs')['totalgold'].mean()

#     # 시각화
#     avg_gold_by_void_grubs.plot(kind='bar', figsize=(10, 6), color='purple', alpha=0.8)
#     plt.title('14 패치 이후 Void Grubs 확보량별 총 골드 획득량 평균 (LCK)', fontsize=16)
#     plt.xlabel('Void Grubs Secured', fontsize=12)
#     plt.ylabel('Average Total Gold Earned', fontsize=12)
#     plt.xticks(rotation=0)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()
# else:
#     print("데이터에 'void_grubs', 'totalgold', 'patch' 열이 필요합니다.")

# # Void Grubs 확보량별 총 골드 획득량의 평균값 막대 그래프
# if {'void_grubs', 'totalgold', 'patch'}.issubset(lck_data.columns):
#     # 14 패치 이후 데이터 필터링
#     patch_14_data = lck_data[lck_data['patch'] >= 14]

#     # Void Grubs 확보량별 총 골드 획득량 평균 계산
#     avg_gold_by_void_grubs = patch_14_data.groupby('void_grubs')['totalgold'].mean()

#     # 시각화
#     avg_gold_by_void_grubs.plot(kind='bar', figsize=(10, 6), color='purple', alpha=0.8)
#     plt.title('14 패치 이후 Void Grubs 확보량별 총 골드 획득량 평균 (LCK)', fontsize=16)
#     plt.xlabel('Void Grubs Secured', fontsize=12)
#     plt.ylabel('Average Total Gold Earned', fontsize=12)
#     plt.xticks(rotation=0)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()
# else:
#     print("데이터에 'void_grubs', 'totalgold', 'patch' 열이 필요합니다.")

"""유충"""

# Combine the two datasets into a single training dataset
combined_data = pd.concat([train_data], ignore_index=True)

# Display basic information and the first few rows of the combined dataset
combined_info = combined_data.info()
combined_head = combined_data.head()

(combined_info, combined_head)

# !pip install seaborn

if {'void_grubs', 'earned gpm', 'patch'}.issubset(combined_data.columns):
    # 14 패치 이후 데이터 필터링
    patch_14_data = combined_data[combined_data['patch'] >= 14]

    # Void Grubs 값을 오름차순 정렬
    patch_14_data['void_grubs'] = patch_14_data['void_grubs'].astype(float)
    sorted_void_grubs = sorted(patch_14_data['void_grubs'].unique())

    # Boxplot 데이터를 준비
    data = [
        patch_14_data.loc[patch_14_data['void_grubs'] == vg, 'earned gpm']
        for vg in sorted_void_grubs
    ]

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=sorted_void_grubs, patch_artist=True, boxprops=dict(facecolor="#D3CCE3"))
    plt.title('14 패치 이후 Void Grubs 확보량별 분당 골드 획득량', fontsize=16)
    plt.xlabel('Void Grubs Secured', fontsize=12)
    plt.ylabel('Earned Gold Per Minute', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("데이터에 'void_grubs', 'earned gpm', 'patch' 열이 필요합니다.")

# 2022-2024년 LCK 팀 데이터 필터링 및 상위권 팀 시각화
def visualize_top_lck_teams(data, start_year=2022, end_year=2024, league='LCK', result_column='result', team_column='teamname', top_n=5):
    if {'year', 'league', result_column, team_column}.issubset(data.columns):
        # 2022-2024년 LCK 데이터 필터링
        lck_data = data[(data['year'] >= start_year) & (data['year'] <= end_year) & (data['league'] == league)]

        # 팀별 평균 result 값 계산
        team_results = lck_data.groupby(team_column)[result_column].mean().sort_values(ascending=False)

        # 상위권 팀 선택
        top_teams = team_results.head(top_n)

        # 시각화
        plt.figure(figsize=(10, 6))
        top_teams.plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title(f'Top {top_n} LCK Teams from {start_year} to {end_year} Based on Average {result_column}', fontsize=16)
        plt.xlabel('Team Name', fontsize=12)
        plt.ylabel(f'Average {result_column}', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print(f"데이터에 'year', 'league', '{result_column}', '{team_column}' 열이 필요합니다.")

# 예시 실행
visualize_top_lck_teams(combined_data)

# # 주요 지표를 연도별로 시각화

# def visualize_gen_g_yearly_stats(data, team_name='Gen.G'):
#     # Filter data for the specific team
#     team_data = data[data['teamname'] == team_name]

#     if {'year', 'kills', 'assists', 'firstdragon', 'firstherald'}.issubset(team_data.columns):
#         # Group by year and calculate mean values for selected metrics
#         metrics = ['kills', 'assists', 'firstdragon', 'firstherald']
#         yearly_stats = team_data.groupby('year')[metrics].mean()

#         # Plot each metric across years
#         plt.figure(figsize=(10, 6))
#         for metric in metrics:
#             plt.plot(yearly_stats.index, yearly_stats[metric], marker='o', label=metric)

#         plt.title(f'{team_name} Yearly Performance Metrics', fontsize=16)
#         plt.xlabel('Year', fontsize=12)
#         plt.ylabel('Average Value', fontsize=12)
#         plt.legend(title='Metrics', fontsize=10)
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Required columns are missing in the dataset.")

# # Example usage
# visualize_gen_g_yearly_stats(combined_data)

# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 데이터 준비
# categories = ['Team KPM', 'VSPM', 'GPR', 'XP Diff at 10']
# values = gen_g_data[['team kpm', 'vspm', 'gpr', 'xpdiffat10']].mean().tolist()  # Gen.G의 각 지표 평균값

# # 레이다 차트를 위한 데이터 처리
# num_vars = len(categories)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# values += values[:1]  # 시작점으로 되돌아가기 위해 첫 값을 마지막에 추가
# angles += angles[:1]

# # 2. 레이다 차트 그리기
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# # 레이다 차트 다각형 및 외곽선 그리기
# ax.fill(angles, values, color='blue', alpha=0.25)
# ax.plot(angles, values, color='blue', linewidth=2)
# ax.set_yticks([])  # 내부 눈금 숨기기

# # 각 축 레이블 추가
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=10)

# # 제목 및 스타일 추가
# plt.title('Gen.G Performance Radar Chart', size=15, y=1.1)
# plt.show()

# Gen.G의 레이다 차트를 생성하는 함수
def create_gen_g_radar_chart(data, team_name='Gen.G'):
    # Gen.G 데이터 필터링
    gen_g_data = data[data['teamname'] == team_name]

    # 필요한 지표
    categories = ['Team KPM', 'VSPM', 'GPR', 'XP Diff at 10']
    metrics = ['team kpm', 'vspm', 'gpr', 'xpdiffat10']

    # 데이터 존재 여부 확인
    if not all(metric in gen_g_data.columns for metric in metrics):
        print(f"One or more metrics are missing: {metrics}")
        return


    # # 지표 평균값 계산
    # values = gen_g_data[metrics].mean().tolist()

    # # 레이다 차트를 위한 데이터 처리
    # num_vars = len(categories)
    # angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # values += values[:1]  # 시작점으로 되돌아가기 위해 첫 값을 마지막에 추가
    # angles += angles[:1]

    # # 레이다 차트 그리기
    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # # 레이다 차트 다각형 및 외곽선 그리기
    # ax.fill(angles, values, color='blue', alpha=0.25)
    # ax.plot(angles, values, color='blue', linewidth=2)
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories, fontsize=10)

    # # 제목 및 스타일 추가
    # plt.title(f'{team_name} Performance Radar Chart', size=15, y=1.1)
    # plt.show()

# Gen.G 팀 데이터를 필터링
gen_g_data = train_data[train_data['teamname'] == 'Gen.G']

# Gen.G 팀의 선택된 지표 평균 계산
metrics = ['dpm', 'vspm', 'gpr', 'goldat10']  # team kpm을 dpm으로 변경
gen_g_metrics = gen_g_data[metrics].mean()

# 'dpm' 값을 2500으로 나누어 스케일링
gen_g_metrics['dpm'] = gen_g_metrics['dpm'] / 2500

# 'goldat10' 값을 20000으로 나누어 스케일링
gen_g_metrics['goldat10'] = gen_g_metrics['goldat10'] / 20000

# 'vspm' 값을 조정 (10으로 나누어 스케일링)
gen_g_metrics['vspm'] = gen_g_metrics['vspm'] / 10

# 조정된 지표 값으로 값 재계산
values = gen_g_metrics.tolist()
values += [values[0]]  # 레이더 차트를 닫기 위해 첫 번째 값을 다시 추가

# 레이더 차트 설정
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += [angles[0]]  # 첫 번째 각도를 다시 추가하여 닫기

# 레이더 차트 생성
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, values, linewidth=2, linestyle='solid', label='Gen.G')
ax.fill(angles, values, alpha=0.25)

# 레이블 및 눈금 추가
ax.set_xticks(angles[:-1])  # 닫는 각도는 레이블에서 제외
ax.set_xticklabels(metrics)

# 제목 및 범례 추가
ax.set_title("Gen.G", size=16,pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 레이더 차트 표시
plt.tight_layout()
plt.show()

# 특정 팀의 데이터를 필터링하고 레이더 차트를 생성하는 함수
def create_radar_chart(teamname, data, metrics, scale_factors, title):
    team_data = data[data['teamname'] == teamname]
    team_metrics = team_data[metrics].mean()

    # 각 지표를 스케일링
    for metric, scale in scale_factors.items():
        team_metrics[metric] = team_metrics[metric] / scale

    # 값 계산 및 닫기
    values = team_metrics.tolist()
    values += [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += [angles[0]]

    # 레이더 차트 생성
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=teamname)
    ax.fill(angles, values, alpha=0.25)

    # 레이블 및 눈금 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # 제목 및 범례 추가
    ax.set_title(title, size=16, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 차트 표시
    plt.tight_layout()
    plt.show()

# 설정된 팀과 지표
teams = ['Gen.G','T1', 'KT Rolster', 'Dplus KIA']
metrics = ['dpm', 'vspm', 'gpr', 'goldat10']
scale_factors = {'dpm': 3000, 'vspm': 12, 'goldat10': 20000, 'gpr':1.2,}  # dpm, vspm, goldat10 스케일링 설정

# 각 팀의 레이더 차트 생성
for team in teams:
    create_radar_chart(
        teamname=team,
        data=train_data,
        metrics=metrics,
        scale_factors=scale_factors,
        title=f"{team}"
    )

# 연도별로 데이터를 구분하여 레이더 차트를 생성하는 함수
def create_yearly_radar_charts(data, years, teams, metrics, scale_factors):
    for year in years:
        # 특정 연도의 데이터 필터링
        yearly_data = data[data['year'] == year]
        if yearly_data.empty:
            print(f"{year}년도 데이터가 존재하지 않습니다.")
            continue

        print(f"{year}년도 데이터를 기준으로 레이더 차트를 생성합니다.")

        for team in teams:
            # 각 팀에 대해 레이더 차트 생성
            create_radar_chart(
                teamname=team,
                data=yearly_data,
                metrics=metrics,
                scale_factors=scale_factors,
                title=f"{team} 레이더 차트 - {year}"
            )

# 설정된 연도
years = [2022, 2023, 2024]

# 연도별 레이더 차트 생성
create_yearly_radar_charts(
    data=train_data,        # 데이터셋
    years=years,            # 연도 목록
    teams=teams,            # 팀 목록
    metrics=metrics,        # 지표 목록
    scale_factors=scale_factors  # 스케일링 설정
)

# 레이더 차트 생성 함수 (team kpm과 xpat10 포함)
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

    # 차트 표시
    plt.tight_layout()
    plt.show()

# 설정된 팀과 지표
teams = ['Gen.G', 'T1', 'KT Rolster', 'Dplus KIA']
metrics = ['dpm', 'vspm', 'gpr', 'goldat10', 'team kpm', 'xpat10']  # 'team kpm'과 'xpat10' 포함
scale_factors = {
    'dpm': 3000,
    'vspm': 12,
    'goldat10': 20000,
    'gpr': 1.2,
    'team kpm': 1,  # 필요하면 수정 가능
    'xpat10': 25000  # 추가된 스케일링
}

# 연도별 레이더 차트 생성
def create_yearly_radar_charts(data, years, teams, metrics, scale_factors):
    for year in years:
        # 특정 연도의 데이터 필터링
        yearly_data = data[data['year'] == year]
        if yearly_data.empty:
            print(f"{year}년도 데이터가 존재하지 않습니다.")
            continue

        print(f"{year}년도 데이터를 기준으로 레이더 차트를 생성합니다.")

        for team in teams:
            # 각 팀에 대해 레이더 차트 생성
            create_radar_chart(
                teamname=team,
                data=yearly_data,
                metrics=metrics,
                scale_factors=scale_factors,
                title=f"{team} 레이더 차트 - {year}"
            )

# 설정된 연도
years = [2022, 2023, 2024]

# 연도별 레이더 차트 생성
create_yearly_radar_charts(
    data=train_data,        # 데이터셋
    years=years,            # 연도 목록
    teams=teams,            # 팀 목록
    metrics=metrics,        # 지표 목록
    scale_factors=scale_factors  # 스케일링 설정
)

# 레이더 차트 생성 함수 (축 회전 추가)
def create_radar_chart(teamname, data, metrics, scale_factors, title, angle_offset=0):
    team_data = data[data['teamname'] == teamname]
    team_metrics = team_data[metrics].mean()

    # 각 지표를 스케일링
    for metric, scale in scale_factors.items():
        team_metrics[metric] = team_metrics[metric] / scale

    # 값 계산 및 닫기
    values = team_metrics.tolist()
    values += [values[0]]  # 시작점으로 돌아가기
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles = [(a + angle_offset) % (2 * np.pi) for a in angles]  # 회전 추가
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

    # 차트 표시
    plt.tight_layout()
    plt.show()

# 레이더 차트 생성 시 각도 회전 적용
angle_offset = np.pi / 4
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

# 연도별 레이더 차트 생성 (축 회전 포함)
create_yearly_radar_charts(
    data=train_data,
    years=[2022, 2023, 2024],
    teams=teams,
    metrics=metrics,
    scale_factors=scale_factors
)