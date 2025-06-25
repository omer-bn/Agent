# agent.py

import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config for dark theme and left alignment
st.set_page_config(layout="wide", page_title="Tactical Dashboard")
st.markdown("""
    <style>
        body, .stApp {background-color: black !important; color: white !important; text-align: left;}
        .css-1d391kg, .css-1v3fvcr {color: white !important;}
        .stDataFrame th, .stDataFrame td {color: white !important; background-color: #111 !important;}
        h2, h3, h4 {color: white !important;}
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
df_players  = pd.read_csv("players_data-2024_2025.csv", encoding='utf-8-sig')
df_teams    = pd.read_csv("team_stats.csv", encoding='utf-8-sig')
df_fixtures = pd.read_csv("2024-2025 fixtures.csv", encoding='utf-8-sig')
df          = pd.read_csv("Merged_Fixture-Team_Data.csv", encoding='utf-8-sig')
# -------------------- SIDEBAR --------------------
st.sidebar.title("Team Selector")
teams = sorted(set(df_players['Squad'].dropna()) & set(df_teams['team'].dropna()))
selected_team = st.sidebar.selectbox("Choose a team", teams)

# -------------------- TITLES --------------------
st.title(f"Tactical Dashboard: {selected_team}")

# -------------------- BEST PLAYMAKER --------------------
st.header("Best Playmaker")
team_df = df_players[df_players['Squad'] == selected_team].copy()

if 'PrgP' in team_df.columns and 'xAG' in team_df.columns:
    team_df['PlaymakerScore'] = team_df['PrgP'] + team_df['xAG']
    top_playmaker = team_df.sort_values(by='PlaymakerScore', ascending=False).iloc[0]

    st.subheader(f"{top_playmaker['Player']}")
    st.markdown(f"- Progressive Passes: `{top_playmaker['PrgP']}`")
    st.markdown(f"- Expected Assists (xAG): `{top_playmaker['xAG']}`")
else:
    st.warning("Missing columns: 'PrgP' or 'xAG'.")



# -------------------- TEAM STYLE ANALYSIS --------------------
st.header("Team Style Analysis")

style_descriptions = []
avg_dist = team_df['Dist'].mean() if 'Dist' in team_df else None
avg_prgp = team_df['PrgP'].mean() if 'PrgP' in team_df else None
avg_tkl = team_df['Tkl'].mean() if 'Tkl' in team_df else None

if avg_dist:
    if avg_dist > 25:
        style_descriptions.append("Runs more than average")
    elif avg_dist < 18:
        style_descriptions.append("Runs less than average")
if avg_prgp:
    if avg_prgp > 50:
        style_descriptions.append("builds up through progressive passing")
if avg_tkl:
    if avg_tkl > 30:
        style_descriptions.append("uses a high pressing system")

if style_descriptions:
    st.markdown("This team typically: **" + ", and ".join(style_descriptions) + ".**")
else:
    st.markdown("Not enough data to define team style.")

st.header("Tactical Summary (Team-Level)")

team_row = df_teams[df_teams['team'] == selected_team].squeeze()
league_avg_age = df_players[df_players['Age'].notna()].groupby('Squad')['Age'].mean().mean()
avg_age = team_df['Age'].mean() if 'Age' in team_df else None
league_avg_goals = df_teams['goals'].mean()
league_avg_xg = df_teams['expected_goals'].mean()
league_avg_assists = df_teams['assists'].mean()
league_avg_prgcarries = df_teams['progressive_carries'].mean() if 'progressive_carries' in df_teams else None

col1, col2 = st.columns(2)
col1.metric("Possession", f"{team_row['possession']}%")
col2.metric("Goals Scored", team_row['goals'], delta=f"{team_row['goals'] - league_avg_goals:.1f} vs avg")

col3, col4 = st.columns(2)
col3.metric("Assists", team_row['assists'], delta=f"{team_row['assists'] - league_avg_assists:.1f} vs avg")
col4.metric("Expected Goals (xG)", round(team_row['expected_goals'], 2), delta=f"{team_row['expected_goals'] - league_avg_xg:.2f} vs avg")

col5, col6 = st.columns(2)
avg_prgp_league = df_teams['progressive_passes'].mean()
col5.metric(
    label="Progressive Passes",
    value=int(team_row['progressive_passes']),
    delta=f"{team_row['progressive_passes'] - avg_prgp_league:.1f} vs avg",
    delta_color="normal"
)
if league_avg_prgcarries:
    col6.metric("Progressive Carries", int(team_row['progressive_carries']), delta=f"{team_row['progressive_carries'] - league_avg_prgcarries:.1f} vs avg")

st.markdown("---")
col7, _ = st.columns(2)
if avg_age:
    col7.metric(
        label="Average Age",
        value=f"{avg_age:.1f}",
        delta=f"{avg_age - league_avg_age:.1f} vs avg",
        delta_color="normal"
    )

# -------------------- ML MODEL --------------------
features = [
    'Home_expected_goals', 'Away_expected_goals',
    'Home_progressive_passes', 'Away_progressive_passes',
    'Home_progressive_carries', 'Away_progressive_carries',
    'Home_possession', 'Away_possession'
]

df_clean = df[df['FTR'].notna()].copy()
df_clean[features] = df_clean[features].fillna(df_clean[features].mean())

X = df_clean[features]
y = df_clean['FTR']
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------- SIMULATION TABLE --------------------
st.subheader("Simulated Matches Based on Season Averages")

all_results = []
for opp_team in teams:
    if opp_team == selected_team:
        continue

    home_row = df_teams[df_teams['team'] == selected_team].squeeze()
    away_row = df_teams[df_teams['team'] == opp_team].squeeze()

    match_features = pd.DataFrame([{ 
        'Home_expected_goals': home_row['expected_goals'],
        'Away_expected_goals': away_row['expected_goals'],
        'Home_progressive_passes': home_row['progressive_passes'],
        'Away_progressive_passes': away_row['progressive_passes'],
        'Home_progressive_carries': home_row['progressive_carries'],
        'Away_progressive_carries': away_row['progressive_carries'],
        'Home_possession': home_row['possession'],
        'Away_possession': away_row['possession']
    }])

    pred_result = model.predict(match_features)[0]
    all_results.append({
        "Home Team": selected_team,
        "Away Team": opp_team,
        "Predicted Result": pred_result
    })

for opp_team in teams:
    if opp_team == selected_team:
        continue

    home_row = df_teams[df_teams['team'] == opp_team].squeeze()
    away_row = df_teams[df_teams['team'] == selected_team].squeeze()

    match_features = pd.DataFrame([{ 
        'Home_expected_goals': home_row['expected_goals'],
        'Away_expected_goals': away_row['expected_goals'],
        'Home_progressive_passes': home_row['progressive_passes'],
        'Away_progressive_passes': away_row['progressive_passes'],
        'Home_progressive_carries': home_row['progressive_carries'],
        'Away_progressive_carries': away_row['progressive_carries'],
        'Home_possession': home_row['possession'],
        'Away_possession': away_row['possession']
    }])

    pred_result = model.predict(match_features)[0]
    all_results.append({
        "Home Team": opp_team,
        "Away Team": selected_team,
        "Predicted Result": pred_result
    })

match_df = pd.DataFrame(all_results)
match_df.index = match_df.index + 1  # Start count from 1

def highlight_prediction(row):
    if (row['Home Team'] == selected_team and row['Predicted Result'] == 'H') or \
       (row['Away Team'] == selected_team and row['Predicted Result'] == 'A'):
        return ["", "", "background-color: #007BFF"]  # Blue
    elif (row['Home Team'] == selected_team and row['Predicted Result'] == 'A') or \
         (row['Away Team'] == selected_team and row['Predicted Result'] == 'H'):
        return ["", "", "background-color: #DC3545"]  # Red
    else:
        return ["", "", "background-color: #6C757D"]  # Gray

st.dataframe(match_df.style.apply(highlight_prediction, axis=1))

# -------------------- PERSONALIZED TEAM ACCURACY --------------------
df_team = df[(df['HomeTeam'] == selected_team) | (df['AwayTeam'] == selected_team)].copy()
df_team[features] = df_team[features].fillna(df[features].mean())
df_team['Prediction'] = model.predict(df_team[features])

def label_team_view(row, team):
    if row['HomeTeam'] == team:
        return row['Prediction']
    elif row['Prediction'] == 'H':
        return 'A'
    elif row['Prediction'] == 'A':
        return 'H'
    else:
        return 'D'

df_team['TeamResult'] = df_team.apply(lambda r: label_team_view(r, selected_team), axis=1)

actual_team_results = df_team.apply(
    lambda row: 'H' if row['HomeTeam'] == selected_team and row['FTR'] == 'H' else
                'A' if row['AwayTeam'] == selected_team and row['FTR'] == 'A' else
                'D',
    axis=1
)
accuracy_team = accuracy_score(actual_team_results, df_team['TeamResult'])
st.write(f"**Model Accuracy for {selected_team}:** {accuracy_team:.2%}")

# -------------------- PERSONALIZED CONFUSION MATRIX --------------------
cm_team = confusion_matrix(actual_team_results, df_team['TeamResult'], labels=['H', 'D', 'A'])
st.write("### Personalized Confusion Matrix")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm_team, annot=True, fmt="d", cmap="Blues", xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
