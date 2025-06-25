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
st.header("Playmaker Analysis")
df_team = df_players[df_players['Squad'] == selected_team].copy()
required_columns = ['PrgP', 'xAG', 'KeyPasses', 'Assists']

missing_cols = [col for col in required_columns if col not in df_team.columns]
if missing_cols:
    st.warning(f"Missing columns for playmaker analysis: {', '.join(missing_cols)}")
else:
    df_team['PlaymakerScore'] = (
        df_team['PrgP'] * 0.4 + df_team['xAG'] * 0.3 + df_team['KeyPasses'] * 0.2 + df_team['Assists'] * 0.1
    )
    top_playmakers = df_team.sort_values(by='PlaymakerScore', ascending=False).head(3)
    st.write("Top 3 Playmakers based on advanced metrics:")
    st.dataframe(top_playmakers[['Player', 'PrgP', 'xAG', 'KeyPasses', 'Assists', 'PlaymakerScore']])

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

# -------------------- PERSONALIZED TEAM ACCURACY --------------------
df_team_matches = df[(df['HomeTeam'] == selected_team) | (df['AwayTeam'] == selected_team)].copy()
df_team_matches[features] = df_team_matches[features].fillna(df[features].mean())
df_team_matches['Prediction'] = model.predict(df_team_matches[features])

def label_team_view(row, team):
    if row['HomeTeam'] == team:
        return row['Prediction']
    elif row['Prediction'] == 'H':
        return 'A'
    elif row['Prediction'] == 'A':
        return 'H'
    else:
        return 'D'

df_team_matches['TeamResult'] = df_team_matches.apply(lambda r: label_team_view(r, selected_team), axis=1)

actual_team_results = df_team_matches.apply(
    lambda row: 'H' if row['HomeTeam'] == selected_team and row['FTR'] == 'H' else
                'A' if row['AwayTeam'] == selected_team and row['FTR'] == 'A' else
                'D',
    axis=1
)
accuracy_team = accuracy_score(actual_team_results, df_team_matches['TeamResult'])
st.write(f"**Model Accuracy for {selected_team}:** {accuracy_team:.2%}")

# -------------------- PERSONALIZED CONFUSION MATRIX --------------------
cm_team = confusion_matrix(actual_team_results, df_team_matches['TeamResult'], labels=['H', 'D', 'A'])
st.write("### Personalized Confusion Matrix")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm_team, annot=True, fmt="d", cmap="Blues", xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
