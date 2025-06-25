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

# -------------------- PLAYMAKER ANALYSIS --------------------
st.header("Playmaker Analysis")
df_team = df_players[df_players['Squad'] == selected_team].copy()
required_columns = ['PrgP', 'xAG', 'Assists', 'Key Passes']
missing_cols = [col for col in required_columns if col not in df_team.columns]

if missing_cols:
    st.warning(f"Missing columns in dataset: {', '.join(missing_cols)}")
else:
    df_team['Playmaker Index'] = df_team['PrgP'] * 0.4 + df_team['xAG'] * 0.4 + df_team['Assists'] * 0.1 + df_team['Key Passes'] * 0.1
    top_playmaker = df_team.sort_values(by='Playmaker Index', ascending=False).head(1)
    st.dataframe(top_playmaker[['Player', 'PrgP', 'xAG', 'Assists', 'Key Passes', 'Playmaker Index']].set_index('Player'))

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

# -------------------- SIMULATED FIXTURES --------------------
st.subheader("Simulated Matches Based on Season Averages")

simulation_results = []

for opponent in teams:
    if opponent == selected_team:
        continue

    # Home game
    home_row = df_teams[df_teams['team'] == selected_team].squeeze()
    away_row = df_teams[df_teams['team'] == opponent].squeeze()

    features_row = pd.DataFrame([{ 
        'Home_expected_goals': home_row['expected_goals'],
        'Away_expected_goals': away_row['expected_goals'],
        'Home_progressive_passes': home_row['progressive_passes'],
        'Away_progressive_passes': away_row['progressive_passes'],
        'Home_progressive_carries': home_row['progressive_carries'],
        'Away_progressive_carries': away_row['progressive_carries'],
        'Home_possession': home_row['possession'],
        'Away_possession': away_row['possession']
    }])

    prediction = model.predict(features_row)[0]
    simulation_results.append({"Home Team": selected_team, "Away Team": opponent, "Predicted Result": prediction})

    # Away game
    features_row = pd.DataFrame([{ 
        'Home_expected_goals': away_row['expected_goals'],
        'Away_expected_goals': home_row['expected_goals'],
        'Home_progressive_passes': away_row['progressive_passes'],
        'Away_progressive_passes': home_row['progressive_passes'],
        'Home_progressive_carries': away_row['progressive_carries'],
        'Away_progressive_carries': home_row['progressive_carries'],
        'Home_possession': away_row['possession'],
        'Away_possession': home_row['possession']
    }])

    prediction = model.predict(features_row)[0]
    simulation_results.append({"Home Team": opponent, "Away Team": selected_team, "Predicted Result": prediction})

sim_df = pd.DataFrame(simulation_results)
sim_df.index = sim_df.index + 1

def highlight_results(row):
    if row['Home Team'] == selected_team and row['Predicted Result'] == 'H':
        return ["", "", "background-color: #007BFF"]
    elif row['Away Team'] == selected_team and row['Predicted Result'] == 'A':
        return ["", "", "background-color: #007BFF"]
    elif row['Home Team'] == selected_team and row['Predicted Result'] == 'A':
        return ["", "", "background-color: #DC3545"]
    elif row['Away Team'] == selected_team and row['Predicted Result'] == 'H':
        return ["", "", "background-color: #DC3545"]
    else:
        return ["", "", "background-color: #6C757D"]

st.dataframe(sim_df.style.apply(highlight_results, axis=1))

# -------------------- PERSONALIZED CONFUSION MATRIX --------------------
df_team_matches = df[(df['HomeTeam'] == selected_team) | (df['AwayTeam'] == selected_team)].copy()
df_team_matches[features] = df_team_matches[features].fillna(df[features].mean())
df_team_matches['Prediction'] = model.predict(df_team_matches[features])

def label_team(row):
    if row['HomeTeam'] == selected_team:
        return row['Prediction']
    elif row['Prediction'] == 'H':
        return 'A'
    elif row['Prediction'] == 'A':
        return 'H'
    else:
        return 'D'

df_team_matches['TeamResult'] = df_team_matches.apply(lambda r: label_team(r), axis=1)
actual_results = df_team_matches.apply(
    lambda row: 'H' if row['HomeTeam'] == selected_team and row['FTR'] == 'H' else
                'A' if row['AwayTeam'] == selected_team and row['FTR'] == 'A' else
                'D', axis=1
)

accuracy = accuracy_score(actual_results, df_team_matches['TeamResult'])
st.write(f"**Model Accuracy for {selected_team}:** {accuracy:.2%}")

cm = confusion_matrix(actual_results, df_team_matches['TeamResult'], labels=['H', 'D', 'A'])
st.write("### Personalized Confusion Matrix")
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['H', 'D', 'A'], yticklabels=['H', 'D', 'A'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
