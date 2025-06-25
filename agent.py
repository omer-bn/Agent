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

# Prepare filtered team data
df_team = df_players[df_players['Squad'] == selected_team].copy()

# -------------------- BEST PLAYMAKER --------------------
st.header("Best Playmaker")

if 'PrgP' in df_team.columns and 'xAG' in df_team.columns:
    df_team['PlaymakerScore'] = df_team['PrgP'] + df_team['xAG']
    top_playmaker = df_team.sort_values(by='PlaymakerScore', ascending=False).iloc[0]

    st.subheader(f"{top_playmaker['Player']}")
    st.markdown(f"- Progressive Passes: `{top_playmaker['PrgP']}`")
    st.markdown(f"- Expected Assists (xAG): `{top_playmaker['xAG']}`")
else:
    st.warning("Missing columns: 'PrgP' or 'xAG'.")

# -------------------- TEAM STYLE ANALYSIS --------------------
st.header("Team Style Analysis")

style_descriptions = []
avg_dist = df_team['Dist'].mean() if 'Dist' in df_team else None
avg_prgp = df_team['PrgP'].mean() if 'PrgP' in df_team else None
avg_tkl = df_team['Tkl'].mean() if 'Tkl' in df_team else None

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
avg_age = df_team['Age'].mean() if 'Age' in df_team else None
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
