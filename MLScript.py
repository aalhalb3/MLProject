import sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer  # <-- NEW

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
print("[INFO] Imports done.")

# Load Data from SQLite

print("[INFO] Connecting to SQLite database...")
conn = sqlite3.connect("database.sqlite")  # adjust path if needed
print("[INFO] Connection opened.")

print("[INFO] Loading tables...")
matches     = pd.read_sql_query("SELECT * FROM Match", conn)
teams       = pd.read_sql_query("SELECT * FROM Team", conn)
team_attr   = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
players     = pd.read_sql_query("SELECT * FROM Player", conn)
player_attr = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
leagues     = pd.read_sql_query("SELECT * FROM League", conn)
countries   = pd.read_sql_query("SELECT * FROM Country", conn)

conn.close()
print("[INFO] Connection closed.")

print(f"[SHAPE] matches: {matches.shape}")
print(f"[SHAPE] teams: {teams.shape}")
print(f"[SHAPE] team_attr: {team_attr.shape}")
print(f"[SHAPE] player_attr: {player_attr.shape}")