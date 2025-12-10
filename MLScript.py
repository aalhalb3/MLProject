# Imports 
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

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# IF DATASET NOT DOWNLOADED USE THIS CODE:

# Also Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

'''
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "" #Insert workspace path

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "hugomathien/soccer",
  file_path,
)'''

# Load data from SQLite Database


print("[INFO] Connecting to SQLite database...")
conn = sqlite3.connect("database.sqlite")  # adjust path if needed
print("[INFO] Connection opened.")

print("[INFO] Loading tables...")
matches   = pd.read_sql_query("SELECT * FROM Match", conn)
teams     = pd.read_sql_query("SELECT * FROM Team", conn)
team_attr = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)
players   = pd.read_sql_query("SELECT * FROM Player", conn)
player_attr = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)
leagues   = pd.read_sql_query("SELECT * FROM League", conn)
countries = pd.read_sql_query("SELECT * FROM Country", conn)

conn.close()
print("[INFO] Connection closed.")

print(f"[SHAPE] matches: {matches.shape}")
print(f"[SHAPE] teams: {teams.shape}")
print(f"[SHAPE] team_attr: {team_attr.shape}")
print(f"[SHAPE] player_attr: {player_attr.shape}")

# 2. Helper functions

def latest_team_attributes(team_attr_df):
    """
    Keep the latest team attributes per team & season (date).
    """
    df = team_attr_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["team_api_id", "date"], inplace=True)
    latest = df.groupby("team_api_id").tail(1)
    return latest

def latest_player_attributes(player_attr_df):
    """
    Keep the latest attributes per player.
    """
    df = player_attr_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["player_api_id", "date"], inplace=True)
    latest = df.groupby("player_api_id").tail(1)
    return latest

print("[INFO] Computing latest team and player attributes...")
team_attr_latest = latest_team_attributes(team_attr)
player_attr_latest = latest_player_attributes(player_attr)
print(f"[SHAPE] team_attr_latest: {team_attr_latest.shape}")
print(f"[SHAPE] player_attr_latest: {player_attr_latest.shape}")

print("[INFO] Encoding target variable (result)...")

# 3.1 Build match‑level features

def encode_result(row):
    if row["home_team_goal"] > row["away_team_goal"]:
        return 2  # Home win
    elif row["home_team_goal"] < row["away_team_goal"]:
        return 0  # Away win
    else:
        return 1  # Draw

matches["result"] = matches.apply(encode_result, axis=1)
print("[INFO] Result column added. Value counts:")
print(matches["result"].value_counts())

# 3.2 Map team attributes to matches (home & away)

print("[INFO] Merging team attributes into matches...")
team_attr_cols = [
    "buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing",
    "chanceCreationCrossing", "chanceCreationShooting",
    "defencePressure", "defenceAggression", "defenceTeamWidth"
]

team_attr_latest_small = team_attr_latest[["team_api_id"] + team_attr_cols].drop_duplicates("team_api_id")
print(f"[SHAPE] team_attr_latest_small: {team_attr_latest_small.shape}")

matches = matches.merge(
    team_attr_latest_small.add_prefix("home_team_"),
    left_on="home_team_api_id",
    right_on="home_team_team_api_id",
    how="left"
)
matches = matches.merge(
    team_attr_latest_small.add_prefix("away_team_"),
    left_on="away_team_api_id",
    right_on="away_team_team_api_id",
    how="left"
)

matches.drop(columns=[c for c in matches.columns if c.endswith("_team_api_id")], inplace=True)
print(f"[INFO] After merging team attributes, matches shape: {matches.shape}")

# 3.3 Aggregate player ratings

print("[INFO] Computing lineup stats (average ratings)...")
player_basic_cols = ["player_api_id", "overall_rating", "stamina", "reactions"]
player_attr_small = player_attr_latest[player_basic_cols]
print(f"[SHAPE] player_attr_small: {player_attr_small.shape}")

def compute_lineup_stats(row, side_prefix):
    player_ids = [row[f"{side_prefix}_player_{i}"] for i in range(1, 12)]
    sub = player_attr_small[player_attr_small["player_api_id"].isin(player_ids)]
    if sub.empty:
        return pd.Series({
            f"{side_prefix}_avg_overall": np.nan,
            f"{side_prefix}_avg_stamina": np.nan,
            f"{side_prefix}_avg_reactions": np.nan,
        })
    return pd.Series({
        f"{side_prefix}_avg_overall": sub["overall_rating"].mean(),
        f"{side_prefix}_avg_stamina": sub["stamina"].mean(),
        f"{side_prefix}_avg_reactions": sub["reactions"].mean(),
    })

for side in ["home", "away"]:
    print(f"[INFO] Computing lineup stats for {side} side...")
    stats = matches.apply(compute_lineup_stats, axis=1, side_prefix=side)
    matches = pd.concat([matches, stats], axis=1)

print("[INFO] Lineup stats added.")
print(matches[[c for c in matches.columns if "avg_overall" in c]].head())

# 3.4 Bookmaker odds & implied probabilities

print("[INFO] Processing bookmaker odds...")
book_cols = ["B365H", "B365D", "B365A"]
matches[book_cols] = matches[book_cols].astype(float)

for col in book_cols:
    null_before = matches[col].isna().sum()
    matches[col] = matches[col].fillna(matches[col].median())
    null_after = matches[col].isna().sum()
    print(f"[IMPUTE] {col}: {null_before} -> {null_after} NaNs")

def implied_probabilities(row):
    odds = np.array([row["B365H"], row["B365D"], row["B365A"]], dtype=float)
    inv = 1.0 / odds
    s = inv.sum()
    if s == 0:
        return pd.Series({
            "prob_home": np.nan,
            "prob_draw": np.nan,
            "prob_away": np.nan
        })
    return pd.Series({
        "prob_home": inv[0] / s,
        "prob_draw": inv[1] / s,
        "prob_away": inv[2] / s
    })

print("[INFO] Computing implied probabilities from odds...")
prob_df = matches.apply(implied_probabilities, axis=1)
matches = pd.concat([matches, prob_df], axis=1)
print(matches[["prob_home", "prob_draw", "prob_away"]].head())

# 3.5 Home/away context & meta info

print("[INFO] Adding home_advantage, season, league_name...")
matches["home_advantage"] = 1
matches["season"] = matches["season"].astype(str)
matches = matches.merge(leagues[["id", "name"]], left_on="league_id", right_on="id", how="left")
matches.rename(columns={"name": "league_name"}, inplace=True)
matches.drop(columns=["id_y"], inplace=True, errors="ignore")
print(f"[INFO] Final matches shape after feature engineering: {matches.shape}")
print(matches[["league_name", "season", "home_advantage"]].head())


# 4. Handle missing data & final feature set

essential_cols = [
    "home_team_goal", "away_team_goal",
    "home_avg_overall", "away_avg_overall",
    "prob_home", "prob_draw", "prob_away"
]
matches_clean = matches.dropna(subset=essential_cols)
print("[INFO] matches_clean shape:", matches_clean.shape)

numeric_features = [
    "home_avg_overall", "home_avg_stamina", "home_avg_reactions",
    "away_avg_overall", "away_avg_stamina", "away_avg_reactions",
    "prob_home", "prob_draw", "prob_away",
    "home_team_buildUpPlaySpeed", "home_team_buildUpPlayPassing",
    "home_team_chanceCreationPassing", "home_team_chanceCreationCrossing",
    "home_team_chanceCreationShooting",
    "home_team_defencePressure", "home_team_defenceAggression",
    "home_team_defenceTeamWidth",
    "away_team_buildUpPlaySpeed", "away_team_buildUpPlayPassing",
    "away_team_chanceCreationPassing", "away_team_chanceCreationCrossing",
    "away_team_chanceCreationShooting",
    "away_team_defencePressure", "away_team_defenceAggression",
    "away_team_defenceTeamWidth"
]

categorical_features = ["league_name", "season"]

X = matches_clean[numeric_features + categorical_features]
y = matches_clean["result"].astype(int)

print("[INFO] NaNs per numeric feature BEFORE split:")
print(X[numeric_features].isna().sum().sort_values(ascending=False).head(10))

# 5. Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("[INFO] Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 6. Preprocessing pipeline

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 7. Models

log_reg = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=RANDOM_STATE
)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

gb_clf = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=150,
    random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "Gradient Boosting": gb_clf
}

# 8. Training & CV evaluation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\n[INFO] Running CV for {name}...")
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    acc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"=== {name} ===")
    print(f"CV Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"CV Macro F1: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    models[name] = pipe  # store full pipeline

# 9. Fit on full train & test evaluation

results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    # Brier score (multi‑class: average of per‑class Brier scores)
    # Convert y_test to one‑hot
    y_test_oh = np.eye(3)[y_test.values]
    brier = np.mean(np.sum((y_proba - y_test_oh) ** 2, axis=1))

    results[name] = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "brier_score": brier,
    }

    print(f"\n=== {name} TEST RESULTS ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"Brier score: {brier:.3f}")
    print("Confusion matrix (rows=true, cols=pred; 0=Away,1=Draw,2=Home):")
    print(cm)

# 10. Basic EDA (home advantage, outcome distribution, correlations)

# Outcome distribution
outcome_counts = matches_clean["result"].value_counts(normalize=True).sort_index()
print("\nOutcome distribution (0=Away,1=Draw,2=Home):")
print(outcome_counts)

# Bar plot
plt.figure(figsize=(4, 3))
sns.barplot(x=outcome_counts.index, y=outcome_counts.values)
plt.xticks([0, 1, 2], ["Away", "Draw", "Home"])
plt.ylabel("Proportion")
plt.title("Match outcome distribution")
plt.tight_layout()
plt.show()

# correlation heatmap of numeric features
plt.figure(figsize=(12, 10))
corr = matches_clean[numeric_features + ["home_team_goal", "away_team_goal"]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation heatmap (numeric features & goals)")
plt.tight_layout()
plt.show()

# 11. Feature importance (RF & GBM)

def get_feature_names(preprocessor, numeric_features, categorical_features):
    """
    Get transformed feature names from ColumnTransformer.
    """
    num_features_out = numeric_features
    cat_transformer = preprocessor.named_transformers_["cat"]["onehot"]
    cat_features_out = list(cat_transformer.get_feature_names_out(categorical_features))
    return num_features_out + cat_features_out

# Random Forest feature importance
rf_pipe = models["Random Forest"]
rf_model = rf_pipe.named_steps["model"]
feature_names = get_feature_names(rf_pipe.named_steps["preprocess"], numeric_features, categorical_features)

rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 15 RF feature importances:")
print(rf_importances.head(15))

plt.figure(figsize=(8, 5))
rf_importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Random Forest Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# Gradient Boosting feature importance
gb_pipe = models["Gradient Boosting"]
gb_model = gb_pipe.named_steps["model"]
gb_importances = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 15 GB feature importances:")
print(gb_importances.head(15))

plt.figure(figsize=(8, 5))
gb_importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Gradient Boosting Top 15 Feature Importances")
plt.tight_layout()
plt.show()
