"""
Step 5: Train/test split & baseline modeling

- Load engineered feature dataset
- Split into train/test by time (train before 2019, test 2019+)
- Train a logistic regression baseline on selected features
- Evaluate accuracy, ROC AUC, log loss, Brier score
"""
import os
import pandas as pd

def load_features(path="data/processed/matches_features.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["tourney_date"])
    raise FileNotFoundError(f"No features data at {path} or {csv_path}")

def split_train_test(df, train_end_date="2019-01-01"):
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    cutoff = pd.to_datetime(train_end_date)
    train = df[df["tourney_date"] < cutoff].reset_index(drop=True)
    test = df[df["tourney_date"] >= cutoff].reset_index(drop=True)
    return train, test

def train_baseline(train, test, feature_cols, target_col="target"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

    X_train = train[feature_cols].fillna(0)
    y_train = train[target_col]
    X_test = test[feature_cols].fillna(0)
    y_test = test[target_col]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
    }
    return model, metrics

def main():
    print("Loading feature dataset...")
    df = load_features()

    print("Splitting train/test...")
    train, test = split_train_test(df)
    print(f"Train matches: {len(train)},  Test matches: {len(test)}")

    # Select baseline feature subset
    feature_cols = [
        "elo_diff", "elo_surf_diff",
        "age_diff", "ht_diff",
        "rank_diff", "rank_points_diff",
        "surf_win_rate_diff",
        "player1_h2h_win_rate",
        "h2h_count",
    ]
    print("Training baseline logistic regression on features:", feature_cols)
    model, metrics = train_baseline(train, test, feature_cols)
    print("Baseline metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")

if __name__ == "__main__":
    main()