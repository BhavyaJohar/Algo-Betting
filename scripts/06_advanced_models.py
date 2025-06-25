"""Step 6: Advanced modeling (Tree-based ensembles: RandomForest, XGBoost, LightGBM)

 - Load engineered feature dataset
 - Split into train/test (time-based)
 - Train and evaluate multiple classifiers
 - Report accuracy, ROC AUC, log loss, Brier score, and feature importances
"""
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss


def load_features(path="data/processed/matches_features.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["tourney_date"])
    raise FileNotFoundError(f"No feature dataset at {path} or {csv_path}")


def split_train_test(df, train_end_date="2019-01-01"):
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    cutoff = pd.to_datetime(train_end_date)
    train = df[df["tourney_date"] < cutoff].reset_index(drop=True)
    test = df[df["tourney_date"] >= cutoff].reset_index(drop=True)
    return train, test


def train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_cols):
    results = {}
    for name, model in models:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "log_loss": log_loss(y_test, proba),
            "brier_score": brier_score_loss(y_test, proba),
        }
        results[name] = {"metrics": metrics}
        if hasattr(model, "feature_importances_"):
            import numpy as np

            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1]
            fi = [(feature_cols[i], imp[i]) for i in idx]
            results[name]["feature_importance"] = fi
    return results


def main():
    print("Loading feature dataset...")
    df = load_features()

    print("Splitting train/test...")
    train, test = split_train_test(df)
    print(f"Train matches: {len(train)},  Test matches: {len(test)}")

    feature_cols = [
        "elo_diff", "elo_surf_diff",
        "age_diff", "ht_diff",
        "rank_diff", "rank_points_diff",
        "surf_win_rate_diff",
        "player1_h2h_win_rate", "h2h_count",
    ]
    X_train = train[feature_cols].fillna(0)
    y_train = train["target"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["target"]

    models = []
    models.append(("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)))
    try:
        from xgboost import XGBClassifier

        models.append(("XGBoost", XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric="logloss", random_state=42)))
    except ImportError:
        print("xgboost not installed; skipping XGBoost.")
    try:
        from lightgbm import LGBMClassifier

        models.append(("LightGBM", LGBMClassifier(n_estimators=50, random_state=42)))
    except ImportError:
        print("lightgbm not installed; skipping LightGBM.")

    print("Training and evaluating advanced models...")
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test, feature_cols)
    for name, out in results.items():
        print(f"\nModel: {name}")
        for m, v in out["metrics"].items():
            print(f" - {m}: {v:.4f}")
        if "feature_importance" in out:
            print(" Top features:")
            for feat, score in out["feature_importance"][:10]:
                print(f"    {feat}: {score:.4f}")


if __name__ == "__main__":
    main()