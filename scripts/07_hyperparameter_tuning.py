"""
Step 7: Hyperparameter tuning for tree-based models

- Load engineered feature dataset
- Split train/test by date
- Define parameter distributions for XGBoost and LightGBM
- Run RandomizedSearchCV to tune each model on train set
- Evaluate best models on test set and report metrics and best parameters
"""
import os
import json
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
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


def report_metrics(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
    }
    print(f"\n{name} test set metrics:")
    for m, v in metrics.items():
        print(f" - {m}: {v:.4f}")


def tune_model(name, model, param_dist, X_train, y_train,
               n_iter=10, cv=3, random_state=42, n_jobs=-1):
    print(f"Tuning {name}...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2,
    )
    search.fit(X_train, y_train)
    print(f"Best neg_log_loss for {name}: {search.best_score_:.4f}")
    return name, search.best_estimator_, search.best_params_


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

    results = []
    try:
        from xgboost import XGBClassifier

        xgb_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 1, 5],
        }
        results.append(
            tune_model(
                "XGBoost",
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                ),
                xgb_dist,
                X_train,
                y_train,
            )
        )
    except ImportError:
        print("xgboost not installed; skipping XGBoost tuning.")

    try:
        from lightgbm import LGBMClassifier

        lgb_dist = {
            "n_estimators": [50, 100, 200],
            "num_leaves": [31, 50, 100],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        results.append(
            tune_model("LightGBM", LGBMClassifier(random_state=42), lgb_dist, X_train, y_train)
        )
    except ImportError:
        print("lightgbm not installed; skipping LightGBM tuning.")

    for name, best_model, best_params in results:
        print(f"\n{name} best parameters: {best_params}")
        report_metrics(name, best_model, X_test, y_test)

    best_params_dict = {name: params for name, _, params in results}
    out_path = os.path.join("data", "processed", "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Saved best parameters to {out_path}")


if __name__ == "__main__":
    main()