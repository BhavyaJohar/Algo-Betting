"""
Step 8: Final evaluation & calibration

- Load engineered features and best hyperparameters
- Split train/test by date
- Train tuned models on train set
- Evaluate on test set (accuracy, ROC-AUC, log-loss, Brier score)
- Compute calibration curves and output numeric reliability data (or plots if matplotlib is installed)
"""
import os
import json

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay

try:
    import matplotlib.pyplot as plt
    plotting = True
except ImportError:
    plotting = False


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


def main():
    print("Loading feature dataset...")
    df = load_features()

    
    params_path = os.path.join("data", "processed", "best_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find best_params.json at {params_path}. Please run Step 7 first.")
    with open(params_path) as f:
        best_params = json.load(f)
    print(f"Loaded best hyperparameters from {params_path}")

    print("Splitting train/test...")
    train, test = split_train_test(df)
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
    if "XGBoost" in best_params:
        from xgboost import XGBClassifier

        xgb_cfg = best_params["XGBoost"].copy()
        xgb_cfg.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})
        models.append(("XGBoost", XGBClassifier(**xgb_cfg)))
    else:
        print("No XGBoost parameters found; skipping XGBoost.")

    if "LightGBM" in best_params:
        from lightgbm import LGBMClassifier

        lgb_cfg = best_params["LightGBM"].copy()
        lgb_cfg.update({"random_state": 42})
        models.append(("LightGBM", LGBMClassifier(**lgb_cfg)))
    else:
        print("No LightGBM parameters found; skipping LightGBM.")

    for name, model in models:
        print(f"\nTraining and evaluating {name}...")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"Metrics for {name}:")
        print(f" - accuracy:    {accuracy_score(y_test, pred):.4f}")
        print(f" - roc_auc:     {roc_auc_score(y_test, proba):.4f}")
        print(f" - log_loss:    {log_loss(y_test, proba):.4f}")
        print(f" - brier_score: {brier_score_loss(y_test, proba):.4f}")

    
    print("\nComputing calibration curves (numeric)...")
    for name, model in models:
        proba = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        print(f"\n{name} calibration curve:")
        for m, f in zip(mean_pred, frac_pos):
            print(f" mean_pred: {m:.3f}, frac_pos: {f:.3f}")

    if plotting:
        os.makedirs("reports", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, model in models:
            CalibrationDisplay.from_estimator(
                model, X_test, y_test, n_bins=10, ax=ax, name=name
            )
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.set_title("Calibration curves")
        ax.legend()
        out_path = os.path.join("reports", "calibration_curve.png")
        fig.savefig(out_path)
        print(f"Saved calibration plot to {out_path}")


if __name__ == "__main__":
    main()