"""
Step 10: Deployment & model export

- Load engineered feature dataset and best hyperparameters
- Train final models on the full dataset
- Serialize model artifacts for serving
"""
import os
import json

import pandas as pd
import pickle

try:
    import joblib
    _USE_JOBLIB = True
except ImportError:
    joblib = None
    _USE_JOBLIB = False


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


FEATURE_COLUMNS = [
    "elo_diff", "elo_surf_diff",
    "age_diff", "ht_diff",
    "rank_diff", "rank_points_diff",
    "surf_win_rate_diff",
    "player1_h2h_win_rate", "h2h_count",
]


def load_features(path="data/processed/matches_features.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace('.parquet', '.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=['tourney_date'])
    raise FileNotFoundError(f"No feature dataset at {path} or {csv_path}")


def load_best_params(path='data/processed/best_params.json'):
    with open(path) as f:
        return json.load(f)


def train_final_model(df, params, model_name):
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df['target']
    if model_name == 'LightGBM' and LGBMClassifier is not None:
        model = LGBMClassifier(**params)
    elif model_name == 'XGBoost' and XGBClassifier is not None:
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f'Model {model_name} is not available in this environment')
    model.fit(X, y)
    return model


def main():
    print('Loading features...')
    df = load_features()
    print('Loading best hyperparameters...')
    best_params = load_best_params()

    os.makedirs('data/processed', exist_ok=True)

    # Train and save each available model
    for model_name, params in best_params.items():
        print(f'Training final {model_name}...')
        try:
            model = train_final_model(df, params, model_name)
        except ValueError as e:
            print(f'Skipping {model_name}: {e}')
            continue
        out_path = f'data/processed/{model_name.lower()}_final.joblib'
        if _USE_JOBLIB:
            joblib.dump(model, out_path)
        else:
            with open(out_path, 'wb') as f:
                pickle.dump(model, f)
        print(f'Saved {model_name} to {out_path}')


if __name__ == '__main__':
    main()