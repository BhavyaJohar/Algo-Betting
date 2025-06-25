"""
Step 4: Feature engineering (ELO ratings, surface win-rates, head-to-head & difference features)

- Compute rolling ELO ratings for each player (overall and per surface).
- Calculate per-surface win-rate (prior to match), head-to-head stats (prior to match).
- Calculate feature differences: age, height, rank, rank points.
"""
import os
import pandas as pd
import numpy as np


def load_framed(path="data/processed/matches_framed.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["tourney_date"])
    raise FileNotFoundError(f"No framed data at {path} or {csv_path}")


def compute_elo(df, K=32):
    df = df.sort_values("tourney_date").reset_index(drop=True)
    init_rating = 1500
    elo_overall = {}
    elo_surface = {}
    pre1, pre2 = [], []
    pre1_s, pre2_s = [], []

    for _, row in df.iterrows():
        p1, p2 = row.player1_id, row.player2_id
        surf = row.surface
        r1 = elo_overall.get(p1, init_rating)
        r2 = elo_overall.get(p2, init_rating)
        rs1 = elo_surface.get((surf, p1), init_rating)
        rs2 = elo_surface.get((surf, p2), init_rating)

        pre1.append(r1)
        pre2.append(r2)
        pre1_s.append(rs1)
        pre2_s.append(rs2)

        s1 = row.target
        s2 = 1 - s1

        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        es1 = 1 / (1 + 10 ** ((rs2 - rs1) / 400))
        es2 = 1 / (1 + 10 ** ((rs1 - rs2) / 400))

        elo_overall[p1] = r1 + K * (s1 - e1)
        elo_overall[p2] = r2 + K * (s2 - e2)
        elo_surface[(surf, p1)] = rs1 + K * (s1 - es1)
        elo_surface[(surf, p2)] = rs2 + K * (s2 - es2)

    df["player1_elo"] = pre1
    df["player2_elo"] = pre2
    df["player1_elo_surface"] = pre1_s
    df["player2_elo_surface"] = pre2_s
    df["elo_diff"] = df.player1_elo - df.player2_elo
    df["elo_surf_diff"] = df.player1_elo_surface - df.player2_elo_surface
    return df


def add_differences(df):
    df["age_diff"] = df.player1_age - df.player2_age
    df["ht_diff"] = df.player1_ht - df.player2_ht
    df["rank_diff"] = df.player1_rank - df.player2_rank
    df["rank_points_diff"] = df.player1_rank_points - df.player2_rank_points
    return df


def compute_surface_winrate(df):
    df = df.sort_values("tourney_date").reset_index(drop=True)
    surf_wins = {}
    surf_totals = {}
    rate1, rate2 = [], []
    for _, row in df.iterrows():
        p1, p2 = row.player1_id, row.player2_id
        surf = row.surface
        w1 = surf_wins.get((surf, p1), 0)
        t1 = surf_totals.get((surf, p1), 0)
        w2 = surf_wins.get((surf, p2), 0)
        t2 = surf_totals.get((surf, p2), 0)
        rate1.append(w1 / t1 if t1 > 0 else np.nan)
        rate2.append(w2 / t2 if t2 > 0 else np.nan)
        surf_totals[(surf, p1)] = t1 + 1
        surf_totals[(surf, p2)] = t2 + 1
        if row.target == 1:
            surf_wins[(surf, p1)] = w1 + 1
        else:
            surf_wins[(surf, p2)] = w2 + 1
    df["player1_surf_win_rate"] = rate1
    df["player2_surf_win_rate"] = rate2
    df["surf_win_rate_diff"] = df.player1_surf_win_rate - df.player2_surf_win_rate
    return df


def compute_head2head(df):
    df = df.sort_values("tourney_date").reset_index(drop=True)
    hh_wins = {}
    hh_totals = {}
    count_list, rate1, rate2 = [], [], []
    for _, row in df.iterrows():
        p1, p2 = row.player1_id, row.player2_id
        key = (p1, p2)
        w = hh_wins.get(key, 0)
        t = hh_totals.get(key, 0)
        count_list.append(t)
        if t > 0:
            rate1.append(w / t)
            rate2.append((t - w) / t)
        else:
            rate1.append(np.nan)
            rate2.append(np.nan)
        hh_totals[key] = t + 1
        if row.target == 1:
            hh_wins[key] = w + 1
    df["h2h_count"] = count_list
    df["player1_h2h_win_rate"] = rate1
    df["player2_h2h_win_rate"] = rate2
    return df


def main():
    print("Loading framed matches...")
    df = load_framed()

    print("Computing ELO ratings...")
    df = compute_elo(df)

    print("Computing surface win-rate features...")
    df = compute_surface_winrate(df)

    print("Computing head-to-head features...")
    df = compute_head2head(df)

    print("Adding difference features...")
    df = add_differences(df)

    print(f"Features added: resulting shape {df.shape}")

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matches_features.parquet")
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        csv_path = out_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved features to {csv_path}")
    else:
        print(f"Saved features to {out_path}")


if __name__ == "__main__":
    main()