"""
Step 3: Labeling & framing matches for modeling

- Rename match-stat columns (w_*, l_*) to winner_*, loser_*
- Define a deterministic ordering: player1 has smaller player ID
- Create binary target: 1 if player1 won, else 0
- Pivot winner_/loser_ features into player1_/player2_ feature groups
"""
import os
import pandas as pd
import numpy as np

def load_cleaned(path="data/processed/matches_players.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No cleaned data at {path} or {csv_path}")

def frame_matches(df):
    stat_map = {c: "winner_" + c[2:] for c in df.columns if c.startswith("w_")}
    stat_map.update({c: "loser_" + c[2:] for c in df.columns if c.startswith("l_")})
    if stat_map:
        df = df.rename(columns=stat_map)

    df["player1_id"] = df[["winner_id", "loser_id"]].min(axis=1)
    df["player2_id"] = df[["winner_id", "loser_id"]].max(axis=1)
    df["target"] = (df["player1_id"] == df["winner_id"]).astype(int)

    winner_cols = [c for c in df.columns if c.startswith("winner_")]
    bases = [c[len("winner_"):] for c in winner_cols]

    data = {
        "player1_id": df["player1_id"],
        "player2_id": df["player2_id"],
        "target": df["target"],
    }
    for base in bases:
        wcol = "winner_" + base
        lcol = "loser_" + base
        data["player1_" + base] = np.where(df["target"] == 1, df[wcol], df[lcol])
        data["player2_" + base] = np.where(df["target"] == 1, df[lcol], df[wcol])

    match_cols = [
        c
        for c in df.columns
        if not any(c.startswith(pref) for pref in ("winner_", "loser_", "player1_", "player2_"))
    ]
    match_cols = [c for c in match_cols if c not in ("winner_id", "loser_id")]
    for c in match_cols:
        data[c] = df[c]

    return pd.DataFrame(data)

def main():
    print("Loading cleaned match-player dataset...")
    df = load_cleaned()
    print(f"Loaded {len(df):,} matches with {df.shape[1]} columns.")
    print("Framing player1/player2 ordering and target...")
    df_framed = frame_matches(df)
    print(f"Resulting framed dataset: {len(df_framed):,} rows, {df_framed.shape[1]} columns.")
    print(df_framed[["player1_id", "player2_id", "target"]].head())

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matches_framed.parquet")
    try:
        df_framed.to_parquet(out_path, index=False)
    except Exception:
        csv_path = out_path.replace(".parquet", ".csv")
        df_framed.to_csv(csv_path, index=False)
        print(f"Saved framed dataset to {csv_path}")
    else:
        print(f"Saved framed dataset to {out_path}")

if __name__ == "__main__":
    main()