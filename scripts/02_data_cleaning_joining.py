"""
Step 2: Data cleaning & joining

 - Convert date columns to datetime
 - Merge player attributes into matches (winner & loser)
 - Output cleaned dataset for modeling
"""
import os
import sqlite3
import pandas as pd


def load_tables(db_path="database.sqlite"):
    conn = sqlite3.connect(db_path)
    matches = pd.read_sql("SELECT * FROM matches", conn)
    players = pd.read_sql("SELECT * FROM players", conn)
    rankings = pd.read_sql("SELECT * FROM rankings", conn)
    conn.close()
    return matches, players, rankings


def preprocess(matches, players, rankings):
    # Dates
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    players["dob"] = pd.to_datetime(players["dob"], format="%Y%m%d", errors="coerce")
    rankings["ranking_date"] = pd.to_datetime(rankings["ranking_date"], format="%Y%m%d", errors="coerce")

    # Merge player meta-data
    winner_meta = players.add_prefix("winner_")
    loser_meta = players.add_prefix("loser_")
    df = matches.merge(
        winner_meta, left_on="winner_id", right_on="winner_player_id", how="left"
    ).merge(
        loser_meta, left_on="loser_id", right_on="loser_player_id", how="left"
    )

    return df


def main():
    print("Loading tables...")
    matches, players, rankings = load_tables()
    print("Preprocessing and merging...")
    df = preprocess(matches, players, rankings)

    # Summary
    print(f"Cleaned dataset shape: {df.shape}")
    print("Null counts for ranking columns in matches:")
    for c in ["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]:
        print(f" - {c}: {df[c].isna().sum()}")

    # Save output
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matches_players.parquet")
    print(f"Saving cleaned data to {out_path}")
    try:
        df.to_parquet(out_path, index=False)
    except ImportError as e:
        csv_path = out_path.replace('.parquet', '.csv')
        print(f"Parquet engine missing ({e}); saving CSV instead to {csv_path}")
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()