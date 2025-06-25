#!/usr/bin/env python3
"""
Simulate Roland Garros from semifinals onward.

Reads the tournament draw from `roland.csv`, uses provided Round of 4 matchups
to predict semifinal winners and then the final using a trained LightGBM model.
Outputs predictions to `roland_predictions.csv`.
"""
import sqlite3
from pathlib import Path
import pickle
import joblib

import numpy as np
import pandas as pd
import unicodedata


def load_players(db_path):
    conn = sqlite3.connect(db_path)
    players = pd.read_sql(
        "SELECT player_id, name_first, name_last, dob, ioc, height FROM players", conn
    )
    conn.close()
    players["dob"] = pd.to_datetime(
        players["dob"].astype(str), format="%Y%m%d", errors="coerce"
    )
    players["full_name"] = players["name_first"].str.strip() + " " + players[
        "name_last"
    ].str.strip()
    return players.set_index("player_id"), players.set_index("full_name")


def load_rankings(db_path, tournament_date):
    conn = sqlite3.connect(db_path)
    ranks = pd.read_sql(
        "SELECT player AS player_id, ranking_date, points, rank FROM rankings", conn
    )
    conn.close()
    ranks["ranking_date"] = pd.to_datetime(
        ranks["ranking_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    recent = ranks[ranks["ranking_date"] <= tournament_date]
    recent = recent.sort_values("ranking_date").groupby("player_id").last()
    return recent[["rank", "points"]]


def load_historical(path, tournament_date):
    df = pd.read_csv(path, parse_dates=["tourney_date"])
    return df[df["tourney_date"] < tournament_date].reset_index(drop=True)


def compute_elo_dict(df, K=32):
    init = 1500
    elo_overall = {}
    elo_surface = {}
    for _, row in df.sort_values("tourney_date").iterrows():
        p1, p2 = int(row.player1_id), int(row.player2_id)
        surf = row.surface
        r1 = elo_overall.get(p1, init)
        r2 = elo_overall.get(p2, init)
        rs1 = elo_surface.get((surf, p1), init)
        rs2 = elo_surface.get((surf, p2), init)
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
    return elo_overall, elo_surface


def compute_surf_stats(df):
    surf_totals = {}
    surf_wins = {}
    for _, row in df.sort_values("tourney_date").iterrows():
        p1, p2 = int(row.player1_id), int(row.player2_id)
        surf = row.surface
        t1 = surf_totals.get((surf, p1), 0)
        t2 = surf_totals.get((surf, p2), 0)
        surf_totals[(surf, p1)] = t1 + 1
        surf_totals[(surf, p2)] = t2 + 1
        if row.target == 1:
            surf_wins[(surf, p1)] = surf_wins.get((surf, p1), 0) + 1
        else:
            surf_wins[(surf, p2)] = surf_wins.get((surf, p2), 0) + 1
    return surf_totals, surf_wins


def compute_h2h_stats(df):
    h2h_totals = {}
    h2h_wins = {}
    for _, row in df.sort_values("tourney_date").iterrows():
        p1, p2 = int(row.player1_id), int(row.player2_id)
        key = (p1, p2)
        h2h_totals[key] = h2h_totals.get(key, 0) + 1
        if row.target == 1:
            h2h_wins[key] = h2h_wins.get(key, 0) + 1
    return h2h_totals, h2h_wins


def make_features(df, elo_overall, elo_surface, surf_totals, surf_wins,
                  h2h_totals, h2h_wins, players_by_id, rankings):
    feats = []
    for _, row in df.iterrows():
        p1, p2 = int(row.player1_id), int(row.player2_id)
        rt = row.tourney_date
        surf = row.surface
        # Elo
        e1 = elo_overall.get(p1, 1500)
        e2 = elo_overall.get(p2, 1500)
        es1 = elo_surface.get((surf, p1), 1500)
        es2 = elo_surface.get((surf, p2), 1500)
        # Age
        dob1 = players_by_id.at[p1, "dob"]
        dob2 = players_by_id.at[p2, "dob"]
        age1 = (rt - dob1).days / 365.25 if pd.notna(dob1) else np.nan
        age2 = (rt - dob2).days / 365.25 if pd.notna(dob2) else np.nan
        # Height
        ht1 = players_by_id.at[p1, "height"]
        ht2 = players_by_id.at[p2, "height"]
        # Rank
        rk1 = rankings["rank"].get(p1, np.nan)
        rk2 = rankings["rank"].get(p2, np.nan)
        rp1 = rankings["points"].get(p1, np.nan)
        rp2 = rankings["points"].get(p2, np.nan)
        # Surface win-rate
        t1 = surf_totals.get((surf, p1), 0)
        t2 = surf_totals.get((surf, p2), 0)
        w1 = surf_wins.get((surf, p1), 0)
        w2 = surf_wins.get((surf, p2), 0)
        sr1 = w1 / t1 if t1 > 0 else np.nan
        sr2 = w2 / t2 if t2 > 0 else np.nan
        # Head-to-head
        tt = h2h_totals.get((p1, p2), 0)
        w = h2h_wins.get((p1, p2), 0)
        hr = w / tt if tt > 0 else np.nan
        feats.append({
            "elo_diff": e1 - e2,
            "elo_surf_diff": es1 - es2,
            "age_diff": age1 - age2,
            "ht_diff": ht1 - ht2,
            "rank_diff": rk1 - rk2,
            "rank_points_diff": rp1 - rp2,
            "surf_win_rate_diff": sr1 - sr2,
            "h2h_count": tt,
            "player1_h2h_win_rate": hr,
        })
    return pd.DataFrame(feats)


def predict_matches(df, model, feature_cols):
    X = df[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["proba"] = proba
    out["predicted_winner_id"] = np.where(
        proba >= 0.5, out["player1_id"], out["player2_id"]
    )
    return out


def reorder_players(df):
    swap = df["player1_id"] > df["player2_id"]
    for c1, c2 in [
        ("player1_id", "player2_id"),
        ("player1_name", "player2_name"),
        ("player1_country", "player2_country"),
        ("player1_seed", "player2_seed"),
        ("player1_entry", "player2_entry"),
    ]:
        df.loc[swap, [c1, c2]] = df.loc[swap, [c2, c1]].values
    return df


def main():
    base = Path(__file__).resolve().parent.parent
    draw_path = base / "roland.csv"
    out_path = base / "roland_predictions.csv"
    draw = pd.read_csv(
        draw_path, parse_dates=["tourney_date"], date_format="%Y%m%d"
    )
    tournament_date = draw["tourney_date"].iat[0]
    db = base / "database.sqlite"
    players_by_id, players_by_name = load_players(db)
    # if draw lacks ID columns, build player1_id/2 from names; otherwise assume IDs are present
    if "player1_id" not in draw.columns or "player2_id" not in draw.columns:
        raw_to_id = players_by_name["player_id"].to_dict()
        def _normalize_name(s: str) -> str:
            if not isinstance(s, str):
                return ""
            s2 = unicodedata.normalize("NFKD", s)
            # replace various dash characters before stripping non-ascii
            s2 = (s2
                .replace("\u2010", "-").replace("\u2011", "-")
                .replace("\u2012", "-").replace("\u2013", "-")
                .replace("\u2014", "-")
            )
            s2 = s2.encode("ascii", errors="ignore").decode("ascii")
            return s2.lower()
        norm_to_id = { _normalize_name(n): pid for n, pid in raw_to_id.items() if isinstance(n, str) }
        draw["player1_id"] = draw["player1_name"].apply(
            lambda n: raw_to_id.get(n) or norm_to_id.get(_normalize_name(n))
        )
        draw["player2_id"] = draw["player2_name"].apply(
            lambda n: raw_to_id.get(n) or norm_to_id.get(_normalize_name(n))
        )
        # drop any Round-of-4 (R8) rows whose player name failed to map to an ID
        semis_mask = draw["round"] == "R8"
        missing_mask = semis_mask & (
            draw["player1_id"].isnull() | draw["player2_id"].isnull()
        )
        if missing_mask.any():
            missing = pd.concat([
                draw.loc[draw["player1_id"].isnull() & semis_mask, "player1_name"],
                draw.loc[draw["player2_id"].isnull() & semis_mask, "player2_name"],
            ]).unique()
            print(f"Warning: missing player name↔id mapping for semis {list(missing)}, skipping these rows")
            draw = draw.loc[~missing_mask].reset_index(drop=True)
    # else: assume draw already contains player1_id and player2_id
    draw = reorder_players(draw)
    hist = load_historical(
        base / "data/processed/matches_framed.csv", tournament_date
    )
    elo_overall, elo_surf = compute_elo_dict(hist)
    surf_totals, surf_wins = compute_surf_stats(hist)
    h2h_totals, h2h_wins = compute_h2h_stats(hist)
    rankings = load_rankings(db, tournament_date)
    model_path = base / "data/processed/lightgbm_final.joblib"
    if not model_path.exists():
        model_path = model_path.with_suffix(".pkl")
    # load LightGBM model (joblib or pickle)
    model = joblib.load(model_path)
    feature_cols = [
        "elo_diff",
        "elo_surf_diff",
        "age_diff",
        "ht_diff",
        "rank_diff",
        "rank_points_diff",
        "surf_win_rate_diff",
        "player1_h2h_win_rate",
        "h2h_count",
    ]
    # Semifinals: use existing Round of 4 matchups from draw (R8 entries now represent semis)
    sf = draw[draw["round"] == "R8"].reset_index(drop=True)
    sf["round"] = "R4"
    sf["tourney_date"] = tournament_date
    sf["surface"] = draw["surface"].iat[0]
    # retain seed and entry info from draw; names and countries are already present
    feats_sf = make_features(
        sf,
        elo_overall,
        elo_surf,
        surf_totals,
        surf_wins,
        h2h_totals,
        h2h_wins,
        players_by_id,
        rankings,
    )
    sf_pred = predict_matches(
        pd.concat([sf, feats_sf], axis=1), model, feature_cols
    )
    # Final
    winners_sf = sf_pred.reset_index(drop=True)
    final = pd.DataFrame([
        {
            "player1_id": winners_sf.at[0, "predicted_winner_id"],
            "player2_id": winners_sf.at[1, "predicted_winner_id"],
        }
    ])
    final["round"] = "R2"
    final["tourney_date"] = tournament_date
    final["surface"] = draw["surface"].iat[0]
    for side in ["player1", "player2"]:
        final[f"{side}_name"] = final[f"{side}_id"].map(players_by_id["full_name"])
        final[f"{side}_country"] = final[f"{side}_id"].map(players_by_id["ioc"])
        final[f"{side}_seed"] = None
        final[f"{side}_entry"] = None
    feats_final = make_features(
        final,
        elo_overall,
        elo_surf,
        surf_totals,
        surf_wins,
        h2h_totals,
        h2h_wins,
        players_by_id,
        rankings,
    )
    final_pred = predict_matches(
        pd.concat([final, feats_final], axis=1), model, feature_cols
    )
    # Save
    out = pd.concat([sf_pred, final_pred], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()