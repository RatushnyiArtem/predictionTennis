#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# -----------------------------
# Helpers
# -----------------------------
def norm_str(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"&", "and", x)
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def split_score_to_goals(matches: pd.DataFrame, score_col: str) -> Tuple[pd.Series, pd.Series]:
    s = matches[score_col].astype(str).str.lower()
    m = s.str.extract(r"(\d+)\s*[-:]\s*(\d+)")
    hg = pd.to_numeric(m[0], errors="coerce").fillna(0).astype(int)
    ag = pd.to_numeric(m[1], errors="coerce").fillna(0).astype(int)
    return hg, ag


def coerce_goals(matches: pd.DataFrame, col: str) -> pd.Series:
    s = matches[col].astype(str)
    nums = s.str.extract(r"(\d+)")[0]
    return pd.to_numeric(nums, errors="coerce").fillna(0).astype(int)


def auto_detect_score_column(matches: pd.DataFrame) -> Optional[str]:
    best_col = None
    best_rate = 0.0
    sample = matches.head(5000) if len(matches) > 5000 else matches
    pattern = re.compile(r"\b\d+\s*[-:]\s*\d+\b")

    for col in matches.columns:
        if pd.api.types.is_numeric_dtype(matches[col]):
            continue
        s = sample[col].astype(str).str.lower()
        rate = s.apply(lambda x: 1 if pattern.search(x) else 0).mean()
        if rate > best_rate:
            best_rate = rate
            best_col = col

    return best_col if best_rate >= 0.05 else None


def find_col_by_keywords(df: pd.DataFrame, keywords: List[str], prefer_exact: bool = True) -> Optional[str]:
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    if prefer_exact:
        for kw in keywords:
            kwl = kw.lower()
            for c, cl in zip(cols, cols_lower):
                if cl == kwl:
                    return c

    for kw in keywords:
        kwl = kw.lower()
        for c, cl in zip(cols, cols_lower):
            if kwl in cl:
                return c

    return None


# -----------------------------
# Identify columns in matches
# -----------------------------
def detect_match_cols(matches: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {
        "competition": find_col_by_keywords(matches, ["competition_name", "competition", "tournament", "comp", "league"]),
        "season": find_col_by_keywords(matches, ["season", "year"]),
        "stage": find_col_by_keywords(matches, ["stage", "round", "matchday", "phase"]),
        "home_team": find_col_by_keywords(matches, ["home_team", "hometeam", "team_home", "home"]),
        "away_team": find_col_by_keywords(matches, ["away_team", "awayteam", "team_away", "away"]),
        "home_goals": find_col_by_keywords(matches, ["home_goals", "home_score", "homegoals", "score_home", "home_ft"]),
        "away_goals": find_col_by_keywords(matches, ["away_goals", "away_score", "awaygoals", "score_away", "away_ft"]),
        "score": find_col_by_keywords(matches, ["score", "ft_score", "full_time_score", "result", "final_score", "scoreline", "full_time", "fulltime"]),
        "date": find_col_by_keywords(matches, ["date", "match_date", "utc_date", "kickoff"]),
    }

    if mapping["score"] is None and (mapping["home_goals"] is None or mapping["away_goals"] is None):
        auto = auto_detect_score_column(matches)
        if auto is not None:
            mapping["score"] = auto

    return mapping


# -----------------------------
# League table from matches
# -----------------------------
def build_league_table(matches: pd.DataFrame, league_name: str, colmap: Dict[str, str]) -> pd.DataFrame:
    comp_col = colmap["competition"]
    home_col = colmap["home_team"]
    away_col = colmap["away_team"]

    df = matches.copy()
    df = df[df[comp_col].astype(str).str.lower().str.contains(str(league_name).lower(), na=False)].copy()
    if df.empty:
        return pd.DataFrame(columns=["team_norm", "place_in_league", "wins"])

    if colmap.get("home_goals") and colmap.get("away_goals"):
        df["_hg"] = coerce_goals(df, colmap["home_goals"])
        df["_ag"] = coerce_goals(df, colmap["away_goals"])
    elif colmap.get("score"):
        df["_hg"], df["_ag"] = split_score_to_goals(df, colmap["score"])
    else:
        raise ValueError("No goals columns found: need home_goals/away_goals OR a scoreline column like '2-1'.")

    df["_home_win"] = (df["_hg"] > df["_ag"]).astype(int)
    df["_away_win"] = (df["_ag"] > df["_hg"]).astype(int)
    df["_draw"] = (df["_hg"] == df["_ag"]).astype(int)

    home = df.groupby(home_col).agg(
        wins=("_home_win", "sum"),
        draws=("_draw", "sum"),
        gf=("_hg", "sum"),
        ga=("_ag", "sum"),
    ).reset_index().rename(columns={home_col: "team"})

    away = df.groupby(away_col).agg(
        wins=("_away_win", "sum"),
        draws=("_draw", "sum"),
        gf=("_ag", "sum"),
        ga=("_hg", "sum"),
    ).reset_index().rename(columns={away_col: "team"})

    table = pd.concat([home, away], ignore_index=True)
    table = table.groupby("team", as_index=False).sum(numeric_only=True)

    table["gd"] = table["gf"] - table["ga"]
    table["points"] = table["wins"] * 3 + table["draws"]

    table = table.sort_values(["wins", "points", "gd", "gf"], ascending=False).reset_index(drop=True)
    table["place_in_league"] = np.arange(1, len(table) + 1)
    table["team_norm"] = table["team"].map(norm_str)

    return table[["team_norm", "place_in_league", "wins"]].rename(columns={"wins": "wins_in_league"})


# -----------------------------
# UCL stage
# -----------------------------
UCL_STAGE_ORDER = ["LEAGUE_STAGE", "PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
UCL_STAGE_SCORE = {s: i / (len(UCL_STAGE_ORDER) - 1) for i, s in enumerate(UCL_STAGE_ORDER)}


def infer_ucl_stage(stage_text: str) -> str:
    s = norm_str(stage_text)
    if ("final" in s) and not any(k in s for k in ["semi", "quarter", "last 16", "round of 16", "r16"]):
        return "FINAL"
    if any(k in s for k in ["semi", "semifinal", "semi final"]):
        return "SEMI_FINALS"
    if any(k in s for k in ["quarter", "quarterfinal", "quarter final", "qf", "1 4"]):
        return "QUARTER_FINALS"
    if any(k in s for k in ["last 16", "round of 16", "r16", "1 8", "eighth"]):
        return "LAST_16"
    if any(k in s for k in ["playoff", "play offs", "play-offs", "knockout round play offs", "ko playoff", "ko play off"]):
        return "PLAYOFFS"
    if any(k in s for k in ["group", "league", "matchday", "md"]):
        return "LEAGUE_STAGE"
    return "LEAGUE_STAGE"


def build_ucl_stage_per_team(matches: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    comp_col = colmap["competition"]
    stage_col = colmap.get("stage")
    home_col = colmap["home_team"]
    away_col = colmap["away_team"]

    df = matches.copy()
    df = df[df[comp_col].astype(str).str.lower().str.contains("champions league", na=False)].copy()
    if df.empty:
        return pd.DataFrame(columns=["team_norm", "ucl_stage", "ucl_stage_score"])

    if stage_col is None:
        stage_col = "__stage_fallback__"
        df[stage_col] = "league stage"

    df["_stage_label"] = df[stage_col].astype(str).map(infer_ucl_stage)

    stage_to_idx = {s: i for i, s in enumerate(UCL_STAGE_ORDER)}
    df["_stage_idx"] = df["_stage_label"].map(stage_to_idx).fillna(0).astype(int)

    home = df.groupby(home_col)["_stage_idx"].max().reset_index().rename(columns={home_col: "team"})
    away = df.groupby(away_col)["_stage_idx"].max().reset_index().rename(columns={away_col: "team"})
    mx = pd.concat([home, away], ignore_index=True).groupby("team", as_index=False)["_stage_idx"].max()

    mx["ucl_stage"] = mx["_stage_idx"].map(lambda i: UCL_STAGE_ORDER[int(i)])
    mx["ucl_stage_score"] = mx["ucl_stage"].map(lambda s: UCL_STAGE_SCORE.get(s, 0.0))
    mx["team_norm"] = mx["team"].map(norm_str)
    return mx[["team_norm", "ucl_stage", "ucl_stage_score"]]


# -----------------------------
# Players detection
# -----------------------------
def add_player_name(players: pd.DataFrame) -> pd.DataFrame:
    name_col = find_col_by_keywords(players, ["player", "name", "player_name", "full_name", "playername"], prefer_exact=False)
    players = players.copy()
    if name_col is None:
        players["player"] = np.arange(len(players))
    else:
        players["player"] = players[name_col].astype(str)
    return players


def add_team_norm(players: pd.DataFrame) -> pd.DataFrame:
    team_col = find_col_by_keywords(players, [
        "team", "club", "squad",
        "team_name", "club_name", "squad_name",
        "current_team", "current_club",
    ], prefer_exact=False)

    if team_col is None:
        print("=== DEBUG: players columns ===")
        print(players.columns.tolist())
        raise ValueError("Can't find team/club column in players CSV. Rename it to team_name/club_name or tell me the column name.")

    players = players.copy()
    players["team"] = players[team_col].astype(str)
    players["team_norm"] = players["team"].map(norm_str)
    return players


def add_league_col(players: pd.DataFrame) -> pd.DataFrame:
    league_col = find_col_by_keywords(players, ["league", "competition", "domestic_league", "country_league", "league_name"], prefer_exact=False)
    players = players.copy()
    players["league"] = players[league_col].astype(str) if league_col else ""
    players["league_norm"] = players["league"].map(norm_str)
    return players


# -----------------------------
# Merge: FIXED (no merge collisions)
# -----------------------------
def merge_league_place(players: pd.DataFrame, matches: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    players = players.copy()

    # ensure columns exist
    if "place_in_league" not in players.columns:
        players["place_in_league"] = np.nan
    if "wins_in_league" not in players.columns:
        players["wins_in_league"] = np.nan

    leagues_in_players = sorted(set([l for l in players["league"].dropna().unique() if str(l).strip() != ""]))
    if not leagues_in_players:
        comp_col = colmap["competition"]
        comps = matches[comp_col].dropna().astype(str).unique().tolist()
        leagues_in_players = sorted({c for c in comps if "champions league" not in c.lower()})

    has_league_info = players["league"].astype(str).str.strip().ne("").any()

    for league_name in leagues_in_players:
        try:
            table = build_league_table(matches, league_name, colmap)
        except Exception:
            continue
        if table.empty:
            continue

        # build mapping dicts (no merge -> no _x/_y)
        place_map = dict(zip(table["team_norm"], table["place_in_league"]))
        wins_map = dict(zip(table["team_norm"], table["wins_in_league"]))

        if has_league_info:
            mask = players["league"].astype(str).str.lower().str.contains(str(league_name).lower(), na=False)
            if mask.sum() == 0:
                continue
        else:
            mask = pd.Series(True, index=players.index)

        players.loc[mask, "place_in_league"] = players.loc[mask, "team_norm"].map(place_map).values
        players.loc[mask, "wins_in_league"] = players.loc[mask, "team_norm"].map(wins_map).values

    return players


def merge_ucl_stage(players: pd.DataFrame, matches: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ucl = build_ucl_stage_per_team(matches, colmap)
    players = players.copy()
    players = players.merge(ucl, on="team_norm", how="left")
    players["ucl_stage"] = players["ucl_stage"].fillna("LEAGUE_STAGE")
    players["ucl_stage_score"] = players["ucl_stage_score"].fillna(0.0)
    return players


# -----------------------------
# Scoring
# -----------------------------
def compute_league_place_score(place: pd.Series) -> pd.Series:
    place_num = pd.to_numeric(place, errors="coerce")
    max_place = np.nanmax(place_num.values) if np.isfinite(place_num).any() else 20
    max_place = max(2, int(max_place))
    score = 1.0 - (place_num - 1) / (max_place - 1)
    return score.clip(0, 1).fillna(0.0)


def compute_rule_based_final_score(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    out += 0.20 * compute_league_place_score(df.get("place_in_league", pd.Series(index=df.index)))
    out += 0.20 * pd.to_numeric(df.get("ucl_stage_score", 0.0), errors="coerce").fillna(0.0).clip(0, 1)

    for col, w in [
        ("goals", 0.20),
        ("assists", 0.10),
        ("g+a", 0.25),
        ("minutes", 0.05),
        ("rating", 0.20),
        ("trophies", 0.15),
        ("motm", 0.05),
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            denom = np.nanpercentile(s.values, 95) if np.isfinite(s).any() else 1.0
            denom = max(1e-9, float(denom))
            out += w * (s / denom).clip(0, 1)

    return out.clip(0, 1)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", default="ballon_dor_predictions_2024_2025.csv")
    args = ap.parse_args()

    players = pd.read_csv(args.players)
    matches = pd.read_csv(args.matches)

    colmap = detect_match_cols(matches)

    base_missing = [k for k in ["competition", "home_team", "away_team"] if colmap.get(k) is None]
    goals_ok = (colmap.get("home_goals") is not None and colmap.get("away_goals") is not None) or (colmap.get("score") is not None)

    if base_missing or not goals_ok:
        print("=== DEBUG: matches columns ===")
        print(matches.columns.tolist())
        raise ValueError(
            f"Matches CSV missing required columns. base_missing={base_missing}, goals_ok={goals_ok}. "
            f"Detected mapping: {colmap}"
        )

    print("Detected match columns:", colmap)

    players = add_player_name(players)
    players = add_team_norm(players)
    players = add_league_col(players)

    players = merge_league_place(players, matches, colmap)
    players = merge_ucl_stage(players, matches, colmap)

    players["league_place_score"] = compute_league_place_score(players["place_in_league"])
    players["ucl_stage_score"] = pd.to_numeric(players["ucl_stage_score"], errors="coerce").fillna(0.0).clip(0, 1)
    players["final_score"] = compute_rule_based_final_score(players)

    out_cols = [
        "player", "team", "league",
        "place_in_league", "wins_in_league",
        "ucl_stage", "league_place_score", "ucl_stage_score",
        "final_score",
    ]
    out_cols = [c for c in out_cols if c in players.columns]

    res = players.sort_values("final_score", ascending=False).reset_index(drop=True)
    res.to_csv(args.out, index=False)

    print("\nSaved:", args.out)
    print("\nTOP 10 Ballon d'Or prediction:")
    print(res[out_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
