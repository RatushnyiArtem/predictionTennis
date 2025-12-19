# main.py
# Wimbledon 2025 prediction + step-by-step bracket + STRONG surface factor:
# ✅ Overall Elo (all surfaces) + Grass Elo (grass-only)
# ✅ Uses BOTH as model features: diff_overall_elo_pre, diff_grass_elo_pre
# ✅ Fixes MergeError by using globally-unique match_key for Elo merges
#
# Folder layout:
#   degree project/
#     main.py
#     wimbledon_2025_draw_r1.csv     (64 rows, columns: player1, player2)
#     tennis_atp-master/            (all your csv files)
#
# Run:
#   /usr/bin/python3 main.py

from __future__ import annotations

import glob
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("tennis_atp-master")
    draw_csv: Path = Path("wimbledon_2025_draw_r1.csv")

    # Training focus
    train_surface: str = "Grass"
    tourney_levels: Tuple[str, ...] = ("G", "M", "A")  # Slam, Masters, ATP

    # Extra datasets (recommended OFF for Wimbledon singles)
    use_amateur: bool = False
    use_futures: bool = False
    use_qual_chall: bool = False
    use_doubles: bool = False

    # Performance
    max_rows: Optional[int] = None  # set if RAM limited, e.g. 2_000_000
    n_sims: int = 5000              # increase to 30000 for final stability
    seed: int = 42                  # reproducibility

    # Elo params
    elo_init: float = 1500.0
    elo_k_overall: float = 24.0
    elo_k_grass: float = 28.0

    # Bracket printing
    print_step_by_step: bool = True
    step_mode: str = "deterministic"  # "deterministic" or "stochastic"


CFG = Config()


# =========================
# FILE DISCOVERY
# =========================
def _list_files(base: Path, pattern: str) -> List[str]:
    return sorted(glob.glob(str(base / pattern)))


def get_match_files(cfg: Config) -> List[str]:
    files: List[str] = []
    files += _list_files(cfg.data_dir, "atp_matches_[0-9][0-9][0-9][0-9].csv")

    if cfg.use_amateur and (cfg.data_dir / "atp_matches_amateur.csv").exists():
        files += [str(cfg.data_dir / "atp_matches_amateur.csv")]

    if cfg.use_futures:
        files += _list_files(cfg.data_dir, "atp_matches_futures_[0-9][0-9][0-9][0-9].csv")

    if cfg.use_qual_chall:
        files += _list_files(cfg.data_dir, "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv")

    if cfg.use_doubles:
        files += _list_files(cfg.data_dir, "atp_matches_doubles_[0-9][0-9][0-9][0-9].csv")

    return files


def get_ranking_files(cfg: Config) -> List[str]:
    return sorted(glob.glob(str(cfg.data_dir / "atp_rankings_*.csv")))


# =========================
# LOADERS
# =========================
def load_matches(files: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "tourney_date" in df.columns:
            df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), errors="coerce")
        dfs.append(df)

    m = pd.concat(dfs, ignore_index=True)
    m = m.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    m["winner_id"] = m["winner_id"].astype(int)
    m["loser_id"] = m["loser_id"].astype(int)
    m["surface"] = m.get("surface", "Unknown").fillna("Unknown").astype(str)

    # ensure keys exist
    if "tourney_id" not in m.columns:
        m["tourney_id"] = ""
    if "match_num" not in m.columns:
        m["match_num"] = np.arange(len(m))

    m["tourney_id"] = m["tourney_id"].fillna("").astype(str)
    m["match_num"] = pd.to_numeric(m["match_num"], errors="coerce").fillna(-1).astype(int)

    # Global unique match key across the whole dataset
    m["match_key"] = (
        m["tourney_id"].astype(str) + "|" +
        m["tourney_date"].dt.strftime("%Y%m%d").astype(str) + "|" +
        m["match_num"].astype(str)
    )

    m = m.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    if max_rows:
        m = m.tail(max_rows).reset_index(drop=True)
    return m


def load_rankings(files: List[str]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["ranking_date"] = pd.to_datetime(df["ranking_date"].astype(str), errors="coerce")
        dfs.append(df)

    r = pd.concat(dfs, ignore_index=True)
    r = r.dropna(subset=["ranking_date", "player", "rank", "points"])
    r["player"] = r["player"].astype(int)
    r["rank"] = r["rank"].astype(int)
    r["points"] = r["points"].astype(int)
    return r


def load_players(cfg: Config) -> pd.DataFrame:
    p = pd.read_csv(cfg.data_dir / "atp_players.csv", low_memory=False)
    p["player_id"] = p["player_id"].astype(int)
    p["name_first"] = p["name_first"].fillna("")
    p["name_last"] = p["name_last"].fillna("")
    p["full_name"] = (p["name_first"] + " " + p["name_last"]).str.strip()
    return p


# =========================
# ELO (overall + grass)
# =========================
def _elo_expect(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


def build_elo_rows(
    matches: pd.DataFrame,
    elo_init: float = 1500.0,
    k_overall: float = 24.0,
    k_grass: float = 28.0,
) -> pd.DataFrame:
    """
    Iterates match-by-match and outputs per-player rows with:
      - overall_elo_pre
      - grass_elo_pre
    Uses globally unique match_key (already in matches):
      match_key = f"{tourney_id}|{tourney_date}|{match_num}"
    """
    m = matches[["match_key", "surface", "winner_id", "loser_id", "tourney_date"]].copy()
    m = m.sort_values(["tourney_date", "match_key"]).reset_index(drop=True)

    overall: Dict[int, float] = {}
    grass: Dict[int, float] = {}

    out_rows = []

    for _, r in m.iterrows():
        w = int(r["winner_id"])
        l = int(r["loser_id"])
        surf = str(r["surface"]).lower()
        mk = str(r["match_key"])

        ow = float(overall.get(w, elo_init))
        ol = float(overall.get(l, elo_init))
        gw = float(grass.get(w, elo_init))
        gl = float(grass.get(l, elo_init))

        # record pre-match
        out_rows.append({"match_key": mk, "player_id": w, "overall_elo_pre": ow, "grass_elo_pre": gw})
        out_rows.append({"match_key": mk, "player_id": l, "overall_elo_pre": ol, "grass_elo_pre": gl})

        # update overall
        pw = _elo_expect(ow, ol)
        overall[w] = ow + k_overall * (1.0 - pw)
        overall[l] = ol + k_overall * (0.0 - (1.0 - pw))

        # update grass only on grass
        if surf == "grass":
            pgw = _elo_expect(gw, gl)
            grass[w] = gw + k_grass * (1.0 - pgw)
            grass[l] = gl + k_grass * (0.0 - (1.0 - pgw))

    elo_rows = pd.DataFrame(out_rows)

    # Defensive: ensure uniqueness for merge
    elo_rows = elo_rows.drop_duplicates(subset=["match_key", "player_id"], keep="first")

    elo_rows["overall_elo_pre"] = elo_rows["overall_elo_pre"].astype(np.float32)
    elo_rows["grass_elo_pre"] = elo_rows["grass_elo_pre"].astype(np.float32)
    return elo_rows


# =========================
# FEATURE ENGINEERING
# =========================
def build_player_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Builds per-player rows (winner + loser) and keeps match_key for Elo merge.
    """
    m = matches.copy()

    optional = [
        "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
        "winner_age", "loser_age",
        "winner_ht", "loser_ht",
        "winner_hand", "loser_hand",
        "w_ace", "l_ace",
        "w_df", "l_df",
        "tourney_level",
    ]
    for c in optional:
        if c not in m.columns:
            m[c] = np.nan

    # match_key already computed in load_matches()
    if "match_key" not in m.columns:
        m["tourney_id"] = m.get("tourney_id", "").fillna("").astype(str)
        m["match_num"] = pd.to_numeric(m.get("match_num", -1), errors="coerce").fillna(-1).astype(int)
        m["match_key"] = (
            m["tourney_id"].astype(str) + "|" +
            m["tourney_date"].dt.strftime("%Y%m%d").astype(str) + "|" +
            m["match_num"].astype(str)
        )

    w = pd.DataFrame({
        "tourney_date": m["tourney_date"],
        "match_key": m["match_key"],
        "surface": m["surface"].astype(str),
        "player_id": m["winner_id"].astype(int),
        "opponent_id": m["loser_id"].astype(int),
        "is_win": 1,
        "rank": pd.to_numeric(m["winner_rank"], errors="coerce"),
        "points": pd.to_numeric(m["winner_rank_points"], errors="coerce"),
        "age": pd.to_numeric(m["winner_age"], errors="coerce"),
        "ht": pd.to_numeric(m["winner_ht"], errors="coerce"),
        "hand": m["winner_hand"],
        "aces": pd.to_numeric(m["w_ace"], errors="coerce"),
        "dfs": pd.to_numeric(m["w_df"], errors="coerce"),
    })

    l = pd.DataFrame({
        "tourney_date": m["tourney_date"],
        "match_key": m["match_key"],
        "surface": m["surface"].astype(str),
        "player_id": m["loser_id"].astype(int),
        "opponent_id": m["winner_id"].astype(int),
        "is_win": 0,
        "rank": pd.to_numeric(m["loser_rank"], errors="coerce"),
        "points": pd.to_numeric(m["loser_rank_points"], errors="coerce"),
        "age": pd.to_numeric(m["loser_age"], errors="coerce"),
        "ht": pd.to_numeric(m["loser_ht"], errors="coerce"),
        "hand": m["loser_hand"],
        "aces": pd.to_numeric(m["l_ace"], errors="coerce"),
        "dfs": pd.to_numeric(m["l_df"], errors="coerce"),
    })

    pm = pd.concat([w, l], ignore_index=True)
    pm = pm.sort_values(["tourney_date", "match_key", "player_id"]).reset_index(drop=True)
    return pm


def add_rolling_features(pm: pd.DataFrame) -> pd.DataFrame:
    def roll_mean(s: pd.Series, window: int) -> pd.Series:
        return s.shift(1).rolling(window=window, min_periods=5).mean()

    pm = pm.sort_values(["player_id", "tourney_date"]).copy()
    pm["is_grass"] = (pm["surface"].str.lower() == "grass").astype(int)

    pm["winrate_10"] = pm.groupby("player_id")["is_win"].transform(lambda s: roll_mean(s, 10))
    pm["winrate_25"] = pm.groupby("player_id")["is_win"].transform(lambda s: roll_mean(s, 25))

    gwin = pm["is_win"].where(pm["is_grass"] == 1)
    pm["grass_winrate_10"] = gwin.groupby(pm["player_id"]).transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean()
    )
    pm["grass_winrate_25"] = gwin.groupby(pm["player_id"]).transform(
        lambda s: s.shift(1).rolling(25, min_periods=5).mean()
    )

    pm["aces_10"] = pm.groupby("player_id")["aces"].transform(lambda s: roll_mean(s, 10))
    pm["dfs_10"] = pm.groupby("player_id")["dfs"].transform(lambda s: roll_mean(s, 10))

    for c in ["winrate_10", "winrate_25", "grass_winrate_10", "grass_winrate_25", "aces_10", "dfs_10"]:
        pm[c] = pm[c].fillna(pm[c].mean())

    return pm


def merge_elo_into_pm(pm: pd.DataFrame, elo_rows: pd.DataFrame, elo_init: float = 1500.0) -> pd.DataFrame:
    """
    Merge Elo pre values into pm on (match_key, player_id).
    This avoids pandas MergeError from non-unique tourney_id/match_num combos.
    """
    merged = pm.merge(
        elo_rows,
        on=["match_key", "player_id"],
        how="left"
    )
    merged["overall_elo_pre"] = merged["overall_elo_pre"].fillna(elo_init).astype(np.float32)
    merged["grass_elo_pre"] = merged["grass_elo_pre"].fillna(elo_init).astype(np.float32)
    return merged


def build_training_set(matches: pd.DataFrame, pm: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build pairwise rows (A vs B) and label y=1 if A won.
    Includes strong features:
      - diff_overall_elo_pre
      - diff_grass_elo_pre
    """
    key = pm[[
        "tourney_date", "player_id", "opponent_id", "surface",
        "winrate_10", "winrate_25",
        "grass_winrate_10", "grass_winrate_25",
        "aces_10", "dfs_10",
        "rank", "points", "age", "ht", "hand",
        "overall_elo_pre", "grass_elo_pre",
    ]].copy()

    base = matches[["tourney_date", "surface", "winner_id", "loser_id"]].copy()
    base["surface"] = base["surface"].astype(str)

    w = pd.DataFrame({
        "tourney_date": base["tourney_date"],
        "surface": base["surface"],
        "A_id": base["winner_id"].astype(int),
        "B_id": base["loser_id"].astype(int),
        "y": 1
    })
    l = pd.DataFrame({
        "tourney_date": base["tourney_date"],
        "surface": base["surface"],
        "A_id": base["loser_id"].astype(int),
        "B_id": base["winner_id"].astype(int),
        "y": 0
    })

    cols_side = [
        "winrate_10", "winrate_25",
        "grass_winrate_10", "grass_winrate_25",
        "aces_10", "dfs_10",
        "rank", "points", "age", "ht", "hand",
        "overall_elo_pre", "grass_elo_pre",
    ]

    def merge_sides(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(
            key,
            left_on=["tourney_date", "A_id", "B_id"],
            right_on=["tourney_date", "player_id", "opponent_id"],
            how="left"
        )
        for c in cols_side:
            df.rename(columns={c: f"A_{c}"}, inplace=True)
        df.drop(columns=["player_id", "opponent_id"], inplace=True, errors="ignore")

        df = df.merge(
            key,
            left_on=["tourney_date", "B_id", "A_id"],
            right_on=["tourney_date", "player_id", "opponent_id"],
            how="left"
        )
        for c in cols_side:
            df.rename(columns={c: f"B_{c}"}, inplace=True)
        df.drop(columns=["player_id", "opponent_id"], inplace=True, errors="ignore")
        return df

    data = pd.concat([merge_sides(w), merge_sides(l)], ignore_index=True)

    hand_map = {"R": 1.0, "L": -1.0}
    data["A_hand"] = data["A_hand"].map(hand_map).fillna(0.0)
    data["B_hand"] = data["B_hand"].map(hand_map).fillna(0.0)

    bases = [
        "winrate_10", "winrate_25",
        "grass_winrate_10", "grass_winrate_25",
        "aces_10", "dfs_10",
        "rank", "points", "age", "ht", "hand",
        "overall_elo_pre", "grass_elo_pre",
    ]
    for b in bases:
        data[f"diff_{b}"] = pd.to_numeric(data[f"A_{b}"], errors="coerce") - pd.to_numeric(data[f"B_{b}"], errors="coerce")

    data["diff_rank_inv"] = (1.0 / (pd.to_numeric(data["A_rank"], errors="coerce") + 1.0)) - (
        1.0 / (pd.to_numeric(data["B_rank"], errors="coerce") + 1.0)
    )
    data["is_grass_match"] = (data["surface"].str.lower() == "grass").astype(int)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean(numeric_only=True))

    feature_cols = [c for c in data.columns if c.startswith("diff_")] + ["diff_rank_inv", "is_grass_match"]

    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data["y"].to_numpy(dtype=np.int32)
    return X, y, feature_cols


# =========================
# TRAINING
# =========================
def train_models(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, object], Dict[str, float]]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]

    models: Dict[str, object] = {"rf": rf}
    metrics: Dict[str, float] = {
        "rf_auc": float(roc_auc_score(yte, p_rf)),
        "rf_logloss": float(log_loss(yte, p_rf)),
    }

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        )
        xgb.fit(np.asarray(Xtr, dtype=np.float32), np.asarray(ytr, dtype=np.int32))
        p_xgb = xgb.predict_proba(np.asarray(Xte, dtype=np.float32))[:, 1]

        models["xgb"] = xgb
        metrics["xgb_auc"] = float(roc_auc_score(yte, p_xgb))
        metrics["xgb_logloss"] = float(log_loss(yte, p_xgb))

    return models, metrics


def select_best_model(models: Dict[str, object], metrics: Dict[str, float]) -> Tuple[str, object]:
    if HAS_XGB and ("xgb_auc" in metrics) and metrics["xgb_auc"] >= metrics["rf_auc"]:
        return "xgb", models["xgb"]
    return "rf", models["rf"]


# =========================
# NAME MAPPING (ROBUST)
# =========================
def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    return s


def build_name_index(players_df: pd.DataFrame) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for _, r in players_df.iterrows():
        idx[normalize_name(r["full_name"])] = int(r["player_id"])
    return idx


def map_one_name(name: str, idx: Dict[str, int]) -> Optional[int]:
    n = normalize_name(name)
    if n in idx:
        return idx[n]

    parts = n.split()
    if not parts:
        return None

    last = parts[-1]
    candidates = [pid for full, pid in idx.items() if full.endswith(" " + last)]
    if len(candidates) == 1:
        return candidates[0]

    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        candidates = []
        for full, pid in idx.items():
            fp = full.split()
            if len(fp) >= 2 and fp[0] == first and fp[-1] == last:
                candidates.append(pid)
        if len(candidates) == 1:
            return candidates[0]

    return None


def map_draw_names_to_ids(draw_df: pd.DataFrame, players_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = build_name_index(players_df)
    mapped = draw_df.copy()
    mapped["player1_id"] = mapped["player1"].apply(lambda x: map_one_name(x, idx))
    mapped["player2_id"] = mapped["player2"].apply(lambda x: map_one_name(x, idx))
    missing = mapped[mapped["player1_id"].isna() | mapped["player2_id"].isna()]
    return mapped, missing


# =========================
# PREDICTION FEATURES FOR 2025
# =========================
def latest_ranking_snapshot(rankings: pd.DataFrame) -> Tuple[pd.Timestamp, pd.DataFrame]:
    d = rankings["ranking_date"].max()
    snap = rankings[rankings["ranking_date"] == d].sort_values("rank").copy()
    return d, snap


def build_latest_player_features(pm: pd.DataFrame, snap: pd.DataFrame, elo_init: float = 1500.0) -> Dict[int, Dict[str, float | str]]:
    last = pm.sort_values("tourney_date").groupby("player_id").tail(1)
    last = last[[
        "player_id",
        "winrate_10", "winrate_25",
        "grass_winrate_10", "grass_winrate_25",
        "aces_10", "dfs_10",
        "age", "ht", "hand",
        "overall_elo_pre", "grass_elo_pre",
    ]].copy()

    snap2 = snap[["player", "rank", "points"]].rename(columns={"player": "player_id"})
    merged = last.merge(snap2, on="player_id", how="left")
    merged["rank"] = merged["rank"].fillna(9999)
    merged["points"] = merged["points"].fillna(0)

    merged["overall_elo_pre"] = merged["overall_elo_pre"].fillna(elo_init)
    merged["grass_elo_pre"] = merged["grass_elo_pre"].fillna(elo_init)

    feats: Dict[int, Dict[str, float | str]] = {}
    for _, r in merged.iterrows():
        pid = int(r["player_id"])
        feats[pid] = {
            "winrate_10": float(r["winrate_10"]),
            "winrate_25": float(r["winrate_25"]),
            "grass_winrate_10": float(r["grass_winrate_10"]),
            "grass_winrate_25": float(r["grass_winrate_25"]),
            "aces_10": float(r["aces_10"]),
            "dfs_10": float(r["dfs_10"]),
            "age": float(r["age"]) if not np.isnan(r["age"]) else 28.0,
            "ht": float(r["ht"]) if not np.isnan(r["ht"]) else 185.0,
            "hand": (r["hand"] if isinstance(r["hand"], str) else "R"),
            "rank": float(r["rank"]),
            "points": float(r["points"]),
            "overall_elo_pre": float(r["overall_elo_pre"]),
            "grass_elo_pre": float(r["grass_elo_pre"]),
        }
    return feats


def make_feature_row_numpy(
    fa: Dict[str, float | str],
    fb: Dict[str, float | str],
    feature_cols: List[str]
) -> np.ndarray:
    hand_map = {"R": 1.0, "L": -1.0}
    A_hand = hand_map.get(str(fa.get("hand", "R")), 0.0)
    B_hand = hand_map.get(str(fb.get("hand", "R")), 0.0)

    bases = [
        "winrate_10", "winrate_25",
        "grass_winrate_10", "grass_winrate_25",
        "aces_10", "dfs_10",
        "rank", "points", "age", "ht", "hand",
        "overall_elo_pre", "grass_elo_pre",
    ]

    row: Dict[str, float] = {}
    for b in bases:
        av = float(fa.get(b, 0.0)) if b != "hand" else float(A_hand)
        bv = float(fb.get(b, 0.0)) if b != "hand" else float(B_hand)
        row[f"diff_{b}"] = av - bv

    ra = float(fa.get("rank", 9999.0))
    rb = float(fb.get("rank", 9999.0))
    row["diff_rank_inv"] = (1.0 / (ra + 1.0)) - (1.0 / (rb + 1.0))
    row["is_grass_match"] = 1.0

    Xrow = pd.DataFrame([row])[feature_cols].astype(np.float32)
    return Xrow.to_numpy(dtype=np.float32)


# =========================
# SIMULATION (FAST with cache)
# =========================
def simulate_bracket(
    draw_pairs: List[Tuple[int, int]],
    player_feats: Dict[int, Dict[str, float | str]],
    model: object,
    feature_cols: List[str],
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    bracket: List[int] = []
    for a, b in draw_pairs:
        bracket.extend([a, b])

    rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "W"]
    reach = {pid: {r: 0 for r in rounds} for pid in set(bracket)}

    prob_cache: Dict[Tuple[int, int], float] = {}

    def pwin(a: int, b: int) -> float:
        key = (a, b)
        if key in prob_cache:
            return prob_cache[key]

        fa = player_feats.get(a)
        fb = player_feats.get(b)

        if fa is None or fb is None:
            ra = float(fa["rank"]) if fa else 9999.0
            rb = float(fb["rank"]) if fb else 9999.0
            p = float(1.0 / (1.0 + np.exp((ra - rb) / 50.0)))
            prob_cache[key] = p
            prob_cache[(b, a)] = 1.0 - p
            return p

        Xrow = make_feature_row_numpy(fa, fb, feature_cols)
        p = float(model.predict_proba(np.asarray(Xrow, dtype=np.float32))[:, 1][0])
        prob_cache[key] = p
        prob_cache[(b, a)] = 1.0 - p
        return p

    step = max(1, n_sims // 20)

    for sim in range(1, n_sims + 1):
        cur = bracket.copy()
        for pid in cur:
            reach[pid]["R128"] += 1

        for rnd in ["R64", "R32", "R16", "QF", "SF", "F", "W"]:
            nxt = []
            for i in range(0, len(cur), 2):
                a, b = cur[i], cur[i + 1]
                p = pwin(a, b)
                w = a if rng.random() < p else b
                nxt.append(w)
                reach[w][rnd] += 1
            cur = nxt

        if sim == 1 or sim == n_sims or (sim % step == 0):
            print(f"Simulation progress: {sim}/{n_sims} ({sim / n_sims:.0%}) | cache={len(prob_cache)//2} matchups")

    rows = []
    for pid in reach.keys():
        rows.append({
            "player_id": pid,
            "P_win": reach[pid]["W"] / n_sims,
            "P_final": reach[pid]["F"] / n_sims,
            "P_SF": reach[pid]["SF"] / n_sims,
            "P_QF": reach[pid]["QF"] / n_sims,
            "P_R16": reach[pid]["R16"] / n_sims,
        })

    return pd.DataFrame(rows).sort_values("P_win", ascending=False).reset_index(drop=True)


# =========================
# STEP-BY-STEP BRACKET PRINT
# =========================
def print_step_by_step_bracket(
    draw_pairs: List[Tuple[int, int]],
    player_feats: Dict[int, Dict[str, float | str]],
    model: object,
    feature_cols: List[str],
    id_to_name: Dict[int, str],
    seed: int = 42,
    mode: str = "deterministic",
) -> int:
    rng = np.random.default_rng(seed)

    bracket: List[int] = []
    for a, b in draw_pairs:
        bracket.extend([a, b])

    prob_cache: Dict[Tuple[int, int], float] = {}

    def pwin(a: int, b: int) -> float:
        key = (a, b)
        if key in prob_cache:
            return prob_cache[key]

        fa = player_feats.get(a)
        fb = player_feats.get(b)

        if fa is None or fb is None:
            ra = float(fa["rank"]) if fa else 9999.0
            rb = float(fb["rank"]) if fb else 9999.0
            p = float(1.0 / (1.0 + np.exp((ra - rb) / 50.0)))
            prob_cache[key] = p
            prob_cache[(b, a)] = 1.0 - p
            return p

        Xrow = make_feature_row_numpy(fa, fb, feature_cols)
        p = float(model.predict_proba(np.asarray(Xrow, dtype=np.float32))[:, 1][0])
        prob_cache[key] = p
        prob_cache[(b, a)] = 1.0 - p
        return p

    def name(pid: int) -> str:
        return id_to_name.get(pid, str(pid))

    def pick_winner(a: int, b: int) -> Tuple[int, float]:
        p = pwin(a, b)
        if mode == "stochastic":
            w = a if rng.random() < p else b
        else:
            w = a if p >= 0.5 else b
        pw = p if w == a else (1.0 - p)
        return w, pw

    round_labels = [("R128", 64), ("R64", 32), ("R32", 16), ("R16", 8), ("QF", 4), ("SF", 2), ("F", 1)]
    cur = bracket

    for rnd, _ in round_labels:
        print("\n" + "=" * 70)
        print(f"{rnd} (matches: {len(cur)//2})")
        print("=" * 70)

        nxt: List[int] = []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i + 1]
            w, pw = pick_winner(a, b)
            print(f"{name(a)} vs {name(b)}  ->  WIN: {name(w)}  (P_win={pw:.3f})")
            nxt.append(w)

        cur = nxt

    champion = cur[0]
    print("\n" + "#" * 70)
    print(f"CHAMPION: {name(champion)}")
    print("#" * 70 + "\n")
    return champion


# =========================
# MAIN
# =========================
def main(cfg: Config) -> None:
    match_files = get_match_files(cfg)
    ranking_files = get_ranking_files(cfg)

    print("Found match files:", len(match_files))
    print("Found ranking files:", len(ranking_files))

    matches_all = load_matches(match_files, max_rows=cfg.max_rows)
    rankings = load_rankings(ranking_files)
    players = load_players(cfg)

    # Training matches: grass + top levels
    train_matches = matches_all[matches_all["surface"].str.lower() == cfg.train_surface.lower()].copy()
    if "tourney_level" in train_matches.columns:
        train_matches = train_matches[train_matches["tourney_level"].isin(cfg.tourney_levels)].copy()

    print("Training matches (filtered):", len(train_matches))

    # Elo from ALL matches => strong overall skill + grass skill
    elo_rows_all = build_elo_rows(
        matches_all,
        elo_init=cfg.elo_init,
        k_overall=cfg.elo_k_overall,
        k_grass=cfg.elo_k_grass,
    )

    # Player-match rows for training matches
    pm = build_player_match_table(train_matches)
    pm = add_rolling_features(pm)
    pm = merge_elo_into_pm(pm, elo_rows_all, elo_init=cfg.elo_init)

    # Training set
    X, y, feature_cols = build_training_set(train_matches, pm)
    print("Training rows:", X.shape[0], "features:", X.shape[1])

    models, metrics = train_models(X, y)
    print("Metrics:", metrics)

    best_name, model = select_best_model(models, metrics)
    print("Best model:", best_name)

    # Latest rankings snapshot
    rank_date = rankings["ranking_date"].max()
    snap = rankings[rankings["ranking_date"] == rank_date].sort_values("rank").copy()
    print("Latest ranking date:", rank_date.date())

    player_feats = build_latest_player_features(pm, snap, elo_init=cfg.elo_init)

    # Load draw
    if not cfg.draw_csv.exists():
        raise FileNotFoundError(f"Draw CSV not found: {cfg.draw_csv.resolve()}")

    draw = pd.read_csv(cfg.draw_csv)
    if not {"player1", "player2"}.issubset(draw.columns):
        raise ValueError("Draw CSV must have columns: player1, player2")

    mapped, missing = map_draw_names_to_ids(draw, players)
    if len(missing) > 0:
        print("\nERROR: Could not map some draw names to ATP IDs.")
        print(missing[["player1", "player2", "player1_id", "player2_id"]].head(80).to_string(index=False))
        raise SystemExit(1)

    draw_pairs = list(zip(mapped["player1_id"].astype(int), mapped["player2_id"].astype(int)))
    if len(draw_pairs) != 64:
        print(f"\nWARNING: draw has {len(draw_pairs)} matches; Wimbledon needs 64.\n")

    name_map = dict(zip(players["player_id"], players["full_name"]))

    # Step-by-step bracket
    if cfg.print_step_by_step:
        print("\n\nSTEP-BY-STEP BRACKET (one run):")
        print_step_by_step_bracket(
            draw_pairs=draw_pairs,
            player_feats=player_feats,
            model=model,
            feature_cols=feature_cols,
            id_to_name=name_map,
            seed=cfg.seed,
            mode=cfg.step_mode,
        )

    # Monte Carlo simulation
    results = simulate_bracket(
        draw_pairs=draw_pairs,
        player_feats=player_feats,
        model=model,
        feature_cols=feature_cols,
        n_sims=cfg.n_sims,
        seed=cfg.seed,
    )

    results["player_name"] = results["player_id"].map(name_map).fillna(results["player_id"].astype(str))

    print("\nTOP 10 Wimbledon 2025 (Monte Carlo):")
    print(results[["player_name", "P_win", "P_final", "P_SF", "P_QF", "P_R16"]].head(10).to_string(index=False))

    out_path = Path("wimbledon_2025_prediction_results.csv")
    results.to_csv(out_path, index=False)
    print("\nSaved:", out_path.resolve())


if __name__ == "__main__":
    main(CFG)
