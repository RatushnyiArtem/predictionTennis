from __future__ import annotations

import argparse
import glob
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================
#          CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("tennis_atp-master")
    draw_csv: Path = Path("wimbledon_2025_draw_r1.csv")
    actual_results_csv: Path = Path("wimbledon_2025_all_rounds_results.csv")
    prediction_cutoff_date: str = "2025-06-30"

    # surface for calculating
    train_surface: str = "Grass"

    # extra datasets
    use_amateur: bool = False
    use_futures: bool = False
    use_qual_chall: bool = True
    use_doubles: bool = False

    # performance
    max_rows: Optional[int] = None
    n_sims: int = 1
    seed: int = 42

    # elo params
    elo_init: float = 1500.0
    elo_k_overall: float = 24.0
    elo_k_grass: float = 28.0

    # output / simulation mode
    step_mode: str = "deterministic"  # deterministic or stochastic

    # default composite weights
    w_ml: float = 0.82
    w_streak: float = 0.12
    w_h2h: float = 0.06

    # weight search settings
    run_weight_search: bool = True
    weight_step_pct: int = 5
    out_prefix: str = "wimbledon"


CFG = Config()


# =========================
# UTILS
# =========================
def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("'", "")
    s = s.replace("’", "")
    s = s.replace("`", "")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_name_to_id(players: pd.DataFrame) -> Dict[str, int]:
    alias_to_ids: Dict[str, set[int]] = {}
    preferred_full_names: Dict[str, int] = {}

    for _, row in players.iterrows():
        pid = int(row["player_id"])
        first = str(row.get("name_first", "") or "").strip()
        last = str(row.get("name_last", "") or "").strip()
        full = str(row.get("full_name", "") or "").strip()

        strong_aliases = {
            full,
            f"{first} {last}".strip(),
            f"{last} {first}".strip(),
        }
        weak_aliases = {
            first,
            last,
        }

        for cand in strong_aliases | weak_aliases:
            norm = normalize_name(cand)
            if norm:
                alias_to_ids.setdefault(norm, set()).add(pid)

        full_norm = normalize_name(full)
        if full_norm:
            preferred_full_names[full_norm] = pid

    mapping: Dict[str, int] = {}
    for alias, ids in alias_to_ids.items():
        if len(ids) == 1:
            mapping[alias] = next(iter(ids))

    mapping.update(preferred_full_names)
    return mapping


def safe_prob(x: float) -> float:
    return float(np.clip(x, 1e-6, 1 - 1e-6))


def validate_weights(w_ml: float, w_streak: float, w_h2h: float) -> Tuple[float, float, float]:
    total = w_ml + w_streak + w_h2h
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return w_ml / total, w_streak / total, w_h2h / total


def parse_cutoff_date(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid cutoff date: {value}")
    return ts.normalize()


RESULT_OVERRIDES = {
    (
        "1/32",
        frozenset({normalize_name("Daniel Evans"), normalize_name("Novak Djokovic")}),
    ): {
        "winner": "Novak Djokovic",
        "score": "6-3 6-2 6-0",
    },
}


def load_and_clean_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv).copy()

    for idx, row in df.iterrows():
        key = (
            str(row.get("round", "")).strip(),
            frozenset({normalize_name(row.get("player1", "")), normalize_name(row.get("player2", ""))}),
        )
        override = RESULT_OVERRIDES.get(key)
        if override:
            for col, val in override.items():
                df.at[idx, col] = val

    return filter_consistent_bracket_results(df)


def filter_consistent_bracket_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "round" not in results_df.columns:
        return results_df.copy()

    round_order = {"1/64": 1, "1/32": 2, "1/16": 3, "1/8": 4, "1/4": 5, "1/2": 6, "Final": 7}
    df = results_df.copy()
    df["_order"] = df["round"].map(round_order)
    df = df.sort_values(["_order"]).reset_index(drop=True)

    cleaned_rows = []
    alive: set[str] | None = None

    for round_name in [r for r, _ in sorted(round_order.items(), key=lambda x: x[1])]:
        rd = df[df["round"] == round_name].copy()
        if rd.empty:
            continue

        if alive is None:
            current_alive = set()
            for _, row in rd.iterrows():
                current_alive.add(normalize_name(row.get("player1", "")))
                current_alive.add(normalize_name(row.get("player2", "")))
            alive = current_alive

        next_alive: set[str] = set()
        used_this_round: set[str] = set()

        for _, row in rd.iterrows():
            p1 = normalize_name(row.get("player1", ""))
            p2 = normalize_name(row.get("player2", ""))
            winner = normalize_name(row.get("winner", ""))

            if not p1 or not p2 or p1 == p2:
                continue
            if p1 not in alive or p2 not in alive:
                continue
            if winner not in {p1, p2}:
                continue
            if p1 in used_this_round or p2 in used_this_round:
                continue

            cleaned_rows.append(row.drop(labels=["_order"]).to_dict())
            used_this_round.add(p1)
            used_this_round.add(p2)
            next_alive.add(winner)

        alive = next_alive

    return pd.DataFrame(cleaned_rows)


# =========================
# FILE DISCOVERY & LOADERS
# =========================
def _list_files(base: Path, pattern: str) -> List[str]:
    return sorted(glob.glob(str(base / pattern)))


def get_match_files(cfg: Config) -> List[str]:
    files: List[str] = []
    files += _list_files(cfg.data_dir, "atp_matches_[0-9][0-9][0-9][0-9].csv")
    if cfg.use_amateur:
        files += _list_files(cfg.data_dir, "atp_matches_amateur.csv")
    if cfg.use_futures:
        files += _list_files(cfg.data_dir, "atp_matches_futures_[0-9][0-9][0-9][0-9].csv")
    if cfg.use_qual_chall:
        files += _list_files(cfg.data_dir, "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv")
    return files


def load_matches(files: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "tourney_date" in df.columns:
                df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        raise ValueError("No match files found or loaded.")

    m = pd.concat(dfs, ignore_index=True)
    m = m.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    m["winner_id"] = m["winner_id"].astype(int)
    m["loser_id"] = m["loser_id"].astype(int)
    m["surface"] = m.get("surface", "Unknown").fillna("Unknown").astype(str)

    if "tourney_id" not in m.columns:
        m["tourney_id"] = ""
    if "match_num" not in m.columns:
        m["match_num"] = -1

    m["match_key"] = (
        m["tourney_id"].astype(str) + "|" +
        m["tourney_date"].dt.strftime("%Y%m%d").astype(str) + "|" +
        m["match_num"].astype(str)
    )

    m = m.sort_values(["tourney_date", "tourney_id", "match_num"]).reset_index(drop=True)
    if max_rows:
        m = m.tail(max_rows).reset_index(drop=True)
    return m


def load_players(cfg: Config) -> pd.DataFrame:
    p = pd.read_csv(cfg.data_dir / "atp_players.csv", low_memory=False)
    p["player_id"] = p["player_id"].astype(int)
    p["full_name"] = (p["name_first"].fillna("") + " " + p["name_last"].fillna("")).str.strip()
    return p


def load_rankings(cfg: Config) -> pd.DataFrame:
    files = sorted(glob.glob(str(cfg.data_dir / "atp_rankings_current.csv")))
    if not files:
        files = sorted(glob.glob(str(cfg.data_dir / "atp_rankings_*.csv")))

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["ranking_date"] = pd.to_datetime(df["ranking_date"].astype(str), format="%Y%m%d", errors="coerce")
        dfs.append(df)

    r = pd.concat(dfs, ignore_index=True)
    r = r.dropna(subset=["ranking_date", "player", "rank"])
    return r


# =========================
# ELO CALCULATION
# =========================
def _elo_expect(ea: float, eb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((eb - ea) / 400.0))


def build_elo_rows(matches: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    m = matches[["match_key", "surface", "winner_id", "loser_id", "tourney_date"]].copy()
    m = m.sort_values(["tourney_date", "match_key"]).reset_index(drop=True)

    overall: Dict[int, float] = {}
    grass: Dict[int, float] = {}

    out_rows = []

    for _, r in m.iterrows():
        w, l = int(r["winner_id"]), int(r["loser_id"])
        surf = str(r["surface"]).lower()
        mk = str(r["match_key"])

        ow = overall.get(w, cfg.elo_init)
        ol = overall.get(l, cfg.elo_init)
        gw = grass.get(w, cfg.elo_init)
        gl = grass.get(l, cfg.elo_init)

        out_rows.append({"match_key": mk, "player_id": w, "overall_elo_pre": ow, "grass_elo_pre": gw})
        out_rows.append({"match_key": mk, "player_id": l, "overall_elo_pre": ol, "grass_elo_pre": gl})

        pw = _elo_expect(ow, ol)
        overall[w] = ow + cfg.elo_k_overall * (1.0 - pw)
        overall[l] = ol + cfg.elo_k_overall * (0.0 - (1.0 - pw))

        if surf == "grass":
            pgw = _elo_expect(gw, gl)
            grass[w] = gw + cfg.elo_k_grass * (1.0 - pgw)
            grass[l] = gl + cfg.elo_k_grass * (0.0 - (1.0 - pgw))

    elo_rows = pd.DataFrame(out_rows)
    elo_rows = elo_rows.drop_duplicates(subset=["match_key", "player_id"], keep="first")
    return elo_rows


# =========================
# FEATURES
# =========================
def build_player_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()

    for col in [
        "winner_rank", "loser_rank", "w_ace", "l_ace", "w_df", "l_df",
        "winner_age", "loser_age", "winner_ht", "loser_ht", "winner_hand", "loser_hand"
    ]:
        if col not in m.columns:
            m[col] = np.nan

    w = pd.DataFrame({
        "tourney_date": m["tourney_date"],
        "match_key": m["match_key"],
        "surface": m["surface"].astype(str),
        "player_id": m["winner_id"].astype(int),
        "opponent_id": m["loser_id"].astype(int),
        "is_win": 1,
        "rank": pd.to_numeric(m["winner_rank"], errors="coerce"),
        "aces": pd.to_numeric(m["w_ace"], errors="coerce"),
        "dfs": pd.to_numeric(m["w_df"], errors="coerce"),
        "age": pd.to_numeric(m["winner_age"], errors="coerce"),
        "ht": pd.to_numeric(m["winner_ht"], errors="coerce"),
        "hand": m["winner_hand"],
    })

    l = pd.DataFrame({
        "tourney_date": m["tourney_date"],
        "match_key": m["match_key"],
        "surface": m["surface"].astype(str),
        "player_id": m["loser_id"].astype(int),
        "opponent_id": m["winner_id"].astype(int),
        "is_win": 0,
        "rank": pd.to_numeric(m["loser_rank"], errors="coerce"),
        "aces": pd.to_numeric(m["l_ace"], errors="coerce"),
        "dfs": pd.to_numeric(m["l_df"], errors="coerce"),
        "age": pd.to_numeric(m["loser_age"], errors="coerce"),
        "ht": pd.to_numeric(m["loser_ht"], errors="coerce"),
        "hand": m["loser_hand"],
    })

    pm = pd.concat([w, l], ignore_index=True)
    pm = pm.sort_values(["player_id", "tourney_date"]).reset_index(drop=True)
    return pm


def add_rolling_features(pm: pd.DataFrame) -> pd.DataFrame:
    def roll_mean(s, w):
        return s.shift(1).rolling(w, min_periods=1).mean()

    pm["winrate_10"] = pm.groupby("player_id")["is_win"].transform(lambda s: roll_mean(s, 10))
    pm["aces_10"] = pm.groupby("player_id")["aces"].transform(lambda s: roll_mean(s, 10))
    pm["dfs_10"] = pm.groupby("player_id")["dfs"].transform(lambda s: roll_mean(s, 10))

    is_grass = (pm["surface"].str.lower() == "grass")
    pm["grass_win"] = pm["is_win"].where(is_grass)
    pm["grass_winrate_25"] = pm.groupby("player_id")["grass_win"].transform(lambda s: s.shift(1).rolling(25, min_periods=1).mean())

    pm = pm.fillna(0)
    return pm


def build_training_set(matches: pd.DataFrame, pm: pd.DataFrame, elo_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    pm_elo = pm.merge(elo_rows, on=["match_key", "player_id"], how="left")
    pm_elo["overall_elo_pre"] = pm_elo["overall_elo_pre"].fillna(1500)
    pm_elo["grass_elo_pre"] = pm_elo["grass_elo_pre"].fillna(1500)

    cols_to_keep = [
        "tourney_date", "player_id", "opponent_id",
        "winrate_10", "grass_winrate_25", "aces_10", "dfs_10",
        "rank", "age", "ht", "hand", "overall_elo_pre", "grass_elo_pre"
    ]
    key = pm_elo[cols_to_keep].copy()

    base = matches[["tourney_date", "winner_id", "loser_id", "surface"]].copy()

    w_rows = base.rename(columns={"winner_id": "A_id", "loser_id": "B_id"})
    w_rows["y"] = 1
    l_rows = base.rename(columns={"loser_id": "A_id", "winner_id": "B_id"})
    l_rows["y"] = 0

    data = pd.concat([w_rows, l_rows], ignore_index=True)

    data = data.merge(key, left_on=["tourney_date", "A_id", "B_id"], right_on=["tourney_date", "player_id", "opponent_id"], how="left")
    data = data.drop(columns=["player_id", "opponent_id"])
    data = data.rename(columns={c: f"A_{c}" for c in cols_to_keep if c not in ["tourney_date"]})

    data = data.merge(key, left_on=["tourney_date", "B_id", "A_id"], right_on=["tourney_date", "player_id", "opponent_id"], how="left")
    data = data.drop(columns=["player_id", "opponent_id"])
    data = data.rename(columns={c: f"B_{c}" for c in cols_to_keep if c not in ["tourney_date"]})

    features = [
        "winrate_10", "grass_winrate_25", "aces_10", "dfs_10",
        "rank", "age", "ht", "overall_elo_pre", "grass_elo_pre"
    ]

    for f in features:
        data[f"diff_{f}"] = data[f"A_{f}"] - data[f"B_{f}"]

    data["A_hand_code"] = data["A_hand"].map({"R": 1, "L": -1}).fillna(0)
    data["B_hand_code"] = data["B_hand"].map({"R": 1, "L": -1}).fillna(0)

    data["is_grass"] = (data["surface"].str.lower() == "grass").astype(int)

    feature_cols = [f"diff_{f}" for f in features] + ["A_hand_code", "B_hand_code", "is_grass"]

    data = data.dropna(subset=feature_cols)
    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data["y"].to_numpy(dtype=np.int32)

    return X, y, feature_cols


# =========================
# MODEL TRAINING
# =========================
def train_model(X: np.ndarray, y: np.ndarray):
    print(f"Training on {len(X)} samples...")
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)

    model.fit(X, y)
    return model


# =========================
# 2025 PREP & ENHANCED LOGIC
# =========================
def get_current_streak(pm_df: pd.DataFrame, player_id: int) -> int:
    hist = pm_df[pm_df["player_id"] == player_id].sort_values("tourney_date", ascending=False)
    if hist.empty:
        return 0

    streak = 0
    first_res = hist.iloc[0]["is_win"]
    direction = 1 if first_res == 1 else -1

    for _, row in hist.iterrows():
        if (row["is_win"] == 1 and direction == 1) or (row["is_win"] == 0 and direction == -1):
            streak += direction
        else:
            break
    return streak


def build_h2h_map(matches_df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[int, int]]:
    h2h: Dict[Tuple[int, int], List[int]] = {}
    grouped = matches_df.groupby(["winner_id", "loser_id"]).size().reset_index(name="counts")

    for _, row in grouped.iterrows():
        w, l, c = int(row["winner_id"]), int(row["loser_id"]), int(row["counts"])
        if (w, l) not in h2h:
            h2h[(w, l)] = [0, 0]
        h2h[(w, l)][0] += c
        if (l, w) not in h2h:
            h2h[(l, w)] = [0, 0]
        h2h[(l, w)][1] += c

    return {k: tuple(v) for k, v in h2h.items()}


def prepare_2025_features(
    pm: pd.DataFrame,
    elo_rows: pd.DataFrame,
    rankings: pd.DataFrame,
    players: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> Dict[int, Dict]:
    cutoff = parse_cutoff_date(as_of_date)

    last_elos = elo_rows.sort_values("match_key").groupby("player_id").tail(1).set_index("player_id")
    last_stats = pm.sort_values("tourney_date").groupby("player_id").tail(1).set_index("player_id")

    rankings = rankings[rankings["ranking_date"] <= cutoff].copy()
    latest_date = rankings["ranking_date"].max() if not rankings.empty else pd.NaT
    if pd.isna(latest_date):
        ranks = pd.DataFrame(columns=rankings.columns).set_index(pd.Index([], name="player"))
    else:
        ranks = rankings[rankings["ranking_date"] == latest_date].sort_values("rank").drop_duplicates(subset=["player"]).set_index("player")

    player_feats = {}
    all_ids = set(pm["player_id"].unique()) | set(players["player_id"].unique())

    for pid in all_ids:
        stat_row = last_stats.loc[pid] if pid in last_stats.index else None
        elo_row = last_elos.loc[pid] if pid in last_elos.index else None
        rank_row = ranks.loc[pid] if pid in ranks.index else None
        player_row = players[players["player_id"] == pid]

        feats = {}
        feats["winrate_10"] = float(stat_row["winrate_10"]) if stat_row is not None else 0.5
        feats["grass_winrate_25"] = float(stat_row["grass_winrate_25"]) if stat_row is not None else 0.0
        feats["aces_10"] = float(stat_row["aces_10"]) if stat_row is not None else 0.0
        feats["dfs_10"] = float(stat_row["dfs_10"]) if stat_row is not None else 0.0

        feats["overall_elo_pre"] = float(elo_row["overall_elo_pre"]) if elo_row is not None else 1500.0
        feats["grass_elo_pre"] = float(elo_row["grass_elo_pre"]) if elo_row is not None else 1500.0

        feats["rank"] = float(rank_row["rank"]) if rank_row is not None else 999.0

        if not player_row.empty:
            p_data = player_row.iloc[0]
            feats["hand"] = p_data.get("hand", "R")
            dob = str(p_data.get("dob", ""))
            feats["age"] = 2025 - int(dob[:4]) if len(dob) >= 4 and dob[:4].isdigit() else 25
            feats["ht"] = float(p_data["height"]) if not pd.isna(p_data.get("height", np.nan)) else 185.0
        else:
            feats["hand"] = "R"
            feats["age"] = 25
            feats["ht"] = 185.0

        feats["streak"] = get_current_streak(pm, pid)
        player_feats[pid] = feats

    return player_feats


def make_inference_row(pA: Dict, pB: Dict, feature_cols: List[str]) -> np.ndarray:
    row = {}
    row["diff_winrate_10"] = pA["winrate_10"] - pB["winrate_10"]
    row["diff_grass_winrate_25"] = pA["grass_winrate_25"] - pB["grass_winrate_25"]
    row["diff_aces_10"] = pA["aces_10"] - pB["aces_10"]
    row["diff_dfs_10"] = pA["dfs_10"] - pB["dfs_10"]
    row["diff_rank"] = pA["rank"] - pB["rank"]
    row["diff_age"] = pA["age"] - pB["age"]
    row["diff_ht"] = pA["ht"] - pB["ht"]
    row["diff_overall_elo_pre"] = pA["overall_elo_pre"] - pB["overall_elo_pre"]
    row["diff_grass_elo_pre"] = pA["grass_elo_pre"] - pB["grass_elo_pre"]

    row["A_hand_code"] = 1 if pA.get("hand") == "R" else -1
    row["B_hand_code"] = 1 if pB.get("hand") == "R" else -1
    row["is_grass"] = 1

    res = []
    for col in feature_cols:
        res.append(row.get(col, 0.0))

    return np.array([res], dtype=np.float32)


def streak_to_probability(streak_a: int, streak_b: int, temperature: float = 2.5) -> float:
    diff = float(np.clip(streak_a - streak_b, -8, 8))
    return float(1.0 / (1.0 + np.exp(-(diff / temperature))))


def h2h_to_probability(wins_a: int, wins_b: int, prior_strength: float = 6.0) -> float:
    return float((wins_a + 0.5 * prior_strength) / (wins_a + wins_b + prior_strength))


def predict_composite(
    id_a: int,
    id_b: int,
    feats: Dict[int, Dict],
    model: object,
    feature_cols: List[str],
    h2h_map: Dict[Tuple[int, int], Tuple[int, int]],
    w_ml: float = 0.82,
    w_streak: float = 0.12,
    w_h2h: float = 0.06,
) -> float:
    base_w_ml, base_w_streak, base_w_h2h = validate_weights(w_ml, w_streak, w_h2h)

    fA = feats.get(id_a)
    fB = feats.get(id_b)

    if fA and fB:
        X = make_inference_row(fA, fB, feature_cols)
        prob_ml = float(model.predict_proba(X)[0][1])
    else:
        prob_ml = 0.5

    sA = fA.get("streak", 0) if fA else 0
    sB = fB.get("streak", 0) if fB else 0
    prob_streak = streak_to_probability(sA, sB)

    wins_a, wins_b = h2h_map.get((id_a, id_b), (0, 0))
    prob_h2h = h2h_to_probability(wins_a, wins_b)

    ml_confidence = float(np.clip(abs(prob_ml - 0.5) * 2.0, 0.0, 1.0))
    aux_gate = float((1.0 - ml_confidence) ** 2)

    eff_w_streak = base_w_streak * aux_gate
    eff_w_h2h = base_w_h2h * aux_gate
    eff_w_ml = 1.0 - eff_w_streak - eff_w_h2h

    final_prob = (eff_w_ml * prob_ml) + (eff_w_streak * prob_streak) + (eff_w_h2h * prob_h2h)

    if fA and fB:
        rank_a = float(fA.get("rank", 999.0))
        rank_b = float(fB.get("rank", 999.0))
        elo_a = float(fA.get("grass_elo_pre", 1500.0))
        elo_b = float(fB.get("grass_elo_pre", 1500.0))
        rank_gap = rank_b - rank_a
        elo_gap = elo_a - elo_b

        if prob_ml >= 0.72 and rank_gap >= 25 and elo_gap >= 45:
            final_prob = max(final_prob, 0.68)
        if prob_ml <= 0.28 and rank_gap <= -25 and elo_gap <= -45:
            final_prob = min(final_prob, 0.32)

    return float(np.clip(final_prob, 1e-6, 1 - 1e-6))


def simulate_bracket(
    draw: List[Tuple[int, int]],
    feats: Dict[int, Dict],
    model: object,
    feature_cols: List[str],
    h2h_map: Dict[Tuple[int, int], Tuple[int, int]],
    n_sims: int = 1,
    id_to_name: Dict[int, str] | None = None,
    w_ml: float = 0.50,
    w_streak: float = 0.30,
    w_h2h: float = 0.20,
    step_mode: str = "deterministic",
    seed: int = 42,
) -> pd.DataFrame:
    rounds_names = ["1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "Final"]
    rng = np.random.default_rng(seed)
    id_to_name = id_to_name or {}

    def _name(pid: int) -> str:
        return id_to_name.get(pid, str(pid))

    def _h2h_str(winner_id: int, loser_id: int) -> str:
        w, l = h2h_map.get((winner_id, loser_id), (0, 0))
        return f"{w}-{l}"

    predictions = []

    print("\n🚀 STARTING SIMULATION (Step-by-Step) 🚀")

    current_round = [list(pair) for pair in draw]
    round_idx = 0

    while len(current_round) > 0:
        round_name = rounds_names[round_idx] if round_idx < len(rounds_names) else f"R{round_idx + 1}"
        print(f"\n=== {round_name} ({len(current_round)} matches) ===")

        next_round = []

        for match in current_round:
            if len(match) == 1:
                next_round.append(match[0])
                continue

            p1, p2 = match[0], match[1]
            prob_p1 = predict_composite(
                p1, p2, feats, model, feature_cols, h2h_map,
                w_ml=w_ml, w_streak=w_streak, w_h2h=w_h2h,
            )

            if step_mode == "stochastic":
                p1_wins = bool(rng.random() < prob_p1)
            else:
                p1_wins = prob_p1 >= 0.5

            if p1_wins:
                winner, loser = p1, p2
                win_prob = prob_p1
            else:
                winner, loser = p2, p1
                win_prob = 1.0 - prob_p1

            winner_name = _name(winner)
            loser_name = _name(loser)

            print(
                f"{winner_name} def. {loser_name} "
                f"({win_prob * 100:.1f}% win for {winner_name}) "
                f"and H2H ({_h2h_str(winner, loser)})"
            )

            predictions.append({
                "round": round_name,
                "player1": _name(p1),
                "player2": _name(p2),
                "predicted_winner": winner_name,
                "predicted_prob": round(float(win_prob), 6),
                "w_ml": w_ml,
                "w_streak": w_streak,
                "w_h2h": w_h2h,
            })

            next_round.append(winner)

        if len(next_round) == 1:
            champ = next_round[0]
            print(f"\n🏆 TOURNAMENT CHAMPION: {_name(champ)} 🏆")
            break

        current_round = []
        for j in range(0, len(next_round) - 1, 2):
            current_round.append([next_round[j], next_round[j + 1]])

        if len(next_round) % 2 == 1:
            current_round.append([next_round[-1]])

        round_idx += 1

    return pd.DataFrame(predictions)


# =========================
# WEIGHT SEARCH / EVALUATION
# =========================
def evaluate_weights_on_actual_results(
    results_csv: Path,
    name_to_id: Dict[str, int],
    feats: Dict[int, Dict],
    model: object,
    feature_cols: List[str],
    h2h_map: Dict[Tuple[int, int], Tuple[int, int]],
    w_ml: float,
    w_streak: float,
    w_h2h: float,
) -> Tuple[float, float, int, int, pd.DataFrame]:
    w_ml, w_streak, w_h2h = validate_weights(w_ml, w_streak, w_h2h)
    results_df = load_and_clean_results(results_csv)

    eval_rows = []
    correct = 0
    total = 0
    skipped = 0
    y_true: List[int] = []
    y_prob: List[float] = []

    for _, row in results_df.iterrows():
        p1_name = str(row["player1"]).strip()
        p2_name = str(row["player2"]).strip()
        actual_winner = str(row["winner"]).strip()
        round_name = str(row.get("round", ""))
        score = row.get("score", np.nan)

        p1_id = name_to_id.get(normalize_name(p1_name))
        p2_id = name_to_id.get(normalize_name(p2_name))
        actual_winner_norm = normalize_name(actual_winner)

        if p1_id is None or p2_id is None:
            skipped += 1
            eval_rows.append({
                "round": round_name,
                "player1": p1_name,
                "player2": p2_name,
                "actual_winner": actual_winner,
                "score": score,
                "predicted_winner": None,
                "prob_player1": np.nan,
                "prob_actual_winner": np.nan,
                "is_correct": np.nan,
                "mapped": 0,
                "w_ml": w_ml,
                "w_streak": w_streak,
                "w_h2h": w_h2h,
            })
            continue

        prob_p1 = predict_composite(
            p1_id, p2_id, feats, model, feature_cols, h2h_map,
            w_ml=w_ml, w_streak=w_streak, w_h2h=w_h2h,
        )

        predicted_winner = p1_name if prob_p1 >= 0.5 else p2_name
        actual_is_p1 = int(actual_winner_norm == normalize_name(p1_name))
        prob_actual_winner = prob_p1 if actual_is_p1 == 1 else (1.0 - prob_p1)
        is_correct = int(normalize_name(predicted_winner) == actual_winner_norm)

        correct += is_correct
        total += 1
        y_true.append(actual_is_p1)
        y_prob.append(safe_prob(prob_p1))

        eval_rows.append({
            "round": round_name,
            "player1": p1_name,
            "player2": p2_name,
            "actual_winner": actual_winner,
            "score": score,
            "predicted_winner": predicted_winner,
            "prob_player1": prob_p1,
            "prob_actual_winner": prob_actual_winner,
            "is_correct": is_correct,
            "mapped": 1,
            "w_ml": w_ml,
            "w_streak": w_streak,
            "w_h2h": w_h2h,
        })

    accuracy = (correct / total) if total > 0 else 0.0
    if y_true:
        eps = 1e-6
        probs = np.clip(np.array(y_prob, dtype=float), eps, 1 - eps)
        truth = np.array(y_true, dtype=int)
        ll = float(-np.mean(truth * np.log(probs) + (1 - truth) * np.log(1 - probs)))
    else:
        ll = float("inf")

    return accuracy, ll, total, skipped, pd.DataFrame(eval_rows)


def summarize_accuracy_by_round(eval_df: pd.DataFrame) -> pd.DataFrame:
    mapped = eval_df[eval_df["mapped"] == 1].copy()
    if mapped.empty:
        return pd.DataFrame(columns=["round", "matches_evaluated", "correct", "accuracy"])

    grouped = mapped.groupby("round", dropna=False).agg(
        matches_evaluated=("is_correct", "count"),
        correct=("is_correct", "sum"),
    ).reset_index()
    grouped["accuracy"] = grouped["correct"] / grouped["matches_evaluated"]

    round_order = {"1/64": 1, "1/32": 2, "1/16": 3, "1/8": 4, "1/4": 5, "1/2": 6, "Final": 7}
    grouped["_order"] = grouped["round"].map(round_order).fillna(999)
    grouped = grouped.sort_values(["_order", "round"]).drop(columns=["_order"]).reset_index(drop=True)
    return grouped


def grid_search_weights(
    results_csv: Path,
    name_to_id: Dict[str, int],
    feats: Dict[int, Dict],
    model: object,
    feature_cols: List[str],
    h2h_map: Dict[Tuple[int, int], Tuple[int, int]],
    step_pct: int = 5,
) -> pd.DataFrame:
    rows = []

    for ml_pct in range(0, 101, step_pct):
        for streak_pct in range(0, 101 - ml_pct, step_pct):
            h2h_pct = 100 - ml_pct - streak_pct
            w_ml = ml_pct / 100.0
            w_streak = streak_pct / 100.0
            w_h2h = h2h_pct / 100.0

            acc, ll, evaluated, skipped, _ = evaluate_weights_on_actual_results(
                results_csv=results_csv,
                name_to_id=name_to_id,
                feats=feats,
                model=model,
                feature_cols=feature_cols,
                h2h_map=h2h_map,
                w_ml=w_ml,
                w_streak=w_streak,
                w_h2h=w_h2h,
            )

            rows.append({
                "ml_weight_pct": ml_pct,
                "streak_weight_pct": streak_pct,
                "h2h_weight_pct": h2h_pct,
                "accuracy": acc,
                "log_loss": ll,
                "evaluated_matches": evaluated,
                "skipped_matches": skipped,
            })
            print(
                f"ML={ml_pct:3d}% | STREAK={streak_pct:3d}% | H2H={h2h_pct:3d}% "
                f"-> acc={acc:.4f}, log_loss={ll:.4f}, evaluated={evaluated}, skipped={skipped}"
            )

    search_df = pd.DataFrame(rows)
    search_df = search_df.sort_values(
        by=["accuracy", "log_loss", "evaluated_matches"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    return search_df


def plot_weight_heatmap(search_df: pd.DataFrame, out_path: Path) -> None:
    if search_df.empty:
        print("Weight search dataframe is empty. Heatmap was not created.")
        return

    pivot = search_df.pivot(index="streak_weight_pct", columns="ml_weight_pct", values="accuracy")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(12, 8))
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="white")

    im = plt.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
    plt.colorbar(im, label="Accuracy")

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("ML Weight (%)")
    plt.ylabel("Streak Weight (%)")
    plt.title("Accuracy Heatmap for Composite Weights\n(H2H Weight = 100 - ML - Streak)")

    best = search_df.sort_values(["accuracy", "log_loss"], ascending=[False, True]).iloc[0]
    best_x = list(pivot.columns).index(best["ml_weight_pct"])
    best_y = list(pivot.index).index(best["streak_weight_pct"])
    plt.scatter([best_x], [best_y], marker="*", s=220, edgecolors="black", linewidths=1.0)
    plt.text(
        best_x,
        best_y + 0.4,
        f"best: {best['accuracy']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    for yi, streak_pct in enumerate(pivot.index):
        for xi, ml_pct in enumerate(pivot.columns):
            val = pivot.loc[streak_pct, ml_pct]
            if pd.notna(val):
                plt.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# DRAW MAPPING
# =========================
def load_draw_pairs(draw_csv: Path, name_to_id: Dict[str, int]) -> List[Tuple[int, int]]:
    draw_df = pd.read_csv(draw_csv)
    draw_pairs = []

    for _, row in draw_df.iterrows():
        p1_name = str(row["player1"]).strip()
        p2_name = str(row["player2"]).strip()

        p1 = name_to_id.get(normalize_name(p1_name))
        p2 = name_to_id.get(normalize_name(p2_name))

        if p1 is None and p1_name.isdigit():
            p1 = int(p1_name)
        if p2 is None and p2_name.isdigit():
            p2 = int(p2_name)

        if p1 is not None and p2 is not None:
            draw_pairs.append((p1, p2))
        else:
            print(f"Warning: Could not map draw players '{p1_name}' or '{p2_name}'")

    return draw_pairs


# =========================
# ARGUMENTS
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wimbledon 2025 predictor with composite weight search and heatmap.")
    parser.add_argument("--data_dir", type=str, default=str(CFG.data_dir))
    parser.add_argument("--draw_csv", type=str, default=str(CFG.draw_csv))
    parser.add_argument("--results_csv", type=str, default=str(CFG.actual_results_csv))
    parser.add_argument("--prediction_cutoff_date", type=str, default=CFG.prediction_cutoff_date)
    parser.add_argument("--max_rows", type=int, default=CFG.max_rows)

    parser.add_argument("--w_ml", type=float, default=CFG.w_ml)
    parser.add_argument("--w_streak", type=float, default=CFG.w_streak)
    parser.add_argument("--w_h2h", type=float, default=CFG.w_h2h)

    parser.add_argument("--search_weights", action="store_true", default=CFG.run_weight_search)
    parser.add_argument("--no_search_weights", action="store_false", dest="search_weights")
    parser.add_argument("--search_step", type=int, default=CFG.weight_step_pct)

    parser.add_argument("--step_mode", choices=["deterministic", "stochastic"], default=CFG.step_mode)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--out_prefix", type=str, default=CFG.out_prefix)

    return parser.parse_args()


# =========================
# MAIN EXECUTION
# =========================
def main() -> None:
    args = parse_args()

    cfg = Config(
        data_dir=Path(args.data_dir),
        draw_csv=Path(args.draw_csv),
        actual_results_csv=Path(args.results_csv),
        prediction_cutoff_date=args.prediction_cutoff_date,
        max_rows=args.max_rows,
        seed=args.seed,
        step_mode=args.step_mode,
        w_ml=args.w_ml,
        w_streak=args.w_streak,
        w_h2h=args.w_h2h,
        run_weight_search=args.search_weights,
        weight_step_pct=args.search_step,
        out_prefix=args.out_prefix,
    )

    print("--- 🎾 TENNIS PREDICTOR + WEIGHT SEARCH 🎾 ---")
    print("Loading data...")
    m_files = get_match_files(cfg)
    matches = load_matches(m_files, max_rows=cfg.max_rows)
    players = load_players(cfg)
    rankings = load_rankings(cfg)

    cutoff = parse_cutoff_date(cfg.prediction_cutoff_date)
    pre_tournament_matches = matches[matches["tourney_date"] < cutoff].copy()
    if pre_tournament_matches.empty:
        raise ValueError(f"No training matches found before cutoff date {cutoff.date()}.")

    print(f"Loaded {len(matches)} matches in total.")
    print(f"Using {len(pre_tournament_matches)} matches before cutoff {cutoff.date()} for model/features.")

    print("Building Elo ratings...")
    elo_rows = build_elo_rows(pre_tournament_matches, cfg)

    print("Building match features...")
    pm = build_player_match_table(pre_tournament_matches)
    pm = add_rolling_features(pm)

    print("Preparing training set...")
    X, y, feature_cols = build_training_set(pre_tournament_matches, pm, elo_rows)
    model = train_model(X, y)

    print("Preparing 2025 player features (streaks, H2H)...")
    player_feats = prepare_2025_features(pm, elo_rows, rankings, players, as_of_date=cutoff)
    h2h_map = build_h2h_map(pre_tournament_matches)

    print(f"Loading draw from {cfg.draw_csv}...")
    name_to_id = build_name_to_id(players)
    draw_pairs = load_draw_pairs(cfg.draw_csv, name_to_id)

    if not draw_pairs:
        print("No valid draw pairs found. Exiting.")
        return

    id_to_name = {int(row["player_id"]): str(row["full_name"]).strip() for _, row in players.iterrows()}

    pred_df = simulate_bracket(
        draw_pairs,
        player_feats,
        model,
        feature_cols,
        h2h_map,
        n_sims=1,
        id_to_name=id_to_name,
        w_ml=cfg.w_ml,
        w_streak=cfg.w_streak,
        w_h2h=cfg.w_h2h,
        step_mode=cfg.step_mode,
        seed=cfg.seed,
    )
    pred_path = Path(f"{cfg.out_prefix}_2025_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")

    if cfg.actual_results_csv.exists():
        print("\nEvaluating current weights on actual Wimbledon results...")
        acc, ll, evaluated, skipped, eval_df = evaluate_weights_on_actual_results(
            results_csv=cfg.actual_results_csv,
            name_to_id=name_to_id,
            feats=player_feats,
            model=model,
            feature_cols=feature_cols,
            h2h_map=h2h_map,
            w_ml=cfg.w_ml,
            w_streak=cfg.w_streak,
            w_h2h=cfg.w_h2h,
        )
        eval_path = Path(f"{cfg.out_prefix}_actual_match_eval_current_weights.csv")
        eval_df.to_csv(eval_path, index=False)
        round_summary = summarize_accuracy_by_round(eval_df)
        round_path = Path(f"{cfg.out_prefix}_accuracy_by_round_current_weights.csv")
        round_summary.to_csv(round_path, index=False)
        print(
            f"Current weights -> accuracy={acc:.4f}, log_loss={ll:.4f}, "
            f"evaluated={evaluated}, skipped={skipped}"
        )
        print(f"Saved: {eval_path}")
        print(f"Saved: {round_path}")
    else:
        print(f"Actual results file not found: {cfg.actual_results_csv}")

    if cfg.run_weight_search and cfg.actual_results_csv.exists():
        print("\nSearching best composite weights on actual Wimbledon 2025 matches...")
        search_df = grid_search_weights(
            results_csv=cfg.actual_results_csv,
            name_to_id=name_to_id,
            feats=player_feats,
            model=model,
            feature_cols=feature_cols,
            h2h_map=h2h_map,
            step_pct=cfg.weight_step_pct,
        )

        search_path = Path(f"{cfg.out_prefix}_weight_search_results.csv")
        search_df.to_csv(search_path, index=False)
        print(f"Saved: {search_path}")

        best_row = search_df.iloc[0]
        best_w_ml = best_row["ml_weight_pct"] / 100.0
        best_w_streak = best_row["streak_weight_pct"] / 100.0
        best_w_h2h = best_row["h2h_weight_pct"] / 100.0

        print(
            "\nBEST WEIGHTS FOUND:\n"
            f"ML={best_row['ml_weight_pct']}% | "
            f"STREAK={best_row['streak_weight_pct']}% | "
            f"H2H={best_row['h2h_weight_pct']}% | "
            f"accuracy={best_row['accuracy']:.4f} | log_loss={best_row['log_loss']:.4f}"
        )

        heatmap_path = Path(f"{cfg.out_prefix}_weight_heatmap.png")
        plot_weight_heatmap(search_df, heatmap_path)
        print(f"Saved: {heatmap_path}")

        best_acc, best_ll, best_evaluated, best_skipped, best_eval_df = evaluate_weights_on_actual_results(
            results_csv=cfg.actual_results_csv,
            name_to_id=name_to_id,
            feats=player_feats,
            model=model,
            feature_cols=feature_cols,
            h2h_map=h2h_map,
            w_ml=best_w_ml,
            w_streak=best_w_streak,
            w_h2h=best_w_h2h,
        )
        best_eval_path = Path(f"{cfg.out_prefix}_actual_match_eval_best_weights.csv")
        best_eval_df.to_csv(best_eval_path, index=False)
        print(
            f"Best weights evaluation -> accuracy={best_acc:.4f}, log_loss={best_ll:.4f}, "
            f"evaluated={best_evaluated}, skipped={best_skipped}"
        )
        print(f"Saved: {best_eval_path}")

        best_round_summary = summarize_accuracy_by_round(best_eval_df)
        best_round_path = Path(f"{cfg.out_prefix}_accuracy_by_round_best_weights.csv")
        best_round_summary.to_csv(best_round_path, index=False)
        print(f"Saved: {best_round_path}")

        best_pred_df = simulate_bracket(
            draw_pairs,
            player_feats,
            model,
            feature_cols,
            h2h_map,
            n_sims=1,
            id_to_name=id_to_name,
            w_ml=best_w_ml,
            w_streak=best_w_streak,
            w_h2h=best_w_h2h,
            step_mode=cfg.step_mode,
            seed=cfg.seed,
        )
        best_pred_path = Path(f"{cfg.out_prefix}_bracket_predictions_best_weights.csv")
        best_pred_df.to_csv(best_pred_path, index=False)
        print(f"Saved: {best_pred_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
