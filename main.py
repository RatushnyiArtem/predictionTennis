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
#          CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("tennis_atp-master")
    draw_csv: Path = Path("wimbledon_2025_draw_r1.csv")

    # surface for calculating
    train_surface: str = "Grass"
    
    # extra datasets
    use_amateur: bool = False
    use_futures: bool = False
    use_qual_chall: bool = True 
    use_doubles: bool = False

    # perfomance
    max_rows: Optional[int] = None 
    n_sims: int = 1  # 1 run for step-by-step bracket, or 1000 for probability stats
    seed: int = 42

    # elo params
    elo_init: float = 1500.0
    elo_k_overall: float = 24.0
    elo_k_grass: float = 28.0

    # Output
    step_mode: str = "stochastic" # "deterministic" (favor higher prob) or "stochastic" (random weighted)


CFG = Config()


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

    # Unique match key
    if "tourney_id" not in m.columns: m["tourney_id"] = ""
    if "match_num" not in m.columns: m["match_num"] = -1
    
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
# FEATURE ENGINEERING
# =========================
def build_player_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    
    for col in ["winner_rank", "loser_rank", "w_ace", "l_ace", "w_df", "l_df", "winner_age", "winner_ht", "winner_hand"]:
        if col not in m.columns: m[col] = np.nan

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
def train_model(X, y):
    print(f"Training on {len(X)} samples...")
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=42)
    
    model.fit(X, y)
    return model

# =========================
# 2025 PREP & ENHANCED LOGIC
# =========================

def get_current_streak(pm_df, player_id):
    """
    active streak.
    + = Winning Streak.
    - = Losing Streak.
    """
    hist = pm_df[pm_df['player_id'] == player_id].sort_values("tourney_date", ascending=False)
    if hist.empty:
        return 0
    
    streak = 0
    first_res = hist.iloc[0]['is_win']
    direction = 1 if first_res == 1 else -1
    
    for _, row in hist.iterrows():
        if (row['is_win'] == 1 and direction == 1) or (row['is_win'] == 0 and direction == -1):
            streak += direction
        else:
            break
    return streak

def build_h2h_map(matches_df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[int, int]]:
    h2h = {}
    grouped = matches_df.groupby(["winner_id", "loser_id"]).size().reset_index(name="counts")
    
    for _, row in grouped.iterrows():
        w, l, c = int(row["winner_id"]), int(row["loser_id"]), int(row["counts"])
        if (w, l) not in h2h: h2h[(w, l)] = [0, 0]
        h2h[(w, l)][0] += c
        if (l, w) not in h2h: h2h[(l, w)] = [0, 0]
        h2h[(l, w)][1] += c
        
    return {k: tuple(v) for k, v in h2h.items()}

def prepare_2025_features(pm: pd.DataFrame, elo_rows: pd.DataFrame, rankings: pd.DataFrame, players: pd.DataFrame) -> Dict[int, Dict]:
    last_elos = elo_rows.sort_values("match_key").groupby("player_id").tail(1).set_index("player_id")
    last_stats = pm.sort_values("tourney_date").groupby("player_id").tail(1).set_index("player_id")
    
    latest_date = rankings["ranking_date"].max()
    
    # DROP DUPLICATES ---
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
            feats["hand"] = p_data["hand"]
            dob = str(p_data["dob"])
            feats["age"] = 2025 - int(dob[:4]) if len(dob) >= 4 else 25
            feats["ht"] = float(p_data["height"]) if not pd.isna(p_data["height"]) else 185.0
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

def predict_composite(id_a: int, id_b: int, feats: Dict, model: object, feature_cols: List[str], h2h_map: Dict) -> float:
    fA = feats.get(id_a)
    fB = feats.get(id_b)
    
    # base ML Probability
    if fA and fB:
        X = make_inference_row(fA, fB, feature_cols)
        prob_ml = model.predict_proba(X)[0][1]
    else:
        prob_ml = 0.5
    
    # Streak/Form Factor
    sA = fA.get("streak", 0) if fA else 0
    sB = fB.get("streak", 0) if fB else 0
    
    score_A = np.clip(sA, -10, 10) / 20.0 + 0.5
    score_B = np.clip(sB, -10, 10) / 20.0 + 0.5
    prob_streak = score_A / (score_A + score_B + 1e-6)
    
    # Head to head Factor
    wins_a, wins_b = h2h_map.get((id_a, id_b), (0, 0))
    total_h2h = wins_a + wins_b
    if total_h2h > 0:
        prob_h2h = wins_a / total_h2h
    else:
        prob_h2h = 0.5
        
    # Weights
    W_ML = 0.50
    W_STREAK = 0.30
    W_H2H = 0.28 
    
    final_prob = (W_ML * prob_ml) + (W_STREAK * prob_streak) + (W_H2H * prob_h2h)
    return float(final_prob)

def simulate_bracket(
    draw: List[Tuple[int, int]],
    feats: Dict,
    model: object,
    feature_cols: List[str],
    h2h_map: Dict,
    n_sims: int = 1,
    id_to_name: Dict = {},
) -> pd.DataFrame:
    rounds_names = ["1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "Final"]

    def _name(pid: int) -> str:
        return id_to_name.get(pid, str(pid))

    def _h2h_str(winner_id: int, loser_id: int) -> str:
        w, l = h2h_map.get((winner_id, loser_id), (0, 0))
        return f"{w}-{l}"

    predictions = []

    print(f"\n🚀 STARTING SIMULATION (Step-by-Step) 🚀")

    current_round = [list(pair) for pair in draw]
    round_idx = 0

    while len(current_round) > 0:
        round_name = rounds_names[round_idx] if round_idx < len(rounds_names) else "F"
        print(f"\n=== {round_name} ({len(current_round)} matches) ===")

        next_round = []

        for match in current_round:
            if len(match) == 1:
                next_round.append(match[0])
                continue

            p1, p2 = match[0], match[1]

            # prob = P(player1 beats player2)
            prob_p1 = predict_composite(p1, p2, feats, model, feature_cols, h2h_map)

            # deterministic pick
            if prob_p1 >= 0.5:
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

            # store prediction row
            predictions.append({
                "round": round_name,
                "player1": _name(p1),
                "player2": _name(p2),
                "predicted_winner": winner_name,
                "predicted_prob": round(win_prob, 6),
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
# MAIN EXECUTION
# =========================
def main():
    print("--- 🎾 TENNIS PREDICTOR: 🎾 ---")
    
    print("Loading data...")
    m_files = get_match_files(CFG)
    matches = load_matches(m_files, max_rows=CFG.max_rows)
    players = load_players(CFG)
    rankings = load_rankings(CFG)
    
    print(f"Loaded {len(matches)} matches.")

    print("Building Elo ratings...")
    elo_rows = build_elo_rows(matches, CFG)
    
    print("Building Match Features...")
    pm = build_player_match_table(matches)
    pm = add_rolling_features(pm)
    
    print("Preparing Training Set...")
    X, y, feature_cols = build_training_set(matches, pm, elo_rows)
    model = train_model(X, y)
    
    print("Preparing 2025 Player Features (Streaks, H2H)...")
    player_feats = prepare_2025_features(pm, elo_rows, rankings, players)
    h2h_map = build_h2h_map(matches)
    
    print(f"Loading Draw from {CFG.draw_csv}...")
    try:
        draw_df = pd.read_csv(CFG.draw_csv)
    except:
        print("Draw file not found.")
        return

    name_to_id = {}
    for _, row in players.iterrows():
        name_to_id[row["full_name"].lower()] = row["player_id"]
    
    draw_pairs = []
    for _, row in draw_df.iterrows():
        p1_name = str(row["player1"]).strip()
        p2_name = str(row["player2"]).strip()
        p1 = name_to_id.get(p1_name.lower())
        p2 = name_to_id.get(p2_name.lower())
        
        if p1 is None and p1_name.isdigit(): p1 = int(p1_name)
        if p2 is None and p2_name.isdigit(): p2 = int(p2_name)
        
        if p1 and p2:
            draw_pairs.append((p1, p2))
        else:
            print(f"Warning: Could not map {p1_name} or {p2_name}")

    if not draw_pairs:
        print("No valid draw pairs found. Exiting.")
        return

    id_to_name = {v: k.title() for k, v in name_to_id.items()}
    pred_df = simulate_bracket(draw_pairs, player_feats, model, feature_cols, h2h_map, n_sims=1, id_to_name=id_to_name)
    pred_df.to_csv("wimbledon_2025_predictions.csv", index=False)
    print("Saved: wimbledon_2025_predictions.csv")

if __name__ == "__main__":
    main()