import pandas as pd
import re
import unicodedata

REAL = "wimbledon_2025_all_rounds_results.csv"
PRED = "wimbledon_2025_predictions.csv"


def norm(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pair_key(a, b):
    a, b = norm(a), norm(b)
    return "__vs__".join(sorted([a, b]))


# load
real = pd.read_csv(REAL)
pred = pd.read_csv(PRED)

# normalize
real["round"] = real["round"].map(norm)
pred["round"] = pred["round"].map(norm)

real["pair"] = real.apply(lambda r: pair_key(r["player1"], r["player2"]), axis=1)
pred["pair"] = pred.apply(lambda r: pair_key(r["player1"], r["player2"]), axis=1)

real["winner_clean"] = real["winner"].map(norm)
pred["pred_clean"] = pred["predicted_winner"].map(norm)

# INNER JOIN: only same real match vs its prediction
m = real.merge(
    pred[["round", "pair", "pred_clean"]],
    on=["round", "pair"],
    how="inner",
)

# correctness
m["correct"] = (m["winner_clean"] == m["pred_clean"]).astype(int)

overall_acc = m["correct"].mean() * 100

print(f"\n=== MATCH-LEVEL ACCURACY ===")
print(f"Matches evaluated: {len(m)}")
print(f"Overall accuracy: {overall_acc:.2f}%")
by_round = (
    m.groupby("round")["correct"]
    .mean()
    .mul(100)
    .reset_index(name="accuracy_%")
)
ROUND_ORDER = ["1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "Final"]

by_round["round"] = pd.Categorical(
    by_round["round"],
    categories=ROUND_ORDER,
    ordered=True
)

by_round = by_round.sort_values("round")

print("\n=== Accuracy by round ===")
print(by_round.to_string(index=False))

