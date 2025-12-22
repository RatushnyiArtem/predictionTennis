## 🎾 Wimbledon 2025 Tournament Prediction System
This project is a probability-based tournament prediction engine designed to analyze and simulate the Wimbledon 2025 men’s singles draw. The focus of the project is not on producing a single guess, but on modeling match outcomes using learned probabilities and propagating them consistently through a real tournament structure.
The system combines machine learning predictions, Elo-based player strength modeling, and structured tournament logic to estimate how players progress across rounds.
---
## 🚀 Implementation
# 1. Calculation
I constructed a structured feature pipeline that captures multiple dimensions of player performance:
***Global and surface-adjusted Elo ratings***
***Recent performance indicators***
***Head-to-head (H2H) statistics***
***Match and tournament context features***
***Aggregated player statistics derived from historical matches***
These features are merged carefully to maintain consistency across all tournament rounds and avoid data leakage.
# 2. Machine Learning Prediction Model
To estimate match outcomes, I implemented and compared two ensemble models:
***XGBoost – gradient-boosted decision trees optimized for structured tabular data***
***Random Forest – bagging-based ensemble model for robust, low-variance predictions***
Both models output a win probability rather than a binary label:
    P(Player A beats Player B)
This probabilistic approach preserves information about confidence and uncertainty and enables flexible downstream decision logic.
# 3. Accuracy Calculation (accuracy.py)
I constructed accuracy calculation, using real all round results vs my prediction (wimbledon_2025_all_rounds_results.csv against wimbledon_2025_predictions.csv)
# 4. Simulation Logic
I constructed real simulation logic what ML predicted stage by stage

## 🛠 Technologies & Concepts Used
Python
Pandas / NumPy
XGBoost
Random Forest
Elo rating systems
Probability-based prediction
Stochastic decision strategies
Tournament modeling & simulation architecture