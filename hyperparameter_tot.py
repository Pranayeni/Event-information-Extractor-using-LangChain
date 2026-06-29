"""
Task 2: Hyperparameter Tuning with Tree-of-Thought (ToT) via Google Gemini
===========================================================================
Trains RandomForestClassifier on Iris across a 30-config grid, records
CV scores / overfitting indicators / training times, then sends results
to Gemini guided by a Tree-of-Thought prompt that explores three analysis
branches before converging on a final best-config recommendation.

Requirements:
    pip install google-generativeai scikit-learn numpy pandas tabulate
    export GOOGLE_API_KEY="your-gemini-api-key-here"
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import google.generativeai as genai

from itertools import product
from tabulate import tabulate
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Configure Gemini
# ---------------------------------------------------------------------------
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-1.5-flash"   # or "gemini-1.5-pro" for higher quality


# ---------------------------------------------------------------------------
# Hyperparameter grid — 30 configurations
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators":      [10, 50, 100, 200],
    "max_depth":         [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "max_features":      ["sqrt", "log2"],
}

_all_combos = [
    dict(zip(PARAM_GRID.keys(), vals))
    for vals in product(*PARAM_GRID.values())
]
random.seed(42)
random.shuffle(_all_combos)
HP_CONFIGS = _all_combos[:30]


# ---------------------------------------------------------------------------
# ML training & evaluation
# ---------------------------------------------------------------------------
def run_hyperparameter_search(configs: list[dict]) -> pd.DataFrame:
    """
    Fit RandomForestClassifier for every config.
    Returns DataFrame with cv_score, train_score, gap, time_s.
    """
    X, y = load_iris(return_X_y=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    print(f"Running {len(configs)} configs on Iris dataset ...\n")
    print(f"{'#':>4}  {'n_est':>6}  {'depth':>6}  {'mss':>4}  {'mf':>6}  "
          f"{'CV Acc':>8}  {'Train':>8}  {'Gap':>7}  {'Time':>6}")
    print("─" * 70)

    for idx, params in enumerate(configs, start=1):
        model = RandomForestClassifier(random_state=42, **params)

        t0 = time.perf_counter()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        model.fit(X, y)
        elapsed = time.perf_counter() - t0

        cv_mean   = cv_scores.mean()
        train_sc  = accuracy_score(y, model.predict(X))
        gap       = train_sc - cv_mean
        depth_str = str(params["max_depth"]) if params["max_depth"] else "None"

        print(
            f"{idx:>4}  {params['n_estimators']:>6}  {depth_str:>6}  "
            f"{params['min_samples_split']:>4}  {params['max_features']:>6}  "
            f"{cv_mean:>8.4f}  {train_sc:>8.4f}  {gap:>7.4f}  {elapsed:>5.2f}s"
        )

        records.append({
            "config_id":         idx,
            "n_estimators":      params["n_estimators"],
            "max_depth":         params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "max_features":      params["max_features"],
            "cv_score":          round(cv_mean, 4),
            "cv_std":            round(cv_scores.std(), 4),
            "train_score":       round(train_sc, 4),
            "gap":               round(gap, 4),
            "time_s":            round(elapsed, 3),
        })

    df = pd.DataFrame(records)
    best = df.loc[df["cv_score"].idxmax()]
    print(f"\n★  Best by CV score: Config #{int(best.config_id)}  →  CV Acc = {best.cv_score:.4f}")
    return df


def save_hp_results(df: pd.DataFrame, path: str = "hp_results.csv") -> None:
    df.to_csv(path, index=False)
    print(f"Grid results saved → {path}")


# ---------------------------------------------------------------------------
# Tree-of-Thought prompt
# ---------------------------------------------------------------------------
TOT_SYSTEM_INSTRUCTION = (
    "You are a senior machine learning engineer expert in hyperparameter optimization. "
    "You reason step by step, explore multiple analysis branches independently, then "
    "synthesize a final decision. Be quantitative — always cite specific config IDs "
    "and numeric evidence. Follow the Tree-of-Thought structure exactly as given."
)


def build_tot_prompt(df: pd.DataFrame) -> str:
    table_str = tabulate(
        df[[
            "config_id", "n_estimators", "max_depth", "min_samples_split",
            "max_features", "cv_score", "train_score", "gap", "time_s",
        ]].rename(columns={
            "config_id": "#", "n_estimators": "n_est", "max_depth": "depth",
            "min_samples_split": "mss", "max_features": "mf",
            "cv_score": "CV Acc", "train_score": "Train Acc",
            "gap": "Gap", "time_s": "Time(s)",
        }),
        headers="keys",
        tablefmt="github",
        showindex=False,
        floatfmt=".4f",
    )

    return f"""Below are results from a 30-configuration hyperparameter grid search of RandomForestClassifier on the Iris dataset (5-fold stratified CV accuracy).

{table_str}

Column definitions:
- CV Acc    : mean 5-fold cross-validation accuracy (primary metric)
- Train Acc : accuracy on the full training set
- Gap       : Train Acc − CV Acc  (proxy for overfitting; higher = more overfit)
- Time(s)   : wall-clock seconds for CV + final fit

=== TREE-OF-THOUGHT ANALYSIS ===

Work through THREE independent reasoning branches, then converge:

─────────────────────────────────────────────────────────────────
BRANCH 1 — Group by broad parameter ranges
─────────────────────────────────────────────────────────────────
1a. Group all 30 configs by n_estimators: small (10–50), medium (100), large (200).
    Compute the average CV Acc for each group.

1b. Group by max_depth: shallow (5), medium (10–20), unlimited (None).
    Compute average CV Acc for each depth group.

1c. Which parameter ranges look most promising?
    Name the top candidate configs emerging from this branch.

─────────────────────────────────────────────────────────────────
BRANCH 2 — Bias-Variance Tradeoff
─────────────────────────────────────────────────────────────────
2a. List configs where Gap > 0.03 (high overfitting / high variance).
    Explain why those parameter settings cause overfitting.

2b. List configs where CV Acc < 0.93 (underfitting / high bias).
    What settings drive this underfitting?

2c. Identify the "sweet spot" configs: CV Acc ≥ 0.95 AND Gap ≤ 0.03.
    These best balance bias and variance — list them all.

─────────────────────────────────────────────────────────────────
BRANCH 3 — Training Efficiency
─────────────────────────────────────────────────────────────────
3a. Among configs with CV Acc ≥ 0.95, rank by training time (fastest first).
    Compute performance-per-second (CV Acc / Time) for the top 5.

3b. Does increasing n_estimators beyond 100 yield meaningful CV gains,
    or are there diminishing returns? Cite specific numbers.

3c. Which configs give the best performance / training-time tradeoff?

─────────────────────────────────────────────────────────────────
CONVERGENCE — Final Recommendation
─────────────────────────────────────────────────────────────────
Synthesize findings from all three branches.

State clearly:
  • Single BEST config (by config #) with its exact parameters.
  • Why it beats the next two runner-up configs (compare explicitly).
  • Which hyperparameter is most responsible for its strong performance.
  • One or two concrete next steps to push CV accuracy even higher.
"""


# ---------------------------------------------------------------------------
# LLM reasoning
# ---------------------------------------------------------------------------
def run_tot_analysis(df: pd.DataFrame) -> str:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=TOT_SYSTEM_INSTRUCTION,
    )
    prompt = build_tot_prompt(df)
    print(f"\nSending results to Gemini ({MODEL_NAME}) for Tree-of-Thought analysis ...")
    response = model.generate_content(prompt)
    return response.text


def save_tot_analysis(text: str, path: str = "tot_analysis.txt") -> None:
    with open(path, "w") as f:
        f.write(text)
    print(f"Tree-of-Thought analysis saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1 — grid search
    df = run_hyperparameter_search(HP_CONFIGS)
    save_hp_results(df)

    # Step 2 — ToT analysis
    tot_text = run_tot_analysis(df)

    print("\n" + "=" * 70)
    print("TREE-OF-THOUGHT ANALYSIS (Gemini)")
    print("=" * 70)
    print(tot_text)

    save_tot_analysis(tot_text)
    print("\nDone. Outputs: hp_results.csv  |  tot_analysis.txt")
