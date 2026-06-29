# LLM Tasks (Gemini) — Sentiment CoT & Hyperparameter ToT

Identical tasks as the Anthropic version, rewritten to use **Google Gemini** via the `google-generativeai` SDK.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free API key from https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY="AIza..."       # Linux / macOS
set GOOGLE_API_KEY=AIza...            # Windows CMD
```

---

## Task 1 — Sentiment Analysis + Chain-of-Thought

```bash
cd task1_sentiment
python sentiment_cot.py
```

**Model used**: `gemini-1.5-flash` (change `MODEL_NAME` in the script to `gemini-1.5-pro` for higher quality)

**What it does**:
- 10 e-commerce product reviews loaded in-script
- Few-shot CoT prompt with one labelled example per class (positive / neutral / negative)
- Gemini reasons through 4 explicit steps per review:
  1. Identify positive-sentiment phrases
  2. Identify negative-sentiment phrases
  3. Check for contradictions / mixed sentiment
  4. Decide final label and explain why
- Prints full CoT reasoning + summary table to console
- Saves `sentiment_results.json`

---

## Task 2 — Hyperparameter Tuning + Tree-of-Thought

```bash
cd task2_hyperparams
python hyperparameter_tot.py
```

**What it does**:
- Trains `RandomForestClassifier` on Iris across **30 hyperparameter configs**
- Records `cv_score`, `train_score`, overfitting `gap`, and `time_s` per config
- Sends the full results table to Gemini with a Tree-of-Thought prompt structured into:

  | Branch | Focus |
  |--------|-------|
  | Branch 1 | Group configs by parameter ranges; identify promising regions |
  | Branch 2 | Bias-variance tradeoff — flag overfitting, underfitting, sweet spot |
  | Branch 3 | Training efficiency — performance-per-second, diminishing returns |
  | Convergence | Single best config with full justification vs runner-ups |

- Saves `hp_results.csv` and `tot_analysis.txt`

---

## Switching Gemini models

In either script, change the `MODEL_NAME` constant at the top:

```python
MODEL_NAME = "gemini-1.5-flash"   # fast, free tier
MODEL_NAME = "gemini-1.5-pro"     # more capable, slower
MODEL_NAME = "gemini-2.0-flash"   # latest flash model (if available)
```

---

## Project structure

```
llm_tasks_gemini/
├── requirements.txt
├── README.md
├── task1_sentiment/
│   └── sentiment_cot.py          # Task 1 — Gemini + CoT
└── task2_hyperparams/
    └── hyperparameter_tot.py     # Task 2 — Gemini + ToT
```
