# LLM-Augmented Event-Study Portfolio Research

Combines **GPT-4o + RAG** with classical event-study and ML to test whether
LLM-extracted signals from earnings call transcripts add incremental alpha
over price-only technical features. Universe: 50 US large caps, 2018–2025,
~1,500 WRDS earnings call transcripts.

## Headline result (honest)

| Test period | Tech-only Sharpe | Event-Study (LLM+Tech) Sharpe | ΔSharpe |
|---|---|---|---|
| 2024 (bull) | +0.30 | +0.43 | +0.13 |
| 2025 (volatile) | −0.13 | +0.95 | +1.07 |

- ΔSharpe is positive in both OOS years (sign-consistent)
- Paired Newey–West and Memmel Sharpe-difference tests do **not** reach 5%
  significance — sample size (~165 trading days/year, ~220 events) is the
  binding constraint, not signal absence
- A 120-cell sensitivity scan (entry × hold × weighting) yields zero cells
  passing Bonferroni-adjusted p, consistent with low-power small-sample regime
- Event-level Spearman ρ(prob, CAR_2_21) pooled p ≈ 0.07 (one-sided MW),
  directionally consistent with PEAD literature

The result is a **weak positive but underpowered** finding, written up with
explicit limitations rather than overstated significance.

## Pipeline (5 sequential notebooks)

```
00_etl.ipynb              # WRDS zips → parquet (one-off; needs WRDS access)
01_eda.ipynb              # Sanity-check the cleaned tables
02_rag_llm_agent.ipynb    # RAG retrieval + GPT-4o signal extraction
03_car_targets.ipynb      # Market-model CAR labels (4 windows)
04_ml_models.ipynb        # Tail-filtered RF on car_2_11 (Optuna-tuned)
05_backtest_ablation.ipynb # Long-short event-driven backtest + significance tests
```

`src/llm_agent/` holds the reusable contract (data loaders, feature builders,
event-study utilities, models, metrics) — both notebooks and scripts import
from there.

## Project structure

```
llm-portfolio/
├── config.py                       # Paths, ticker universe, time splits
├── pyproject.toml
├── .env.example                    # Copy to .env, fill OPENAI_API_KEY
├── src/llm_agent/
│   ├── data_loader.py              # Unified data access (load_*, split_by_period)
│   ├── transcript_parser.py        # Prepared remarks vs Q&A segmentation
│   ├── price_data.py               # yfinance + market-model utilities
│   ├── features.py                 # Tech / sector / market-cap / LLM features
│   ├── models.py                   # Sklearn pipelines for logreg/SVM/RF/XGB
│   └── metrics.py                  # Sharpe / IR / MDD / AUC
├── notebooks/                      # 00 → 05 in order
├── data/
│   └── clean/                      # earnings_*, prices, car_labels (committed)
└── models/event_study_best.pkl     # Tuned RF on car_2_11 (committed)
```

## Quick start

```bash
pip install -e .
cp .env.example .env       # fill OPENAI_API_KEY
jupyter lab notebooks/     # run 03 → 05 to reproduce backtest
```

`data/clean/*.parquet`, `data/signals/llm_signals.parquet`, and
`models/event_study_best.pkl` are committed, so NB03–NB05 reproduce end-to-end
without rerunning NB00 (WRDS) or NB02 (OpenAI).

To re-run NB00 you need WRDS access and the raw zips locally — set
`WRDS_ZIP_DIR` in `.env` to point at them. To re-run NB02 you need an OpenAI
key and ~$30–50 of API budget for the full 1,500-transcript pass.

## Three-arm backtest design (NB05)

| Arm | Definition |
|---|---|
| **BuyHold_EW50** | Equal-weight passive hold of all 50 names |
| **Tech-only** | RF on technical indicators + sector + log market cap |
| **Event-Study (LLM+Tech)** | RF on Tech features + 6 GPT-4o-extracted signals |

Trade rule: open at `event + 5` trading days, hold 15 days, weight by
`2·(p − 0.5)` clipped to ±1. Long/short thresholds are val-2023 quintile
boundaries, frozen for test. Hyperparameters (entry / hold / weighting) were
selected on val 2023 — see NB05 §1 for the grid that produced the chosen config.

## Methodology choices worth noting

- **Target = market-model CAR_[+2, +11]** with α, β estimated on a
  [−120, −20] estimation window per event
- **Tail filter** on labels: drop events with |CAR| < 0.5·σ_e·√10 to
  remove no-information observations (label noise dominated)
- **Anti-leakage**: scaler / winsorization / sector dummy alignment all fit
  on train only; RAG retrieval filters by `before_date`
- **Significance**: report Newey–West (HAC, lag=5) on daily PnL and
  Memmel (1996) ΔSharpe with block bootstrap (block_len=10, 5000 reps);
  also event-level Spearman + Mann-Whitney as a power-friendlier alternative

## Limitations

- N ≈ 220 OOS events is small for daily-Sharpe inference
- Universe is 50 large-caps; PEAD literature documents larger effects on
  smaller-cap names that we did not test
- LLM signals are 6 numeric scores; richer text representations
  (QA-only, evasion topic codes) were not explored

Extending the universe to Russell 1000 and the history to 2015 would
materially address the power problem, at the cost of more WRDS scraping
and OpenAI spend.
