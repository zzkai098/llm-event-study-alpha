# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research pipeline that combines **OpenAI GPT-4o + RAG** for active portfolio management on 50 US large-cap equities (2018–2025, ~1,500 WRDS earnings call transcripts). Workflow lives in six sequential notebooks (`notebooks/00`–`05`) plus four interpretation notebooks (`notebooks/06`–`09`); shared logic lives in `src/llm_agent/`.

## Commands

```bash
pip install -e .                  # install (editable); also installs ANTHROPIC + OpenAI SDKs
pip install -e ".[dev]"           # adds jupyter + ruff
cp .env.example .env              # then fill OPENAI_API_KEY

jupyter lab notebooks/            # run 00 → 05 in order; 06–09 are independent
python config.py                  # sanity-check paths and ticker count
ruff check .                      # lint (line-length 100, target py310)
```

There is no test suite; validation is via the notebooks (NB1 EDA, NB5 backtest metrics, NB6 importance audit, NB7 event-level rank tests, NB8 model bake-off reproduction, NB9 feature ablations).

## Architecture

**Data flow (one-way, notebook-ordered):**

```
earningcall/*.zip  ──NB0──▶  data/clean/{earnings_sessions,earnings_segments,prices,labels}.parquet
                              │
                              ├──NB1── EDA only
                              ├──NB2── RAG (OpenAI embed cache → numpy cosine) + GPT-4o agent
                              │         → data/signals/{llm_signals,evidence}.parquet
                              ├──NB3── market-model CAR labels
                              │         → data/clean/car_labels.parquet
                              ├──NB4── 4-model bake-off + Optuna RF tuning
                              │         → models/event_study_best.pkl
                              ├──NB5── three-arm ablation backtest (BuyHold_EW50 / Tech / LLM+Tech)
                              ├──NB6── permutation importance audit on NB4's RF
                              │         → output/tables/perm_importance_*.csv
                              ├──NB7── event-level Spearman + Mann-Whitney rank tests
                              │         → output/tables/event_tests_*.csv
                              ├──NB8── 4-model bake-off reproduction (verifies pkl end-to-end)
                              │         → output/tables/bakeoff_summary.csv
                              └──NB9── feature ablations (block-level + drop-one LLM)
                                        → output/tables/{block,drop_one_llm}_ablation.csv
```

NB6–NB9 are independent audit notebooks that consume `models/event_study_best.pkl` plus `data/clean/`; none of them write back to `data/` or `models/`. Each maps to one paper appendix:

- NB6 → Appendix D.3 (permutation importance)
- NB7 → Appendix E.4 (event-level rank tests)
- NB8 → Appendix C   (model selection: bake-off + Optuna verification)
- NB9 → Appendix D.1, D.2 (block ablation, drop-one LLM ablation)

**`src/llm_agent/` is the stable contract** between teammates — both notebooks import from it; do not duplicate path or split logic in notebooks:

- `data_loader.py` — the *only* sanctioned entry point for reading data. Adds project root to `sys.path` and re-exports `config`. Exposes `load_sessions / load_segments / load_prices / load_labels / load_llm_signals / load_single_transcript / split_by_period`.
- `transcript_parser.py` — splits raw transcripts into prepared remarks vs. Q&A pairs (NLP unit of analysis is the *segment*, not the call).
- `price_data.py` — yfinance fetch + forward / excess returns vs. SPY.
- `features.py` — per-event feature matrix for the three ablation arms (A/B/C). Anti-leakage is enforced at the function level (`_technicals_at_event` uses `<` not `<=` on event_date).
- `models.py` — sklearn pipeline factory for the NB4 bake-off (logreg / SVM / RF / XGB). Uses `TimeSeriesSplit`, never plain `KFold`.
- `metrics.py` — Sharpe / annualised return / MDD / IR / AUC.
- `importance.py` — permutation importance + block summary utilities used by NB6. Relies on `features.build_feature_matrix` as single source of truth — does not rebuild the matrix.
- `event_tests.py` — event-level Spearman ρ and Mann-Whitney U used by NB7. Quintile thresholds are frozen on val 2023 to match NB5's trade rule; the default evaluation window is `CAR_2_21` (longer than the trained `car_2_11`, as a stricter PEAD check).
- `bakeoff.py` — 4-model GridSearchCV comparison + per-family CV/val scoring used by NB8. Imports the model factory from `models.py` rather than redefining the grid.
- `ablation.py` — block-level and drop-one feature ablations used by NB9. Refits RF with the pkl's Optuna-tuned hyperparameters on each feature subset, isolating the contribution of features (not hyperparameters).

**`config.py` is the single source of truth** for paths, the 50-ticker universe (`TARGET_COMPANIES` with name-aliases for fuzzy matching, plus auto-built `NAME_TO_TICKER` reverse index), time splits, and model IDs. Always import from `config` rather than hard-coding.

**Splits are time-ordered and load-bearing** for the anti-leakage story:
- Train 2018–2022, Val 2023, Test₁ 2024 (bull), Test₂ 2025 (volatile)
- RAG retrieval **must** filter by `before_date`; prompt iteration uses train period only
- Backtest opens positions on the **second trading day** after the call; holding window `LABEL_HOLDING_DAYS = 10`, labels are excess-of-SPY by default
- Tail filter `|CAR| > 0.5·σ_e·√10` applies to train/val only — test sets are never filtered

**Models:** main = `gpt-4o`, cheap = `gpt-4o-mini`, embeddings = `text-embedding-3-large`. The LLM call uses `response_format={"type":"json_object"}` for strict JSON. RAG uses plain numpy cosine similarity, not a vector DB.

**Three-arm ablation in NB5:**
- **A: BuyHold_EW50** — passive equal-weight benchmark (the bar to clear; SPY is too easy because the 50-name roster pre-selects winners)
- **B: Tech-only** — XGBoost/RF on 5 technicals + 11 sector dummies
- **C: Event-Study (LLM+Tech)** — B + 6 GPT-4o-extracted signals
- **B vs C** isolates LLM-signal alpha. Don't change the comparator without also updating the writeup.

**Outputs convention:**
- Analysis artifacts go to `output/tables/` (CSVs) or `output/figures/` (PNG/PDF). Both are gitignored — anything reproducible from notebook + committed data lives there.
- Trained models go to `models/` (committed if small and load-bearing, e.g. `event_study_best.pkl`).
- `data/clean/` is read-only input; never write derived analysis there.

## Local-environment notes

- `config.WRDS_ZIP_DIR` is hard-coded to a teammate's machine path (`/Users/yixuanwang/...`). Override locally before running NB0; everything downstream reads from `data/clean/` so it's only needed for the ETL re-run.
- Python pinned to `>=3.10,<3.12` because Intel Mac caps `torch` at 2.2.2.
- NB2 batch run costs ~\$30–50 in OpenAI spend for the full 1,500-transcript pass. `data/signals/llm_signals.parquet` and `data/signals/evidence.parquet` are committed so NB3–NB9 reproduce without rerunning NB2.
