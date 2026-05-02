# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research pipeline that combines **OpenAI GPT-4o + RAG + text-CNN** for active portfolio management on 50 US large-cap equities (2018–2025, ~1,500 WRDS earnings call transcripts). Workflow lives in five sequential notebooks (`notebooks/00`–`04`); shared logic lives in `src/llm_agent/`.

## Commands

```bash
pip install -e .                  # install (editable); also installs ANTHROPIC + OpenAI SDKs
pip install -e ".[dev]"           # adds jupyter + ruff
cp .env.example .env              # then fill OPENAI_API_KEY

jupyter lab notebooks/            # run 00 → 04 in order
python config.py                  # sanity-check paths and ticker count
ruff check .                      # lint (line-length 100, target py310)
```

There is no test suite; validation is via the notebooks (NB1 EDA, NB4 backtest metrics).

## Architecture

**Data flow (one-way, notebook-ordered):**

```
earningcall/*.zip  ──NB0──▶  data/clean/{earnings_sessions,earnings_segments,prices,labels}.parquet
                              │
                              ├──NB1── EDA only
                              ├──NB2── RAG (OpenAI embed cache → numpy cosine) + GPT-4o agent
                              │         → data/signals/llm_signals.parquet + evidence
                              ├──NB3── text-CNN trained on NB2 evidence
                              │         → data/signals/cnn_predictions.parquet
                              └──NB4── ablation backtest (Baseline / A / B / C / D)
```

**`src/llm_agent/` is the stable contract** between teammates — both notebooks import from it; do not duplicate path or split logic in notebooks:

- `data_loader.py` — the *only* sanctioned entry point for reading data. Adds project root to `sys.path` and re-exports `config`. Exposes `load_sessions / load_segments / load_prices / load_labels / load_llm_signals / load_single_transcript / split_by_period`.
- `transcript_parser.py` — splits raw transcripts into prepared remarks vs. Q&A pairs (NLP unit of analysis is the *segment*, not the call).
- `price_data.py` — yfinance fetch + forward / excess returns vs. SPY.
- `metrics.py` — Sharpe / annualised return / MDD / IR / AUC.

**`config.py` is the single source of truth** for paths, the 50-ticker universe (`TARGET_COMPANIES` with name-aliases for fuzzy matching, plus auto-built `NAME_TO_TICKER` reverse index), time splits, and model IDs. Always import from `config` rather than hard-coding.

**Splits are time-ordered and load-bearing** for the anti-leakage story:
- Train 2018–2022, Val 2023, Test₁ 2024 (bull), Test₂ 2025 (volatile)
- RAG retrieval **must** filter by `before_date`; prompt iteration uses train period only
- Backtest opens positions on the **second trading day** after the call; holding window `LABEL_HOLDING_DAYS = 10`, labels are excess-of-SPY by default

**Models:** main = `gpt-4o`, cheap = `gpt-4o-mini`, embeddings = `text-embedding-3-large`. The LLM call uses `response_format={"type":"json_object"}` for strict JSON. RAG uses plain numpy cosine similarity, not a vector DB.

## Local-environment notes

- `config.WRDS_ZIP_DIR` is hard-coded to a teammate's machine path (`/Users/yixuanwang/...`). Override locally before running NB0; everything downstream reads from `data/clean/` so it's only needed for the ETL re-run.
- Python pinned to `>=3.10,<3.12` because Intel Mac caps `torch` at 2.2.2.
- Ablation arms in NB4: B (XGBoost on technicals) is the comparator — **B vs C** isolates LLM-signal alpha, **B vs D** isolates CNN-on-evidence alpha. Don't change the comparator without also updating the writeup.
