# LLM-Powered Active Portfolio Management Agent

A research project that combines **OpenAI GPT-4o + RAG + text-CNN** for active
portfolio management on US large-cap equities.

## Project Structure

```
llm-portfolio/
├── config.py                     # Global config (paths / models / RAG / backtest)
├── pyproject.toml
├── .env.example                  # Copy to .env and fill in API keys
├── src/llm_agent/
│   ├── data_loader.py            # Unified data-access interface
│   ├── transcript_parser.py      # Split transcript into prepared remarks + Q&A pairs
│   ├── price_data.py             # yfinance + forward / excess returns
│   └── metrics.py                # Sharpe / annualised return / MDD / IR / AUC
├── notebooks/
│   ├── 00_etl.ipynb              # One-off ETL (zip → parquet)  (Teammate B)
│   ├── 01_eda.ipynb              # EDA on the clean tables      (Teammate B)
│   ├── 02_rag_llm_agent.ipynb    # RAG (OpenAI embed) + GPT-4o  (Teammate A)
│   ├── 03_cnn_on_evidence.ipynb  # text-CNN on RAG evidence     (Teammate A)
│   └── 04_backtest.ipynb         # Ablation backtest            (Teammate B)
├── reference/
│   └── Mutual_fund_chat_with_OpenAI_MF815.ipynb   # template for NB2
├── earningcall/                  # raw WRDS zips (input to NB0)
└── data/
    ├── clean/                    # earnings_*.parquet, prices, labels
    ├── embeddings/               # OpenAI embedding cache (.npy)
    └── signals/                  # llm_signals + evidence + cnn_predictions
```

## Ownership

| Teammate | Notebooks | Scope |
|---|---|---|
| **A** | NB2, NB3 | Transcript parsing, RAG retrieval, GPT-4o agent prompt + signal extraction, text-CNN training on RAG evidence |
| **B** | NB0, NB1, NB4 | ETL, EDA, label construction, ablation backtest, evaluation figures |

Shared code in `src/llm_agent/` is reused by both sides; treat it as a stable contract.

## Data

- **Companies**: 50 US large caps (S&P 500 core, balanced across 11 GICS sectors)
- **Time range**: 2018-01 to 2025-12 (~31 quarters)
- **Transcripts**: ~1,500 earnings calls (WRDS Capital IQ)
- **Splits**: Train 2018–2022 / Val 2023 / Test 2024 (bull) / Test 2025 (volatile)

## Tech Stack

| Purpose | Choice |
|---|---|
| LLM signal extraction | **OpenAI** (`gpt-4o` main / `gpt-4o-mini` cheap) with JSON mode |
| Embedding | OpenAI `text-embedding-3-large` |
| RAG retrieval | numpy cosine similarity (no vector DB needed) |
| Supervised models | text-CNN (PyTorch) + XGBoost |
| Backtest | In-house engine + `metrics.py` |

## Quick Start

```bash
# 1. Install
pip install -e .
cp .env.example .env   # fill in OPENAI_API_KEY

# 2. Launch Jupyter and run notebooks/00 → 04 in order
#    NB0 builds data/clean/*.parquet (already done; only re-run if config changes)
jupyter lab notebooks/
```

## Ablation Design

| Arm | Strategy | Purpose |
|---|---|---|
| Baseline | Buy-and-hold SPY | Market benchmark |
| Strategy A | 50-stock equal weight | Stock-selection benchmark |
| Strategy B | Technical indicators + XGBoost | Traditional quant benchmark |
| Strategy C | B + GPT-4o LLM signals (NB2) | Incremental alpha from LLM |
| Strategy D | B + text-CNN evidence (NB3) | Incremental alpha from supervised text model |

B vs C / B vs D quantifies the value added by text-derived signals.

## Anti-leakage Guarantees

- RAG retrieval strictly filters by `before_date`
- Prompt iteration uses **only** the train period (2018–2022)
- Backtest uses time-ordered splits; positions opened on the second trading day
  after the call
