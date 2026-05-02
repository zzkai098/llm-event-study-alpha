"""
Global configuration
====================
Constants and paths shared across all pipeline scripts and notebooks.
"""

import os
from pathlib import Path

# ---------- Paths ----------
# Project root (directory containing this config.py)
PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                       # raw WRDS zip files
TRANSCRIPTS_DIR = DATA_DIR / "transcripts_us"    # filtered single-call JSON
CLEAN_DIR = DATA_DIR / "clean"                   # parquet / csv tables
SIGNALS_DIR = DATA_DIR / "signals"               # LLM signals + CNN predictions
EMBEDDINGS_DIR = DATA_DIR / "embeddings"         # OpenAI embedding cache (.npy)

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# External data — WRDS zips. Set WRDS_ZIP_DIR in .env (or shell) to override.
# Defaults to ./earningcall under the project root.
WRDS_ZIP_DIR = Path(os.environ.get("WRDS_ZIP_DIR", PROJECT_ROOT / "earningcall"))

# ---------- Time range ----------
START_YEAR = 2018
END_YEAR = 2025

# Train / Val / Test splits
TRAIN_YEARS = list(range(2018, 2023))  # 2018-2022
VAL_YEARS = [2023]
TEST_YEARS_1 = [2024]
TEST_YEARS_2 = [2025]

# Pull prices one extra quarter back so technical indicators have a warm-up window
PRICE_START_DATE = "2017-10-01"
PRICE_END_DATE = "2026-04-23"

# ---------- 50 target companies ----------
TARGET_COMPANIES = {
    # Information Technology (7)
    "AAPL":  ["Apple Inc"],
    "MSFT":  ["Microsoft Corporation", "Microsoft Corp"],
    "NVDA":  ["NVIDIA Corporation", "NVIDIA Corp"],
    "AVGO":  ["Broadcom Inc", "Broadcom Limited"],
    "ORCL":  ["Oracle Corporation", "Oracle Corp"],
    "CRM":   ["Salesforce, Inc", "Salesforce.com", "salesforce.com, inc"],
    "ADBE":  ["Adobe Inc", "Adobe Systems"],
    # Communication Services (5)
    "GOOGL": ["Alphabet Inc"],
    "META":  ["Meta Platforms", "Facebook, Inc"],
    "NFLX":  ["Netflix, Inc", "Netflix Inc"],
    "DIS":   ["Walt Disney Company", "Walt Disney Co"],
    "TMUS":  ["T-Mobile US"],
    # Consumer Discretionary (6)
    "LOW":   ["Lowe's Companies", "Lowes Companies"],
    "TSLA":  ["Tesla, Inc", "Tesla Inc", "Tesla Motors"],
    "HD":    ["Home Depot, Inc", "Home Depot Inc"],
    "MCD":   ["McDonald's Corporation", "McDonald's Corp"],
    "NKE":   ["NIKE, Inc", "Nike, Inc"],
    "SBUX":  ["Starbucks Corporation", "Starbucks Corp"],
    # Consumer Staples (4)
    "WMT":   ["Walmart Inc", "Wal-Mart Stores"],
    "PG":    ["Procter & Gamble Company", "Procter & Gamble Co"],
    "KO":    ["Coca-Cola Company", "Coca-Cola Co"],
    "COST":  ["Costco Wholesale"],
    # Health Care (6)
    "UNH":   ["UnitedHealth Group"],
    "JNJ":   ["Johnson & Johnson"],
    "LLY":   ["Eli Lilly and Company", "Eli Lilly & Co"],
    "PFE":   ["Pfizer Inc"],
    "ABBV":  ["AbbVie Inc"],
    "MRK":   ["Merck & Co"],
    # Financials (6)
    "JPM":   ["JPMorgan Chase", "JP Morgan Chase"],
    "BAC":   ["Bank of America Corporation", "Bank of America Corp"],
    "WFC":   ["Wells Fargo & Company", "Wells Fargo & Co"],
    "GS":    ["Goldman Sachs Group"],
    "MS":    ["Morgan Stanley"],
    "SCHW":  ["Charles Schwab Corporation", "Charles Schwab Corp"],
    # Industrials (4)
    "CAT":   ["Caterpillar Inc"],
    "BA":    ["Boeing Company", "Boeing Co"],
    "HON":   ["Honeywell International"],
    "UPS":   ["United Parcel Service"],
    # Energy (3)
    "XOM":   ["Exxon Mobil", "ExxonMobil"],
    "CVX":   ["Chevron Corporation", "Chevron Corp"],
    "COP":   ["ConocoPhillips"],
    # Utilities (2)
    "SO":    ["Southern Company"],
    "DUK":   ["Duke Energy"],
    # Real Estate (2)
    "PLD":   ["Prologis, Inc", "Prologis Inc"],
    "AMT":   ["American Tower"],
    # Materials (2)
    "LIN":   ["Linde plc", "Linde PLC"],
    "SHW":   ["Sherwin-Williams"],
    # Extras (3)
    "V":     ["Visa Inc"],
    "NOW":   ["ServiceNow, Inc", "ServiceNow Inc"],
    "CMG":   ["Chipotle Mexican Grill"],
}

# Reverse lookup: company name -> ticker (used by all scripts)
NAME_TO_TICKER = {}
for ticker, name_list in TARGET_COMPANIES.items():
    for name in name_list:
        NAME_TO_TICKER[name.lower()] = ticker

TICKERS = list(TARGET_COMPANIES.keys())
BENCHMARK = "SPY"

# GICS sector for each ticker (manually maintained alongside TARGET_COMPANIES groups above)
SECTOR_MAP = {
    "AAPL": "InfoTech", "MSFT": "InfoTech", "NVDA": "InfoTech", "AVGO": "InfoTech",
    "ORCL": "InfoTech", "CRM": "InfoTech", "ADBE": "InfoTech", "NOW": "InfoTech",
    "GOOGL": "CommServices", "META": "CommServices", "NFLX": "CommServices",
    "DIS": "CommServices", "TMUS": "CommServices",
    "LOW": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "NKE": "ConsDisc", "SBUX": "ConsDisc", "CMG": "ConsDisc",
    "WMT": "ConsStaples", "PG": "ConsStaples", "KO": "ConsStaples", "COST": "ConsStaples",
    "UNH": "HealthCare", "JNJ": "HealthCare", "LLY": "HealthCare", "PFE": "HealthCare",
    "ABBV": "HealthCare", "MRK": "HealthCare",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "SCHW": "Financials", "V": "Financials",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SO": "Utilities", "DUK": "Utilities",
    "PLD": "RealEstate", "AMT": "RealEstate",
    "LIN": "Materials", "SHW": "Materials",
}
SECTOR_LIST = sorted(set(SECTOR_MAP.values()))
assert set(SECTOR_MAP.keys()) == set(TICKERS), "SECTOR_MAP must cover all TICKERS"

# ---------- Event study ----------
EVENT_WINDOWS = {
    "car_2_3":  (2, 3),
    "car_2_6":  (2, 6),
    "car_2_11": (2, 11),
    "car_2_21": (2, 21),
}
ESTIMATION_WINDOW = (-120, -20)  # market-model OLS window relative to event_date

# ---------- Label construction ----------
LABEL_HOLDING_DAYS = 10   # holding window after the earnings call (trading days)
LABEL_USE_EXCESS = True   # use return excess of SPY rather than raw return

# ---------- LLM ----------
# OpenAI — used for both signal extraction and embeddings
OPENAI_MODEL_MAIN = "gpt-4o"
OPENAI_MODEL_CHEAP = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024

OPENAI_EMBED_MODEL = "text-embedding-3-large"

# RAG chunking
CHUNK_WORDS = 200
CHUNK_OVERLAP = 50
RAG_TOP_K = 5

# ---------- Backtest ----------
TRANSACTION_COST = 0.001     # 0.1% per side
POSITION_SIZE_CAP = 0.10     # max 10% per name

# ---------- Logging ----------
LOG_LEVEL = "INFO"


if __name__ == "__main__":
    # Quick config sanity check
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"TICKERS      = {len(TICKERS)} companies")
    print(f"Time range   = {START_YEAR}-{END_YEAR}")
