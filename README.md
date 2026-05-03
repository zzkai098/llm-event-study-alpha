# LLM-Augmented Event-Study Portfolio Research

Combines **GPT-4o + RAG** with classical event-study econometrics and tree-based ML to test whether LLM-extracted signals from earnings call transcripts add incremental, *risk-adjusted* alpha over price-only technical features.

- **Universe:** 50 US large caps, 2018–2025
- **Data:** ~1,500 WRDS earnings call transcripts + yfinance daily prices + SPY benchmark
- **Splits:** Train 2018–2022 · Val 2023 · Test₁ 2024 (bull) · Test₂ 2025 (volatile)
- **Stack:** OpenAI `gpt-4o` + `text-embedding-3-large` (3072-d), numpy cosine RAG, scikit-learn / XGBoost / Optuna

---

## TL;DR — what the data actually shows

Six layers of evidence, ranked from weakest to strongest claim:

| # | Question | Verdict |
|---|----------|---------|
| 1 | Does LLM+Tech beat Tech-only on **headline Sharpe**? | Yes in both OOS years (Δ +0.13 / +1.07), but regime-dependent |
| 2 | Are those daily-PnL Sharpe gaps **statistically significant**? | **No** — Newey–West / Memmel / block-bootstrap all fail to clear p<0.05 (sample-size bound) |
| 3 | Is the cross-sectional ranking edge significant on **event-level** tests? | **Yes** — pooled Spearman ρ=+0.091 (p=0.042), Mann–Whitney long−short = +1.66% (p=0.029) |
| 4 | Is the LLM block a real driver under a **leakage-free permutation** audit? | **Yes** — LLM = 36.6% of pooled OOS importance (3.1× its impurity share); confirmed by refit-without-block ablation |
| 5 | Are all six GPT-4o features pulling weight? | **No** — `evasion` alone accounts for ~72% of the LLM block's contribution; the other five net to zero |
| 6 | Is the 2025 ΔSharpe = +1.07 the same evidence as the cross-sectional edge? | **No** — it is concentrated in a handful of extreme-`p` events; per-year ranking is actually *stronger in 2024* |

**Bottom line.** A weak-positive but underpowered finding, written up with explicit limitations rather than overstated significance. The robust scientific claim is *"management evasiveness on Q&A (a single GPT-4o signal) carries a real, statistically significant cross-sectional edge on post-call abnormal returns at the +2 to +21 day horizon"* — not *"six LLM signals add Sharpe."*

---

## Headline metrics (NB05 §6 — the full ablation table)

| Test period | Arm | Sharpe | Ann. Return | Vol | Max DD | IR vs SPY |
|-------------|-----|-------:|------------:|----:|-------:|----------:|
| **2024** (bull) | BuyHold_EW50 | **2.07** | +28.0% | 13.5% | −5.6% | +0.18 |
|             | Tech-only      | +0.30 | +3.7% | 12.2% | −9.1% | −1.07 |
|             | Event-Study    | +0.43 | +5.4% | 12.6% | −8.4% | −1.01 |
| **2025** (volatile) | BuyHold_EW50 | +0.86 | +14.5% | 16.9% | −17.1% | +0.05 |
|                  | Tech-only       | −0.13 | −1.6% | 12.4% | −11.0% | −1.20 |
|                  | **Event-Study** | **+0.95** | +9.1% | 9.6% | **−3.9%** | −0.42 |

Two regime takeaways:

- **2024 bull market.** Passive equal-weight is a wall (Sharpe 2.07). Active arms cannot out-pick a market where everything goes up — selectivity is a tax. ΔSharpe (ES − Tech) = +0.13: small, sign-consistent.
- **2025 volatile market.** Tech-only goes negative; Event-Study delivers Sharpe 0.95 ≈ BuyHold's 0.86 **at one-quarter the drawdown** (−3.9% vs −17.1%). ΔSharpe = +1.07. This is the headline, but it is also the most fragile number in the report — see §8 and the per-year diagnostics below.

---

## Why the daily-Sharpe story is *underpowered*, not absent (NB05 §8)

| Test | 2024 p | 2025 p |
|------|-------:|-------:|
| Newey–West HAC paired t (lag=5) | 0.93 | 0.45 |
| Memmel (1996) Sharpe-difference z | 0.91 | 0.40 |
| Circular block bootstrap 95% CI on ΔSharpe | [−2.2, +3.2] | [−2.2, +3.2] |

A 120-cell sensitivity scan over (entry day × hold window × weighting) yields **zero** cells passing Bonferroni-adjusted p. With ~250 trading days/year and event-driven autocorrelation, daily-PnL machinery cannot detect anything below a ~2-Sharpe gap. The +1.07 in 2025 is real-valued but not defensible at the daily level.

## Where the signal *does* clear significance — event-level tests (NB05 §9)

Switching the unit of analysis from **days (~500)** to **events (~360)** and the target to a longer **`car_2_21`** window (stricter PEAD check than the trained `car_2_11`):

| Test | Statistic | One-sided p |
|------|-----------|------------:|
| Spearman ρ(model `p`, CAR_2_21) | **+0.091** | **0.042** |
| Mann–Whitney U (top quintile vs bottom quintile of `p`) | long mean **+1.43%** vs short **−0.22%** (Δ = **+1.66%**) | **0.029** |

This is the **first defensibly significant result** in the project. The signal generalises *beyond* the trained horizon, ruling out training-window memorisation.

Per-year, the ranking story flips the daily-PnL story:

| | §6 ΔSharpe | §9 Spearman ρ | §9 long−short |
|---|----:|----:|----:|
| 2024 | +0.13 | **0.110** (p=0.062) | **+2.22%** (p=0.058) |
| 2025 | +1.07 | 0.073 (p=0.176) | +0.82% (p=0.232) |

The 2024 ranking edge is the genuine one; the 2025 daily-Sharpe edge comes from a few extreme-`p` events resolving favourably and being amplified by daily aggregation.

## How much of the model is the LLM doing? (NB05 §10 + §11.1)

Two independent attribution methods agree on the ranking:

| Block | OOS perm Δ AUC | perm % | NB04 impurity % | Refit drop-Δ AUC (§11.1) |
|-------|---------------:|------:|----------------:|-------------------------:|
| Tech (5 features) | 0.0073 | 47.0% | 84.8% | −0.038 |
| **LLM (6 features)** | **0.0056** | **36.6%** | 11.7% | **−0.030** |
| Sector (11 dummies) | 0.0025 | 16.4% | 3.5% | +0.002 |

- **Impurity importance is misleading** — it favours continuous high-cardinality features (Tech floats) and structurally underweights bounded-integer LLM signals.
- **Permutation** (shuffle features at predict time) and **refit-without** (retrain after removing the block) are different counterfactuals but agree: LLM is the second-largest driver, on par with Tech in OOS contribution.

## Which LLM features are actually pulling weight? (NB05 §11.2)

Drop-one-LLM ablation, refit RF on val 2023:

| Drop | val AUC | Δ vs full |
|------|--------:|----------:|
| full (all 6 LLM) | 0.6012 | — |
| sentiment | 0.6022 | +0.001 |
| confidence | 0.6070 | +0.006 |
| certainty | 0.6118 | +0.011 |
| **evasion** | **0.5799** | **−0.021** |
| guidance | 0.6025 | +0.001 |
| tone | 0.6039 | +0.003 |

**Only `evasion` is doing work.** It accounts for ~72% (−0.0213 of −0.0298) of the entire LLM block's contribution. The other five could be dropped in production: prompt simplifies, inference cost drops, val AUC slightly *rises*. The paper's honest framing is **management evasiveness on Q&A predicts post-call drift**, not **six LLM signals add alpha**.

---

## Pipeline (6 sequential notebooks)

```
00_etl.ipynb              # WRDS zips → parquet (one-off; needs WRDS access)
01_eda.ipynb              # Sanity-check the cleaned tables; equal-weight bar to clear
02_rag_llm_agent.ipynb    # RAG retrieval + GPT-4o → 6 signals + evidence chunks
03_car_targets.ipynb      # Market-model CAR labels (4 windows: 2-3, 2-6, 2-11, 2-21)
04_ml_models.ipynb        # 4-model bake-off + Optuna RF on car_2_11 → event_study_best.pkl
05_backtest_ablation.ipynb # 3-arm backtest + significance tests + permutation + ablation
```

NB05 self-contains the analysis previously spread across notebooks 06–09: §9 event-level rank tests, §10 permutation importance, §11.1 block ablation, §11.2 drop-one LLM ablation, §12 conclusion, §13 limitations.

`src/llm_agent/` holds the reusable contract (data loaders, feature builders, event-study utilities, models, metrics, importance, event-tests, ablation) — both notebooks and scripts import from there.

## Project structure

```
llm-portfolio/
├── config.py                       # Paths, ticker universe, time splits, model IDs
├── pyproject.toml
├── .env.example                    # Copy to .env, fill OPENAI_API_KEY
├── src/llm_agent/
│   ├── data_loader.py              # Unified data access (load_*, split_by_period)
│   ├── transcript_parser.py        # Prepared remarks vs Q&A segmentation
│   ├── price_data.py               # yfinance + market-model utilities
│   ├── features.py                 # Tech / sector / LLM features (anti-leakage at function level)
│   ├── models.py                   # Sklearn pipelines for logreg / SVM / RF / XGB
│   ├── metrics.py                  # Sharpe / IR / MDD / AUC
│   ├── importance.py               # OOS permutation + block summary
│   ├── event_tests.py              # Spearman + Mann–Whitney on quintile thresholds
│   ├── bakeoff.py                  # 4-model GridSearchCV used by NB04
│   └── ablation.py                 # Block + drop-one ablation refit utilities
├── notebooks/                      # 00 → 05 in order
├── data/
│   ├── clean/                      # earnings_*, prices, car_labels.parquet (committed)
│   └── signals/                    # llm_signals.parquet, evidence.parquet (committed)
├── models/event_study_best.pkl     # Optuna-tuned RF on car_2_11 (committed)
└── output/                         # tables/ + figures/ (gitignored, regenerated by NB05)
```

## Quick start

```bash
pip install -e .                  # editable install, pulls OpenAI + Anthropic SDKs
cp .env.example .env              # fill OPENAI_API_KEY
jupyter lab notebooks/            # run 03 → 05 to reproduce backtest end-to-end
python config.py                  # sanity-check paths + ticker count
```

`data/clean/*.parquet`, `data/signals/llm_signals.parquet`, and `models/event_study_best.pkl` are committed, so **NB03 → NB05 reproduces from scratch in ~3 minutes without any API spend**. Re-running NB00 needs WRDS access; re-running NB02 needs an OpenAI key and ~$30–50 of API budget for the full 1,500-transcript pass.

## Three-arm backtest design (NB05)

| Arm | Features | Purpose |
|-----|----------|---------|
| **A · BuyHold_EW50** | none (passive equal-weight 50 names) | Floor — the bar to clear (SPY is too easy because the 50-name roster pre-selects winners) |
| **B · Tech-only** | 5 technicals + 11 sector dummies | Fair comparator — same Optuna RF, LLM features removed |
| **C · Event-Study (LLM+Tech)** | B + 6 GPT-4o signals | Full feature set — main candidate |

**B vs C** isolates LLM-signal alpha (same algorithm, same hyperparameters, only the 6 LLM features differ).

**Trade rule.** Open at `event + ENTRY_OFFSET` trading days, hold `HOLDING_DAYS`, equal-weight overlapping positions, weight by `2·(p − 0.5)` clipped to ±1. Long/short thresholds are val-2023 quintile boundaries, **frozen for test**.

## Methodology choices worth noting

- **Target = market-model CAR\_[+2, +11]** with α, β estimated on a [−120, −20] estimation window per event (MacKinlay 1997 framework; OLS market model with SPY proxy)
- **Tail filter** on labels: drop train/val events with |CAR| < 0.5·σ_e·√10 to remove no-information observations. **Test sets are never filtered** — the OOS evaluation is unconditional
- **Anti-leakage**: scaler / winsorisation / sector-dummy alignment all fit on train only; RAG retrieval filters by strict `before_date <`; technicals computed with `< event_date` not `≤`
- **Significance stack**: report Newey–West HAC (lag=5) on daily PnL, Memmel (1996) ΔSharpe z, circular block bootstrap (block_len=10, 5000 reps), event-level Spearman + Mann–Whitney as the power-friendlier alternative

## Limitations

- **Sample size.** Two OOS years (~500 days, ~360 events) is genuinely short. The wide §8 ΔSharpe CI is a sample-size problem first; 5+ years of OOS would do more for the headline than any modelling change.
- **Token / cost budget.** Each full GPT-4o pass costs ~$30–50, which capped how many prompt variants could be tried. One clean end-to-end pipeline was prioritised over wider iteration.
- **Prompt is too generic.** A single fixed-schema prompt is reused for every call. A call-conditional prompt — using prior-quarter guidance, the actual analyst questions, sector-specific risk vocabulary — would likely extract more. The §11.2 finding that only `evasion` carries weight is partial evidence that the generic prompt leaves signal on the table.
- **RAG is doing the minimum.** Plain cosine similarity over chunk embeddings with a `before_date` filter. Hybrid retrieval, reranking, or letting the agent issue its own queries are open follow-ups.
- **Universe is 50 large-caps.** PEAD literature documents larger effects on smaller-cap names that we did not test. Extending to Russell 1000 would materially address the power problem.

## Paper-section ↔ notebook mapping

| Paper section | NB05 section |
|---------------|--------------|
| Headline ablation table | §6 |
| Significance (HAC, Memmel, bootstrap) | §8 |
| Event-level rank tests (Appendix E.4) | §9 |
| OOS permutation importance (Appendix D.3) | §10 |
| Block ablation, drop-one LLM (Appendix D.1, D.2) | §11.1, §11.2 |
| Conclusion (six layers of evidence) | §12 |
| Limitations & future work | §13 |

NB04 covers the model-selection bake-off and Optuna verification (Appendix C); NB03 covers the event-study methodology and CAR construction.

## References

- Ball & Brown (1968), Fama, Fisher, Jensen & Roll (1969), Brown & Warner (1985) — event-study foundations
- MacKinlay (1997, *JEL*) — definitive event-study methodology review
- Memmel (2003) — Sharpe-difference test
- Newey & West (1987) — HAC standard errors
- Politis & Romano (1994) — circular block bootstrap
