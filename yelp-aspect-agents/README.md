# Yelp Aspect Agents

Multi-agent sentiment analysis of Yelp restaurant reviews along **three dimensions**: **food**,
**service** and **ambience**. The LLM agents only make categorical judgements. **Every number is
computed in Python** and checked against the reviewers' star ratings. A web UI (FastAPI + a
single-page frontend) displays the results; the browser only renders numbers the Python API returns.

This builds on the *Prompting LLMs for Sentiment Analysis* notebook (Module 7). The notebook covered
zero-shot and few-shot prompting and chaining LLMs; this project extends those ideas to one agent per
aspect.

```
            ┌ food agent ─────┐   quotes +     ┌────────── Python ──────────┐
review ───► ├ service agent ──┤ ─ polarity + ─►│ ground quotes in the text  │─► arbiter agent ─► lead agent ─► Python: overall score,
            └ ambience agent ─┘   intensity    │ score = mean(sign·int/3)   │   (only on        (overall             rule stars, calibrated
               (parallel)                      └────────────────────────────┘    overlaps)         polarity)            stars (ridge, CV)
```

## How the scores are computed (all Python, `packages/yelp-scoring`)

| Step | Rule |
|---|---|
| Agent output | Verbatim quotes, each with polarity (positive / negative / neutral) and intensity (1 mild, 2 clear, 3 strong). No numbers. |
| Grounding | Each quote must appear in the review. The check normalises case, quote marks and whitespace, then allows a near-exact match (≥ 85%). A quote needs at least 2 words, and `...` elisions are checked fragment by fragment. Quotes that aren't found are **excluded** and shown in the UI. |
| Arbiter | If two aspects cite overlapping text, an arbiter agent decides which aspect the passage belongs to. The losing aspect is rescored. |
| Aspect score | `mean(sign(polarity) × intensity / 3)` over grounded, de-duplicated mentions, in [−1, 1]. The score is 0 and the label "not mentioned" when there are none. |
| Overall score | `sign(lead polarity) × lead intensity / 3`. Rule-based stars are `3 + 2 × overall`. |
| Calibrated stars | Ridge regression on [food, service, ambience, 3 "mentioned" flags, overall]. It is evaluated with 5-fold `cross_val_predict`, so no review is scored by a model that saw it. |
| Metrics | scikit-learn `classification_report(y_true, y_pred)`, `confusion_matrix`, MAE and RMSE, plus scipy Pearson and Spearman. Accuracy has a Wilson 95% CI and macro-F1 a bootstrap 95% CI. |

Ground truth is the star rating only (Yelp has no aspect labels): 1–2★ = negative, 3★ = neutral,
4–5★ = positive.

## Monorepo layout (uv workspace)

| Path | Package | Role |
|---|---|---|
| `packages/yelp-core` | `yelp_core` | Data contracts, settings, LLM clients (NVIDIA NIM, Hugging Face, Ollama, offline mock) |
| `packages/yelp-scoring` | `yelp_scoring` | Grounding, scoring formulas, cross-validated star model |
| `packages/yelp-data` | `yelp_data` | Reservoir-samples reviews per star rating straight out of `yelp_dataset.tar` (scans all ~7M reviews in ~20 s, extracts nothing) |
| `packages/yelp-agents` | `yelp_agents` | Aspect agents ×3, arbiter, lead agent, orchestrator (parallel, resumable) |
| `packages/yelp-eval` | `yelp_eval` | Metrics with confidence intervals; 3-D plot for the CLI |
| `apps/cli` | `yelp-agents` | `providers`, `sample`, `run`, `eval`, `viz` |
| `apps/web` | `yelp-web` | FastAPI API + web UI (Analyze / Explore / Metrics) |

## Quick start

```bash
uv sync
uv run yelp-agents providers            # which LLM backend will be used
uv run yelp-agents sample -n 100        # 20 reviews per star rating, drawn from the whole file
uv run yelp-agents run                  # agent team over the sample (resumable)
uv run yelp-agents eval                 # accuracy report
uv run yelp-web                         # http://127.0.0.1:8000
```

Other useful options:
- `sample --city "New Orleans"`: gives meaningful per-business profiles.
- `run --zero-shot`: compare against the few-shot prompts.
- `run --aggregator-model <id>`: put the lead agent on a different model.
- `--provider mock`: run with no model at all.

## The UI

* **Analyze**: the agents run live on a pasted or random Yelp review, streamed as each one finishes.
  Each aspect card shows its verified quotes with their values, the arithmetic behind the score
  (e.g. `(+3 +2 +2) / (3 × 3) = +0.78`), and any excluded quotes with the reason. You also see arbiter
  decisions, the lead agent's rule-based stars next to the calibrated model and the real rating, and
  the quotes highlighted in the review.
* **Explore**: a 3-D scatter of reviews on the food, service and ambience axes, coloured by actual
  rating or predicted sentiment. Click a point or a table row for detail, including the out-of-fold
  calibrated stars. You can start batch runs from here.
* **Metrics**: accuracy with a 95% CI, macro-F1, star error for the rule and the calibrated model,
  quote grounding rate, confusion matrix, per-class precision/recall/F1, per-dimension mention rate
  and Spearman ρ vs stars, and a business table.

## LLM backends

With `LLM_PROVIDER=auto`, the first available backend is used:

1. **ASU** (`https://openai.rc.asu.edu/v1`): uses `API_KEY` (or `ASU_API_KEY`) from `.env`. The default
   model is `gemma4-31b-it`; set `ASU_MODEL` to change it. JSON mode is on, and rate limits (HTTP 429)
   are retried after the server's `Retry-After`.
2. **NVIDIA NIM**: `NVIDIA_API_KEY`
3. **Hugging Face**: `HF_TOKEN`
4. **Ollama** (local): `gemma4:latest`
5. **Offline keyword mock**: used by the tests

The UI's **Model** menu can also switch between ASU models (`qwen3-235b-a22b-instruct-2507`,
`llama4-maverick-17b`) for comparison. `.env` is git-ignored, and the key is never logged or shown.

## Tests

```bash
uv run pytest    # grounding, formulas, CV leakage, metrics vs hand-computed values, agents, web API
```
