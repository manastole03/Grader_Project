# Yelp Aspect Agents

Multi-agent sentiment analysis of Yelp restaurant reviews along **three dimensions**: **food**,
**service** and **ambience**. The LLM agents only make categorical judgements. **Every number is
computed in Python** and checked against the reviewers' star ratings. A web UI (FastAPI + a
single-page frontend) displays the results; the browser only renders numbers the Python API returns.

This builds on the *Prompting LLMs for Sentiment Analysis* notebook (Module 7). The notebook covered
zero-shot and few-shot prompting and chaining LLMs; this project extends those ideas to one agent per
aspect.

**Demo guide:** [Yelp Aspect Agents: Live Demo Guide](https://claude.ai/code/artifact/7145e65d-09c9-47c4-8a76-d778d4ced031)
has the run of show for presenting this project, talking points, results, likely questions, and setup
steps for students who want to run it afterwards.

**Colab notebook:** [`notebooks/yelp_aspect_agents_colab.ipynb`](notebooks/yelp_aspect_agents_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/manastole03/Grader_Project/blob/main/yelp-aspect-agents/notebooks/yelp_aspect_agents_colab.ipynb)) runs the whole pipeline in one notebook with the same model, prompts and formulas.

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

## Tech stack

| Layer | Technology | Where | How it's used |
|---|---|---|---|
| Language | Python ≥ 3.10 (developed on 3.12) | everywhere | All logic, scoring and statistics |
| Workspace and packaging | [uv](https://docs.astral.sh/uv/) workspace, hatchling | `pyproject.toml` at the root and in each package | `uv sync` installs the 5 packages and 2 apps in editable mode; `uv run` runs commands in that environment |
| LLM | ASU Voyager LLM API (OpenAI-compatible), model `gemma4-31b-it` | `packages/yelp-core/src/yelp_core/llm.py` | Plain HTTPS calls with `requests` to `/v1/chat/completions` in JSON mode, temperature 0.1, with retries. NVIDIA NIM, the Hugging Face router, local Ollama and an offline mock are alternatives. |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | `packages/yelp-agents/src/yelp_agents/orchestrator.py` | The 3 aspect agents run in parallel per review, and `--workers` reviews run at once |
| Text matching | `difflib.SequenceMatcher` | `packages/yelp-scoring/src/yelp_scoring/grounding.py` | Checks that each quoted phrase is really in the review (near-exact, ≥ 85%) |
| Data | Yelp Open Dataset (`yelp_dataset.tar`), `tarfile` streaming, JSONL files | `packages/yelp-data/src/yelp_data/sampler.py` | One streaming pass with reservoir sampling; samples and results are saved as JSON lines |
| Statistics and ML | NumPy, pandas, scikit-learn, SciPy | `packages/yelp-scoring/.../star_model.py`, `packages/yelp-eval/` | Ridge regression with `cross_val_predict`; classification metrics; Pearson and Spearman; F distribution for ANOVA; ICC(1) |
| Charts | Plotly (Python) and plotly.js 2.35.2 (CDN); Matplotlib in the notebook | `packages/yelp-eval/.../figures.py`, `apps/web/.../static/app.js` | 3-D aspect-space scatter for `yelp-agents viz` and the Explore tab |
| Command line | `argparse` | `apps/cli/src/yelp_cli/main.py` | The `yelp-agents` command and its subcommands |
| Web backend | FastAPI, Uvicorn, Pydantic, Server-Sent Events | `apps/web/src/yelp_web/server.py` | JSON API; `/api/analyze` streams each agent's result as it finishes |
| Web frontend | HTML, CSS and plain JavaScript (no framework or build step) | `apps/web/src/yelp_web/static/` | Single page with Analyze, Explore, Compare and Metrics tabs; light and dark themes; only renders numbers the API returns |
| Tests | pytest, httpx (FastAPI `TestClient`) | `tests/` | 40 tests with the offline mock: no API key or dataset needed |
| Notebook | Google Colab, the `openai` client library, Colab Secrets, Google Drive | `notebooks/yelp_aspect_agents_colab.ipynb` | The same pipeline in one notebook for teaching |

Installed versions at the last run: FastAPI 0.141, Uvicorn 0.54, Pydantic 2.13, requests 2.34, NumPy 2.5,
pandas 3.0, scikit-learn 1.9, SciPy 1.18, Plotly 7.1, pytest 9.1.

## Codebase walkthrough

```
yelp-aspect-agents/
├── pyproject.toml            uv workspace root: lists the 7 members, dev tools (pytest, httpx)
├── .env.example              copy to .env and add your Voyager key (git-ignored)
├── packages/                 libraries; each layer imports only the layers below it
│   ├── yelp-core/            bottom layer, imported by everything
│   │   ├── schemas.py        Review, Mention, AspectResult, OverallResult, ReviewAnalysis
│   │   ├── config.py         Settings.from_env(): keys, models, temperature, retries, .env reader
│   │   └── llm.py            LLMClient + ASU/NVIDIA/HF, Ollama and mock clients; build_llm(); extract_json()
│   ├── yelp-scoring/         pure Python, no LLM
│   │   ├── grounding.py      ground_quote(): is the quote really in the review?
│   │   ├── scoring.py        mention value, aspect score and label, overall score, rule-based stars
│   │   └── star_model.py     StarModel: ridge regression, out-of-fold predictions
│   ├── yelp-data/
│   │   └── sampler.py        sample_from_tar() (per star), sample_restaurants_from_tar(), JSONL I/O
│   ├── yelp-agents/
│   │   ├── prompts.py        every prompt: aspect definitions, rubric, few-shot examples
│   │   ├── base.py           Agent.messages(): system + examples + request
│   │   ├── aspect.py         AspectAgent: quotes for one aspect -> AspectResult
│   │   ├── arbiter.py        ArbiterAgent: which aspect owns a disputed passage
│   │   ├── aggregator.py     AggregatorAgent (the lead agent): overall verdict, with a fallback
│   │   └── orchestrator.py   Orchestrator: runs the team per review; batch runs, resumable JSONL
│   └── yelp-eval/
│       ├── metrics.py        evaluate(): accuracy, CIs, confusion matrix, star errors, per-aspect stats
│       ├── compare.py        compare_restaurants(): agreement, verdicts, ANOVA, ICC(1)
│       └── figures.py        3-D Plotly scatter for `yelp-agents viz`
├── apps/
│   ├── cli/main.py           yelp-agents: providers, sample, sample-restaurants, run, eval, compare, viz
│   └── web/                  yelp-web
│       ├── server.py         FastAPI app: /api/status, analyze (SSE), results, metrics, compare, runs
│       └── static/           index.html, app.js, styles.css
├── notebooks/                the Colab notebook
├── tests/                    test_scoring, test_pipeline, test_compare, test_llm_backends, test_web
├── data/                     samples you draw (git-ignored)
└── outputs/                  results and metrics (git-ignored)
```

(Each package's code lives in `packages/<name>/src/<name_with_underscores>/`; the tree shortens those paths.)

**Follow one review through the code:**

1. `yelp-agents sample` (`cli/main.py` → `sample_from_tar()`) streams the tar and writes `data/sample_reviews.jsonl`.
2. `yelp-agents run` loads it with `load_reviews()` and calls `Orchestrator.run()`, which sends batches of
   reviews to a thread pool.
3. For each review, `Orchestrator.analyze_steps()` runs the three `AspectAgent`s in parallel. Each builds its
   prompt from `prompts.py` and calls `LLMClient.chat_json()`.
4. Each agent hands the model's mentions to `build_aspect_result()`, which runs `ground_quote()` on every quote,
   drops duplicates and computes the aspect score.
5. `find_conflicts()` finds quotes two aspects share. `ArbiterAgent` settles each one, and `rescore()` updates
   the aspect that lost.
6. `AggregatorAgent` gives the overall verdict, and `build_overall()` turns it into the overall score and
   rule-based stars.
7. The finished `ReviewAnalysis` is appended to `outputs/results.jsonl` as one JSON line.
8. `yelp-agents eval` (`evaluate()`), `yelp-agents compare` (`compare_restaurants()`) and the web server
   (`server.py`) all read that file. None of them call the LLM again.

## Setup for ASU students

You need an ASURITE ID with Duo, [uv](https://docs.astral.sh/uv/getting-started/installation/), git,
and about 5 GB of disk space.

1. **Connect to the ASU VPN.** Install the Cisco Secure Client from [sslvpn.asu.edu](https://sslvpn.asu.edu/),
   connect to `sslvpn.asu.edu/2fa`, and approve the Duo prompt. Voyager requires the VPN even on campus
   ([ASU SSL VPN](https://docs.rc.asu.edu/sslvpn/)).
2. **Create a Voyager API key.** Sign in at [voyager.rc.asu.edu](https://voyager.rc.asu.edu), open the
   **LLM Access** tab and click **Create Key**. LLM API access is free for anyone with an ASURITE ID
   ([Voyager accounts](https://docs.rc.asu.edu/voyager-accounts/), [LLM API](https://docs.rc.asu.edu/ai/api/)).
3. **Download the dataset.** Use **Download JSON** on the
   [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) page, and keep
   `yelp_dataset.tar` (4.35 GB) as it is. The commands look for `~/Downloads/Yelp JSON/yelp_dataset.tar`;
   pass `--tar <path>` if it is somewhere else.
4. **Install and add your key.** Run all commands from this folder, because `.env` is read from the
   current directory:

   ```bash
   uv sync
   cp .env.example .env        # then set API_KEY=<your Voyager key>
   uv run yelp-agents providers   # should end with: resolves to asu:gemma4-31b-it
   ```

`.env` is git-ignored: never commit your key.

## Quick start

```bash
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

What a run looked like on 2026-09-28 (`gemma4-31b-it` on ASU Voyager, about 5.5 s per review):

| Run | Result |
|---|---|
| Accuracy check, 100 reviews | 84% agreement with the star class (95% CI 76–90%), macro-F1 0.79, star MAE 0.41. Most misses are 3★ reviews. |
| Restaurant comparison, 50 × 10 reviews | Service splits reviewers of the same restaurant most (41 of 50 restaurants have positive and negative service reviews). Food tracks the Yelp rating best (Spearman ρ 0.61). Averaging 10 reviews halves the star error (0.30 → 0.15). |

## Comparing reviews of the same restaurant

The default sample takes single reviews from all over the dataset. To compare what different
reviewers say about the *same* restaurant, sample whole restaurants instead:

```bash
uv run yelp-agents sample-restaurants -b 50 -k 10   # 50 random restaurants x 10 random reviews each
uv run yelp-agents run --sample data/restaurant_reviews.jsonl --out outputs/restaurant_results.jsonl --workers 4
uv run yelp-agents compare                          # summary + one line per restaurant
uv run yelp-agents compare --restaurant "nonna"     # one restaurant's reviews side by side
uv run yelp-web                                     # Compare tab reads outputs/restaurant_results.jsonl
```

The Compare tab always reads `outputs/restaurant_results.jsonl` when it exists, so the other tabs can
stay on the stratified sample. `yelp-web --restaurants` points every tab at the restaurant sample
instead, for example to start batch runs over it from the Explore tab.

Restaurants are drawn uniformly from those with at least `k` reviews in the 40–2000 character window.
Reviews are then drawn uniformly from each restaurant, so star ratings follow that restaurant's real mix
and are not balanced. Each review keeps its restaurant's overall Yelp rating, so the sample can be
checked against it. Reviews are stored grouped by restaurant, so `run -n 100` finishes 10 whole
restaurants.

What `compare` computes (all Python, `packages/yelp-eval/src/yelp_eval/compare.py`):

| Level | Measure |
|---|---|
| Restaurant × aspect | Reviews that mention it; mean and sd of their scores; positive / neutral / negative / silent counts |
| | **Agreement**: share of those reviewers on the majority polarity. The verdict needs ≥ 75% (2+ mentions), otherwise it is *mixed* |
| | **Contested**: at least one positive and one negative reviewer. The most positive and most negative quote show where they part |
| Across restaurants | Within-restaurant sd vs between-restaurant sd; one-way ANOVA and **ICC(1)** with restaurants as groups (does a restaurant's identity explain its reviews' scores?). The reviewers' own stars are the baseline |
| | Spearman ρ of each restaurant's mean aspect score against its overall Yelp rating |
| | Star error per single review vs per restaurant (mean of its reviews), and the sample's mean stars vs the Yelp rating |

## The UI

* **Analyze**: the agents run live on a pasted or random Yelp review, streamed as each one finishes.
  Each aspect card shows its verified quotes with their values, the arithmetic behind the score
  (e.g. `(+3 +2 +2) / (3 × 3) = +0.78`), and any excluded quotes with the reason. You also see arbiter
  decisions, the lead agent's rule-based stars next to the calibrated model and the real rating, and
  the quotes highlighted in the review.
* **Explore**: a 3-D scatter of reviews on the food, service and ambience axes, coloured by actual
  rating or predicted sentiment. Click a point or a table row for detail, including the out-of-fold
  calibrated stars. You can start batch runs from here.
* **Compare**: the restaurant sample, one restaurant at a time. For each aspect you see its verdict
  (positive, negative or mixed), a dot plot of every reviewer's score, and the quotes that disagree
  most. Below that, all the reviews side by side; select a dot or a row to read a review with its
  evidence highlighted. A summary table shows the ANOVA / ICC(1) results across restaurants.
* **Metrics**: accuracy with a 95% CI, macro-F1, star error for the rule and the calibrated model,
  quote grounding rate, confusion matrix, per-class precision/recall/F1, per-dimension mention rate
  and Spearman ρ vs stars, and a business table.

## Colab notebook

[`notebooks/yelp_aspect_agents_colab.ipynb`](notebooks/yelp_aspect_agents_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/manastole03/Grader_Project/blob/main/yelp-aspect-agents/notebooks/yelp_aspect_agents_colab.ipynb)) is a self-contained teaching version of the pipeline. It calls the same model
(`gemma4-31b-it` on Voyager) with the project's prompts and quote checking, copied verbatim from the
packages. It uses the same formulas, samplers and statistics, and with seed 42 draws exactly the same
reviews as `sample` and `sample-restaurants`.

- **API key:** store it as a Colab secret named `VOYAGER_API_KEY`; the notebook never prints it.
- **Data:** `yelp_dataset.tar` in Google Drive (one streaming pass draws both samples), uploaded JSONL
  samples, or 8 made-up demo reviews.
- **Results:** saved to `My Drive/yelp_aspect_agents/` and resumed after a disconnect.
- **Connection:** the API answered without the ASU VPN in testing (only the Voyager website needs it).
  If Colab can't connect, run the notebook locally with the VPN on.

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
uv run pytest    # 40 tests, about 15 s; no API key or dataset needed
```

They cover grounding, the formulas against hand-computed values, cross-validation leakage, ANOVA and
ICC(1) against scipy, the agents on the offline mock, the restaurant sampler on a synthetic tar,
`.env` parsing, and every web API endpoint.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Voyager doesn't load | Connect the ASU VPN (`sslvpn.asu.edu/2fa`) |
| `providers` resolves to `ollama` or `mock` | Run from this folder and check `.env` has `API_KEY=<your key>` |
| `HTTP 401 - check the API key` | Create a new key in Voyager's **LLM Access** tab |
| `HTTP 429 rate limited` | Retries happen automatically; use `--workers 2` if it persists |
| Compare tab blank or "Couldn't load the comparison" | Restart `uv run yelp-web`: the server was started before you pulled new code |
