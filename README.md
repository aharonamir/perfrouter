# PerfRouter

Benchmark-driven LLM routing using XGBoost and sentence-BERT. Routes incoming queries to the model with the highest predicted utility (quality − α × cost) based on task type classification.

## How PerfRouter differs from TRouter

| | TRouter | PerfRouter |
|---|---|---|
| **Training signal** | Human preference labels from WildClawBench | Benchmark scores + WCB ground truth |
| **Routing input** | Query embedding → MLP | Task type (BERT cosine sim) → XGBoost |
| **Model pool** | Fixed at train time | Updatable via `models.yaml` |
| **Inference cost** | Neural forward pass | Microseconds (tree lookup) |
| **Adding a model** | Full retrain | Edit YAML, run step 1 |

## Quick start

```bash
cp .env.example .env
# edit .env with your API keys
uv sync
uv run perf-pipeline
```

## Build pipeline

Each step is idempotent — re-run only if its inputs changed.

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `uv run perf-pipeline --force-step 1` | Fetch AA benchmark data → `data/model_registry.json` |
| 2 | `uv run perf-pipeline --force-step 2` | Fetch Arena ELO scores → merged into registry |
| 3 | `uv run perf-pipeline --force-step 3` | Patch taxonomy with Arena weights |
| 4 | `uv run perf-pipeline --force-step 4` | Build benchmark weight matrix → `data/benchmark_weights.json` |
| 5 | `uv run perf-pipeline --force-step 5` | Compute model feature vectors → `data/model_features.csv` |
| 6 | `uv run perf-pipeline --force-step 6` | Train XGBoost → `models/perf_router.pkl` |

Full rebuild:

```bash
uv run perf-pipeline --force
```

## Adding a new model

1. Edit `models.yaml` — add the model entry with its `id`, `aa_slug`, pricing, and flags.
2. Run:
   ```bash
   uv run perf-pipeline --force-step 1
   ```
   This re-fetches AA data, rebuilds features, and retrains from step 1 onwards.

## Runtime environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_ROUTER_COST_WEIGHT` | `0.3` | α — how much cost penalises quality (0 = quality only, 1 = cost only) |
| `PERF_ROUTER_DEGRADATION_THRESHOLD` | `0.0` | Accept up to this fraction of quality drop for a cheaper model |
| `PERF_ROUTER_MIN_SIMILARITY` | `0.20` | Below this task-type similarity, fall back to cheapest model |
| `PERF_ROUTER_BASELINE` | `deepseek/deepseek-v4-pro` | Baseline model for cost and quality comparison logging |

## WCB simulation

Validate routing decisions against WildClawBench task prompts without executing any tasks:

```bash
uv run perf-simulate --tasks-dir path/to/WildClawBench/tasks
```

Optional flags:
```bash
uv run perf-simulate \
    --tasks-dir path/to/WildClawBench/tasks \
    --cost-weight 0.5 \
    --degradation-threshold 0.10 \
    --out my_simulation.csv
```

## optmod integration

To use PerfRouter inside optmod:

1. Copy these files into `routing/`:
   - `perfrouter/inference/perf_router_inference.py`
   - `data/task_taxonomy.json`
   - `data/model_registry.json`
   - `data/model_features.csv`
   - `models/perf_router.pkl`

2. In `routing/config.yaml`, set:
   ```yaml
   router: perfrouter
   perf_router_cost_weight: 0.3
   perf_router_baseline: deepseek/deepseek-v4-pro
   ```
