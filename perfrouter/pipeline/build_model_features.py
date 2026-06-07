#!/usr/bin/env python3
"""
build_model_features.py
=======================
Computes the feature vector for each model in model_registry.json
and outputs model_features.csv — the training input for PerfRouter.

─────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES
─────────────────────────────────────────────────────────────────────

For each model, it builds a feature vector consisting of:

  A) Task-type affinity scores  [33 features]
     One score per task type. Computed as a weighted average of the
     model's benchmark scores, using the weights from benchmark_weights.json.

     affinity(model, task_type) =
         Σ benchmark_weight(b, task_type) × model_score(b)
         ─────────────────────────────────────────────────
         Σ benchmark_weight(b, task_type)   [for benchmarks where score is not null]

     If a model has no score for any benchmark in a task type's weight
     vector, that task type's affinity is None (missing).

  B) Structural features  [8 features]
     - total_params_B          (model size)
     - active_params_B         (active params for MoE, = total_params_B for dense)
     - moe_density             (active/total — 1.0 for dense models)
     - context_window_k        (total context)
     - effective_context_k     (practical context limit)
     - context_headroom_ratio  (effective/total)
     - price_blended_per_1M    (blended cost signal)
     - speed_output_tps        (output tokens per second from AA)

  C) Boolean flags  [5 features, as 0/1 integers]
     - flag_reasoning          (is_reasoning model)
     - flag_thinking           (has thinking mode)
     - flag_vision             (supports vision input)
     - flag_tools              (supports tool calling)
     - flag_free               (zero cost)

  D) WildClawBench seed scores  [3 features — only for flash/pro/gpt-oss]
     For models we ran WildClawBench on, we add the actual observed
     avg score per model as a ground-truth quality signal. This anchors
     the XGBoost training to real routing outcomes, not just benchmarks.
     Models without WCB runs get NaN here.

Total features per model: 33 + 8 + 5 + 3 = 49 features

─────────────────────────────────────────────────────────────────────
SEEDING FROM WILDCLAW BENCH
─────────────────────────────────────────────────────────────────────

The WCB seed provides per-model ground-truth utility scores that
XGBoost uses to calibrate the relationship between benchmark features
and real routing utility. Without it, we're purely extrapolating from
benchmarks. With it, the model learns "a model with AA-coding=0.45 and
LiveCode=0.65 achieves utility 0.261 in real agent tasks" — grounding
the predictions in reality.

Seed data comes from wcb_training_labeled.csv if present.

Usage:
  python3 build_model_features.py
  python3 build_model_features.py \\
      --registry model_registry.json \\
      --weights  benchmark_weights.json \\
      --wcb-csv  wcb_training_labeled.csv \\
      --out      model_features.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"


# ── Feature column definitions ────────────────────────────────────────────────

STRUCTURAL_FEATURES = [
    "total_params_B",
    "active_params_B",
    "moe_density",
    "context_window_k",
    "effective_context_k",
    "context_headroom_ratio",
    "price_blended_per_1M",
    "speed_output_tps",
    "latency_ttft_s",
]

FLAG_FEATURES = [
    "flag_reasoning",
    "flag_thinking",
    "flag_vision",
    "flag_tools",
    "flag_free",
]

# WCB seed columns
WCB_SEED_FEATURES = [
    "wcb_avg_score",           # avg overall_score across all 60 tasks
    "wcb_avg_score_nonzero",   # avg score on tasks where model scored >0
    "wcb_zero_rate",           # fraction of tasks where model scored 0
]


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_registry(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("models", [])


def load_weights(path: Path) -> tuple[list[str], list[str], dict]:
    """Returns (benchmarks, task_types, matrix)."""
    data    = json.loads(path.read_text(encoding="utf-8"))
    return (
        data["benchmarks"],
        data["task_types"],
        data["matrix"],          # {task_type → {benchmark → weight}}
    )


def load_wcb_seed(csv_path: Path | None) -> dict[str, dict]:
    """
    Load WildClawBench scores and compute per-model summary stats.
    Returns {model_id_normalised → {wcb_avg_score, wcb_avg_score_nonzero, wcb_zero_rate}}
    """
    if csv_path is None or not csv_path.exists():
        return {}

    scores_by_model = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model   = row["model"]
            score   = float(row["overall_score"])
            scores_by_model[model].append(score)

    seed = {}
    for model, scores in scores_by_model.items():
        nonzero = [s for s in scores if s > 0]
        seed[model] = {
            "wcb_avg_score":         round(sum(scores) / len(scores), 6),
            "wcb_avg_score_nonzero": round(sum(nonzero) / len(nonzero), 6) if nonzero else 0.0,
            "wcb_zero_rate":         round(sum(1 for s in scores if s == 0) / len(scores), 6),
        }

    return seed


def normalise_model_id(model_id: str) -> str:
    """Normalise model ID for matching: lowercase, collapse separators."""
    import re
    return re.sub(r"[/_:\-]", "_", model_id.lower())


# ── Feature computation ───────────────────────────────────────────────────────

def compute_affinity_scores(
    model: dict,
    benchmarks: list[str],
    task_types: list[str],
    matrix: dict,
) -> dict[str, float | None]:
    """
    Compute task-type affinity scores for one model.

    For each task type:
      1. Get the benchmark weights for that task type
      2. Look up the model's score for each benchmark
      3. Compute weighted average, skipping null scores
      4. Return None if no benchmark scores available for this task type
    """
    affinities = {}

    for task_type in task_types:
        weights = matrix.get(task_type, {})

        weighted_sum   = 0.0
        effective_weight = 0.0

        for benchmark, weight in weights.items():
            if weight == 0.0:
                continue

            score = model.get(benchmark)
            if score is None:
                # Skip this benchmark — model wasn't tested on it
                continue

            # Normalise benchmark scores to [0, 1] range
            # AA intelligence/coding/math indices are on 0-100 scale
            if benchmark in ("aa_intelligence_index", "aa_coding_index", "aa_math_index"):
                score = score / 100.0

            weighted_sum     += weight * score
            effective_weight += weight

        if effective_weight == 0.0:
            affinities[task_type] = None   # no data
        else:
            # Renormalise: if some benchmarks had null scores,
            # scale up the remaining weights to sum to 1.0
            affinities[task_type] = round(weighted_sum / effective_weight, 6)

    return affinities


def build_feature_row(
    model: dict,
    benchmarks: list[str],
    task_types: list[str],
    matrix: dict,
    wcb_seed: dict,
) -> dict:
    """Build the complete feature row for one model."""
    row = {"model_id": model["id"]}

    # ── A: Task-type affinity scores ──────────────────────────────────────────
    affinities = compute_affinity_scores(model, benchmarks, task_types, matrix)
    for task_type in task_types:
        col = f"affinity_{task_type.replace('.', '_')}"
        row[col] = affinities.get(task_type)

    # ── B: Structural features ────────────────────────────────────────────────
    for feat in STRUCTURAL_FEATURES:
        row[feat] = model.get(feat)

    # ── C: Boolean flags ──────────────────────────────────────────────────────
    for feat in FLAG_FEATURES:
        row[feat] = model.get(feat, 0)

    # ── D: WildClawBench seed scores ──────────────────────────────────────────
    norm_id = normalise_model_id(model["id"])

    # Try to match by normalised ID
    wcb_data = None
    for wcb_model, wcb_scores in wcb_seed.items():
        if normalise_model_id(wcb_model) == norm_id:
            wcb_data = wcb_scores
            break
        # Also try matching on the last component (e.g. 'deepseek-v4-flash')
        local_id = normalise_model_id(wcb_model.split("/")[-1])
        model_local = normalise_model_id(model["id"].split("/")[-1].split(":")[0])
        if local_id == model_local:
            wcb_data = wcb_scores
            break

    for feat in WCB_SEED_FEATURES:
        row[feat] = wcb_data.get(feat) if wcb_data else None

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute model feature vectors for PerfRouter training"
    )
    parser.add_argument("--registry", default=str(_DATA_DIR / "model_registry.json"))
    parser.add_argument("--weights",  default=str(_DATA_DIR / "benchmark_weights.json"))
    parser.add_argument("--wcb-csv",  default=str(_DATA_DIR / "wcb_training_labeled.csv"),
                        help="WildClawBench labeled CSV for seed scores")
    parser.add_argument("--out",      default=str(_DATA_DIR / "model_features.csv"))
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser()
    weights_path  = Path(args.weights).expanduser()
    wcb_path      = Path(args.wcb_csv).expanduser()
    out_path      = Path(args.out).expanduser()

    for p, label in [(registry_path, "--registry"), (weights_path, "--weights")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"Loading model registry from {registry_path}")
    models = load_registry(registry_path)
    print(f"  {len(models)} models loaded")

    print(f"Loading benchmark weights from {weights_path}")
    benchmarks, task_types, matrix = load_weights(weights_path)
    print(f"  {len(benchmarks)} benchmarks × {len(task_types)} task types")

    wcb_seed = load_wcb_seed(wcb_path if wcb_path.exists() else None)
    if wcb_seed:
        print(f"Loading WCB seed from {wcb_path}")
        print(f"  {len(wcb_seed)} models with WCB ground truth")
    else:
        print(f"No WCB seed found at {wcb_path} — affinity features only")

    # ── Build feature rows ────────────────────────────────────────────────────
    print(f"\nComputing feature vectors...")
    rows = []
    for model in models:
        row = build_feature_row(model, benchmarks, task_types, matrix, wcb_seed)
        rows.append(row)
        has_wcb  = row.get("wcb_avg_score") is not None
        n_affin  = sum(1 for k, v in row.items()
                       if k.startswith("affinity_") and v is not None)
        aa_idx   = model.get("aa_intelligence_index")
        aa_str   = f"AA={aa_idx:.0f}" if aa_idx else "AA=—"
        wcb_str  = f"WCB={row['wcb_avg_score']:.3f}" if has_wcb else "WCB=—"
        print(f"  {model['id']:<48} {aa_str:<8} {wcb_str:<12} "
              f"{n_affin}/{len(task_types)} affinities")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not rows:
        print("ERROR: No rows produced", file=sys.stderr)
        sys.exit(1)

    # Column order: model_id, affinities (sorted), structural, flags, wcb
    affinity_cols  = sorted([k for k in rows[0] if k.startswith("affinity_")])
    fieldnames     = (
        ["model_id"]
        + affinity_cols
        + STRUCTURAL_FEATURES
        + FLAG_FEATURES
        + WCB_SEED_FEATURES
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to: {out_path}")
    print(f"  Rows    : {len(rows)} models")
    print(f"  Columns : {len(fieldnames)} features")
    print(f"            {len(affinity_cols)} task-type affinities")
    print(f"            {len(STRUCTURAL_FEATURES)} structural features")
    print(f"            {len(FLAG_FEATURES)} boolean flags")
    print(f"            {len(WCB_SEED_FEATURES)} WCB seed scores")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\n── Affinity score sample (top 5 task types per model) ────────────────")
    for row in rows:
        affins = {k.replace("affinity_", ""): v
                  for k, v in row.items()
                  if k.startswith("affinity_") and v is not None}
        top5 = sorted(affins.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  {row['model_id']}")
        for task, score in top5:
            bar = "█" * int(score * 20)
            print(f"    {task:<45} {score:.3f}  {bar}")

    print(f"\n── Missing affinity coverage ─────────────────────────────────────────")
    print("  (task types where model has no benchmark data)")
    for row in rows:
        missing = [k.replace("affinity_", "")
                   for k, v in row.items()
                   if k.startswith("affinity_") and v is None]
        if missing:
            print(f"  {row['model_id']:<48} missing: {len(missing)} types")
            for m in missing[:5]:
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing)-5} more")

    print(f"\nDone. Next step: python3 train_perf_router.py --features model_features.csv")


if __name__ == "__main__":
    main()