#!/usr/bin/env python3
"""
train_perf_router.py
====================
Trains PerfRouter — an XGBoost model that predicts which model in the pool
will produce the best utility (quality - α×cost) for a given task type.

─────────────────────────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────────────────────────

Unlike TRouter (which routes based on query embeddings), PerfRouter
routes based on task type + model features. It answers the question:

  "Given that this query is of type X, which model has the best
   expected utility (quality adjusted for cost)?"

The XGBoost model is trained per-task-type:
  - Input:  model feature vector (49 features from model_features.csv)
  - Output: predicted utility score for that model on that task type

At inference time:
  1. Classify the incoming query into a task type (via sentence-BERT)
  2. For each model in the pool, predict its utility using XGBoost
  3. Route to the model with the highest predicted utility

─────────────────────────────────────────────────────────────────────
TRAINING DATA
─────────────────────────────────────────────────────────────────────

Training rows come from two sources:

  A) WCB ground truth (high confidence)
     For the 3 models that ran WildClawBench, we have real
     overall_score per task type. These rows get sample_weight=3.0.

  B) Benchmark-derived affinity scores (lower confidence)
     For all 10 models, the affinity score per task type is the
     weighted average of their benchmark scores. These rows get
     sample_weight=1.0.

The utility label combines quality and cost:
  utility = score - α × normalised_cost

Where:
  score          = WCB overall_score (source A) or affinity score (source B)
  normalised_cost = price_blended_per_1M / max(price_blended_per_1M across pool)
  α              = cost_weight (default 0.3)

─────────────────────────────────────────────────────────────────────
WHY XGBOOST
─────────────────────────────────────────────────────────────────────

With only 10 models × 33 task types = 330 training rows (after
combining sources A and B), a neural network would overfit badly.

XGBoost:
  - Handles small datasets naturally (built-in regularisation)
  - Handles missing features (None/NaN) natively — no imputation needed
  - Produces feature importance scores — useful for understanding
    which features actually drive routing decisions
  - Fast to train (sub-second) and fast to infer (microseconds)
  - Robust to irrelevant features

─────────────────────────────────────────────────────────────────────
Usage:
  pip install xgboost scikit-learn
  python3 train_perf_router.py --features model_features.csv
  python3 train_perf_router.py --features model_features.csv \\
      --cost-weight 0.5 --out perf_router.pkl
─────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ── Constants ─────────────────────────────────────────────────────────────────

# Features used by XGBoost — everything except model_id and WCB scores
# (WCB scores are used to build labels, not as input features)
EXCLUDE_FROM_FEATURES = {
    "model_id",
    "wcb_avg_score",
    "wcb_avg_score_nonzero",
    "wcb_zero_rate",
    # Pricing excluded from XGBoost features — cost is applied explicitly
    # at inference time as: adjusted_utility = predicted_quality - α × cost
    # Including it here would cause XGBoost to double-count cost signal.
    "price_blended_per_1M",
    "price_input_per_1M",
    "price_output_per_1M",
    "price_cache_read_per_1M",
}

# WCB models — these rows get higher sample weight during training
# WCB models identified by normalised ID (no prefix, no :free suffix)
WCB_MODELS_NORM = {
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "gpt_oss_120b",
}

WCB_SAMPLE_WEIGHT  = 3.0   # WCB rows are more reliable
BASE_SAMPLE_WEIGHT = 1.0   # benchmark-derived rows


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric strings to float, keep None for empty
            converted = {}
            for k, v in row.items():
                if v == "" or v == "None":
                    converted[k] = None
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v   # keep as string (model_id)
            rows.append(converted)
    return rows


def get_task_types(rows: list[dict]) -> list[str]:
    """Extract task type names from affinity column names."""
    sample = rows[0]
    return sorted([
        k.replace("affinity_", "").replace("_", ".", 1)
        # re-join: affinity_code_generation → code.generation
        for k in sample if k.startswith("affinity_")
    ])


def affinity_col(task_type: str) -> str:
    """task_type → column name in features CSV."""
    return "affinity_" + task_type.replace(".", "_", 1)


# ── Training data builder ─────────────────────────────────────────────────────

def build_training_data(
    feature_rows: list[dict],
    task_types:   list[str],
    cost_weight:  float,
    wcb_csv_path: Path | None,
) -> tuple[list, list, list, list]:
    """
    Build XGBoost training data: X (features), y (utility labels),
    weights (sample weights), task_type_labels (one per row).

    Returns (X, y, weights, task_labels) where each element is a list
    of length n_rows.
    """
    # Load WCB per-model-per-task-type scores if available
    wcb_scores = _load_wcb_scores(wcb_csv_path)

    # Compute normalised cost across the pool
    costs = [r.get("price_blended_per_1M") or 0.0 for r in feature_rows]
    max_cost = max(costs) if any(c > 0 for c in costs) else 1.0

    # Feature column names (consistent order, excluding labels)
    all_cols = [k for k in feature_rows[0].keys()
                if k not in EXCLUDE_FROM_FEATURES]

    X, y, weights, task_labels = [], [], [], []

    for task_type in task_types:
        acol = affinity_col(task_type)

        for row in feature_rows:
            model_id = row["model_id"]

            # ── Determine the utility label ────────────────────────────────
            # Priority: WCB ground truth > affinity score
            wcb_key  = (_normalise_model_id(model_id), task_type)
            has_wcb  = wcb_key in wcb_scores
            score    = wcb_scores[wcb_key] if has_wcb else row.get(acol)

            if score is None:
                # No data for this (model, task_type) pair — skip
                continue

            # Label = quality score only.
            # Cost is applied at inference time via adjusted_utility = quality - α×cost.
            # This keeps concerns separate: XGBoost predicts quality,
            # cost adjustment happens at routing decision time.
            utility = score

            # ── Build feature vector ───────────────────────────────────────
            # Mask out all affinity columns except the one for THIS task type.
            # This teaches XGBoost which affinity column is relevant per row,
            # making predictions task-type-aware rather than globally ranking.
            acol_active = affinity_col(task_type)
            x_row = []
            for col in all_cols:
                val = row.get(col)
                if col.startswith("affinity_") and col != acol_active:
                    x_row.append(None)   # mask — NaN in numpy
                else:
                    x_row.append(val)

            # Sample weight — WCB rows are more reliable
            is_wcb_model = _normalise_model_id(model_id) in WCB_MODELS_NORM
            weight = WCB_SAMPLE_WEIGHT if (has_wcb or is_wcb_model) else BASE_SAMPLE_WEIGHT

            X.append(x_row)
            y.append(utility)
            weights.append(weight)
            task_labels.append(task_type)

    # ── Normalise labels within each task type ───────────────────────────
    # Without this, XGBoost learns absolute utility values and picks
    # the globally best model for everything. With normalisation, it
    # learns relative rankings within each task type — which is what
    # routing actually needs.
    from collections import defaultdict as _dd
    task_scores = _dd(list)
    for i, tl in enumerate(task_labels):
        task_scores[tl].append((i, y[i]))

    y_norm = list(y)
    for tl, idx_scores in task_scores.items():
        vals = [s for _, s in idx_scores]
        mn, mx = min(vals), max(vals)
        spread = mx - mn
        for idx, score in idx_scores:
            if spread > 0.01:
                # Normalise to [0, 1] within task type
                y_norm[idx] = (score - mn) / spread
            else:
                # All models score the same on this task type — label 0.5
                y_norm[idx] = 0.5

    return X, y_norm, weights, task_labels, all_cols


def _normalise_model_id(model_id: str) -> str:
    """
    Normalise a model ID for matching across different naming conventions.
    Strips provider prefix, free suffix, version separators.
    e.g. "deepseek/deepseek-v4-flash"  → "deepseek_v4_flash"
         "deepseek-v4-flash"            → "deepseek_v4_flash"
         "openai/gpt-oss-120b:free"     → "gpt_oss_120b"
         "openai_gpt-oss-120b_free"     → "gpt_oss_120b"
    """
    import re
    s = model_id.lower()
    # Strip provider prefix (everything before the last /)
    if "/" in s:
        s = s.split("/")[-1]
    # Strip :free or _free suffix
    s = re.sub(r"[_:]free$", "", s)
    # Strip leading duplicate: "deepseek-v4-flash" → already clean
    # but "deepseek-deepseek-v4-flash" → "deepseek-v4-flash"
    parts = s.split("-")
    if len(parts) > 1 and parts[0] == parts[1]:
        parts = parts[1:]
    s = "-".join(parts)
    # Collapse all separators to underscore
    s = re.sub(r"[-_./]+", "_", s)
    return s.strip("_")


def _load_wcb_scores(wcb_csv_path: Path | None) -> dict:
    """
    Load per-model per-task-type scores from wcb_training_labeled.csv.
    Returns {(normalised_model_id, task_type): avg_score}.

    Uses normalised model IDs to bridge the gap between WCB naming
    (e.g. "deepseek-v4-flash") and registry naming
    (e.g. "deepseek/deepseek-v4-flash").
    """
    if wcb_csv_path is None or not wcb_csv_path.exists():
        return {}

    # Group scores by (normalised_model, task_type)
    raw: dict[tuple, list[float]] = defaultdict(list)
    with open(wcb_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            norm_model = _normalise_model_id(row["model"])
            key = (norm_model, row["task_type"])
            raw[key].append(float(row["overall_score"]))

    return {k: sum(v)/len(v) for k, v in raw.items()}


# ── Training ──────────────────────────────────────────────────────────────────

def train(X, y, weights, feature_cols: list[str], cost_weight: float):
    """Train XGBoost regressor and return the trained model."""
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as e:
        print(f"ERROR: {e}. Run: pip install xgboost numpy", file=sys.stderr)
        sys.exit(1)

    # Convert to numpy, replacing None with NaN (XGBoost handles NaN natively)
    X_np = np.array([
        [float("nan") if v is None else float(v) for v in row]
        for row in X
    ], dtype=np.float32)
    y_np      = np.array(y,       dtype=np.float32)
    w_np      = np.array(weights, dtype=np.float32)

    print(f"  Training on {len(X_np)} rows × {X_np.shape[1]} features")
    print(f"  Label range: [{y_np.min():.3f}, {y_np.max():.3f}]")
    print(f"  NaN rate per feature: "
          f"{(np.isnan(X_np).mean(axis=0) * 100).mean():.1f}% avg")

    model = xgb.XGBRegressor(
        n_estimators      = 200,
        max_depth         = 4,         # shallow — we have few training rows
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 1,
        reg_alpha         = 0.1,       # L1 regularisation
        reg_lambda        = 1.0,       # L2 regularisation
        random_state      = 42,
        n_jobs            = -1,
        tree_method       = "hist",    # handles NaN natively
        enable_categorical = False,
    )
    model.fit(X_np, y_np, sample_weight=w_np)

    return model, X_np, y_np


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_np, y_np, feature_cols, task_labels, feature_rows):
    """Print evaluation metrics and feature importance."""
    import numpy as np

    y_pred = model.predict(X_np)
    residuals = y_pred - y_np
    mae  = float(np.abs(residuals).mean())
    rmse = float(np.sqrt((residuals ** 2).mean()))
    r2   = float(1 - np.var(residuals) / np.var(y_np))

    print(f"\n── Training metrics ──────────────────────────────────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Feature importance — top 15
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:15]
    print(f"\n── Top 15 features by importance ────────────────────────────────────")
    for i in top_idx:
        bar = "█" * int(importances[i] * 200)
        print(f"  {feature_cols[i]:<50} {importances[i]:.4f}  {bar}")

    # Per-task-type routing simulation
    print(f"\n── Routing simulation per task type ─────────────────────────────────")
    print(f"  (Which model PerfRouter would choose for each task type)")

    model_ids = [r["model_id"] for r in feature_rows]
    n_models  = len(model_ids)
    n_task_types = len(set(task_labels))

    # For each task type, predict utility for all models and pick best
    all_cols_set = set(feature_cols)
    task_type_groups = defaultdict(list)
    for i, tl in enumerate(task_labels):
        task_type_groups[tl].append(i)

    for task_type in sorted(task_type_groups.keys()):
        indices = task_type_groups[task_type]
        if len(indices) < n_models:
            continue   # incomplete — skip

        # Apply cost adjustment at decision time (not during training)
        preds = y_pred[indices]
        costs = np.array([
            feature_rows[i % n_models].get("price_blended_per_1M") or 0.0
            for i in indices
        ])
        max_c = costs.max() if costs.max() > 0 else 1.0
        cost_weight_val = 0.3   # default α for simulation
        adjusted = preds - cost_weight_val * (costs / max_c)
        best_idx   = int(np.argmax(adjusted))
        best_model = model_ids[best_idx % n_models]
        best_score = adjusted[best_idx]

        short = best_model.split("/")[-1][:35]
        print(f"  {task_type:<45} → {short:<35} ({best_score:.3f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PerfRouter XGBoost model on model feature vectors"
    )
    parser.add_argument("--features",    default=str(_DATA_DIR / "model_features.csv"))
    parser.add_argument("--wcb-csv",     default=None,
                        help="WildClawBench labeled CSV for ground-truth labels")
    parser.add_argument("--out",         default=str(_MODELS_DIR / "perf_router.pkl"))
    parser.add_argument("--cost-weight", type=float, default=0.3,
                        help="α: quality-cost trade-off (default 0.3)")
    args = parser.parse_args()

    features_path = Path(args.features).expanduser()
    wcb_path      = Path(args.wcb_csv).expanduser() if args.wcb_csv else None
    out_path      = Path(args.out).expanduser()

    if not features_path.exists():
        print(f"ERROR: {features_path} not found. Run build_model_features.py first.",
              file=sys.stderr)
        sys.exit(1)

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading features from {features_path}")
    feature_rows = load_features(features_path)
    task_types   = get_task_types(feature_rows)
    print(f"  {len(feature_rows)} models × {len(task_types)} task types")

    # ── Build training data ───────────────────────────────────────────────────
    print(f"\nBuilding training data (cost_weight α={args.cost_weight})...")
    X, y, weights, task_labels, feature_cols = build_training_data(
        feature_rows, task_types, args.cost_weight, wcb_path
    )
    print(f"  {len(X)} training rows")
    print(f"  {feature_cols[:5]}... ({len(feature_cols)} features total)")

    # WCB vs benchmark breakdown
    wcb_rows   = sum(1 for w in weights if w == WCB_SAMPLE_WEIGHT)
    bench_rows = len(weights) - wcb_rows
    print(f"  WCB ground-truth rows : {wcb_rows} (weight={WCB_SAMPLE_WEIGHT})")
    print(f"  Benchmark-derived rows: {bench_rows} (weight={BASE_SAMPLE_WEIGHT})")

    if len(X) < 10:
        print("ERROR: Too few training rows. Check feature CSV.", file=sys.stderr)
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining XGBoost...")
    xgb_model, X_np, y_np = train(X, y, weights, feature_cols, args.cost_weight)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluate(xgb_model, X_np, y_np, feature_cols, task_labels, feature_rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict = {
        "model":        xgb_model,
        "feature_cols": feature_cols,
        "task_types":   task_types,
        "model_ids":    [r["model_id"] for r in feature_rows],
        "cost_weight":  args.cost_weight,
        "training": {
            "n_rows":          len(X),
            "wcb_rows":        wcb_rows,
            "benchmark_rows":  bench_rows,
            "wcb_sample_weight": WCB_SAMPLE_WEIGHT,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"\nSaved PerfRouter to: {out_path}")
    print(f"\nDone. Next step: python3 perf_router_inference.py")


if __name__ == "__main__":
    main()