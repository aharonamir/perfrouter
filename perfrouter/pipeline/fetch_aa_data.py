#!/usr/bin/env python3
"""
fetch_aa_data.py
================
Reads models.yaml, fetches benchmark and performance data from the
Artificial Analysis API for each model that has an aa_slug, and
merges everything into model_registry.json.

model_registry.json is the single source of truth for PerfRouter's
feature vectors — one entry per model containing:
  - Manual specs from models.yaml (params, pricing, flags)
  - AA benchmark scores (intelligence index, coding, agentic, benchmarks)
  - AA performance data (speed, latency, context)

Models with aa_slug: null get only the manual specs from the YAML.
Models where the AA API returns no data are flagged with aa_data: false.

─────────────────────────────────────────────────────────────────────
Artificial Analysis API
─────────────────────────────────────────────────────────────────────
Free tier: 1,000 requests/day
Endpoint : GET https://artificialanalysis.ai/api/v2/data/llms/models
Auth     : Bearer token via AA_API_KEY env var (or --api-key flag)
Sign up  : https://artificialanalysis.ai/documentation

─────────────────────────────────────────────────────────────────────
Usage:
  export AA_API_KEY=your_key_here
  python3 fetch_aa_data.py --models models.yaml --out model_registry.json

  # Dry run — show what would be fetched without calling API
  python3 fetch_aa_data.py --models models.yaml --dry-run

  # Force re-fetch even if model_registry.json already exists
  python3 fetch_aa_data.py --models models.yaml --force
─────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


# ── AA API config ─────────────────────────────────────────────────────────────

AA_API_BASE    = "https://artificialanalysis.ai/api/v2/data/llms"
AA_MODELS_URL  = f"{AA_API_BASE}/models"
AA_TIMEOUT_S   = 30
AA_RETRY_COUNT = 3
AA_RETRY_DELAY = 2.0   # seconds between retries


# ── AA benchmark field mapping ────────────────────────────────────────────────
# Maps the AA API response field names to our internal names.
# These are the fields we extract from the AA response per model.
#
# Benchmark → task-type relevance (used later in build_benchmark_weights.py):
#   gdpval_aa          → tool_use.agentic_real_world, tool_use.multi_step
#   terminal_bench     → code.debugging, code.cli_shell, tool_use.shell
#   tau2_bench         → tool_use.api_chaining, tool_use.multi_step
#   aa_lcr             → retrieval.long_context, synthesis.multi_document
#   omniscience_acc    → retrieval.factual, knowledge.domain
#   omniscience_nohal  → confidence signal (cross-cutting, not task-specific)
#   hle                → reasoning.hard, reasoning.scientific
#   gpqa               → reasoning.scientific, reasoning.causal
#   scicode            → code.scientific, code.generation
#   ifbench            → safety.instruction_following, safety.constraint
#   critpt             → reasoning.physics, reasoning.mathematical
#   apex_agents        → tool_use.long_horizon_agentic
#   mmmu_pro           → synthesis.visual, reasoning.multimodal

AA_BENCHMARK_FIELDS = {
    # Composite indices — nested under 'evaluations' in API response
    "artificial_analysis_intelligence_index": "aa_intelligence_index",
    "artificial_analysis_coding_index":       "aa_coding_index",
    "artificial_analysis_math_index":         "aa_math_index",

    # Individual benchmarks — nested under 'evaluations'
    "terminalbench_hard": "bench_terminal_bench",
    "tau2":               "bench_tau2_bench",
    "lcr":                "bench_aa_lcr",
    "hle":                "bench_hle",
    "gpqa":               "bench_gpqa",
    "scicode":            "bench_scicode",
    "ifbench":            "bench_ifbench",
    "aime_25":            "bench_aime",
    "livecodebench":      "bench_livecodebench",
    "mmlu_pro":           "bench_mmlu_pro",
    "math_500":           "bench_math_500",
}

# Performance fields — top-level in API response (not nested)
AA_PERF_FIELDS = {
    "median_output_tokens_per_second":    "speed_output_tps",
    "median_time_to_first_token_seconds": "latency_ttft_s",
    "median_time_to_first_answer_token":  "latency_e2e_500_s",
}

# Pricing fields — nested under 'pricing' in API response
AA_PRICING_FIELDS = {
    "price_1m_input_tokens":  "aa_price_input_per_1M",
    "price_1m_output_tokens": "aa_price_output_per_1M",
    "price_1m_blended_3_to_1": "aa_price_blended_per_1M",
}

# AA_SPEC_FIELDS replaced by AA_PRICING_FIELDS and AA_PERF_FIELDS above


# ── AA API client ─────────────────────────────────────────────────────────────

def fetch_aa_models(api_key: str) -> list[dict] | None:
    """
    Fetch the full AA model list from the API.
    Returns a list of model dicts, or None on failure.

    The API returns all models in one response — we fetch once
    and then look up each model by slug locally.
    """
    headers = {
        "x-api-key": api_key,
        "Accept":        "application/json",
        "User-Agent":    "PerfRouter/1.0",
    }

    for attempt in range(1, AA_RETRY_COUNT + 1):
        try:
            print(f"  Fetching AA model list (attempt {attempt}/{AA_RETRY_COUNT})...")
            resp = requests.get(AA_MODELS_URL, headers=headers, timeout=AA_TIMEOUT_S)

            if resp.status_code == 200:
                data = resp.json()
                # AA API returns {"data": [...]} or just [...]
                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                else:
                    print(f"  WARN: unexpected AA API response structure: {type(data)}")
                    return None

            elif resp.status_code == 401:
                print("  ERROR: AA API key invalid or missing (401 Unauthorized)")
                return None

            elif resp.status_code == 429:
                print(f"  WARN: AA API rate limited (429) — waiting {AA_RETRY_DELAY * attempt}s")
                time.sleep(AA_RETRY_DELAY * attempt)

            else:
                print(f"  WARN: AA API returned {resp.status_code}: {resp.text[:200]}")
                if attempt < AA_RETRY_COUNT:
                    time.sleep(AA_RETRY_DELAY)

        except requests.RequestException as e:
            print(f"  ERROR: AA API request failed: {e}")
            if attempt < AA_RETRY_COUNT:
                time.sleep(AA_RETRY_DELAY)

    return None


def find_aa_model(aa_models: list[dict], slug: str) -> dict | None:
    """
    Find a model in the AA response by slug.
    Tries exact match first, then normalised match.
    """
    slug_norm = slug.lower().replace("-", "_")

    for m in aa_models:
        # AA models have a 'model_id', 'slug', or 'id' field depending on API version
        for field in ("slug", "model_id", "id", "model_slug"):
            val = m.get(field, "")
            if val and (val.lower() == slug.lower() or
                        val.lower().replace("-", "_") == slug_norm):
                return m

    return None


def extract_aa_fields(aa_model: dict) -> dict:
    """
    Extract benchmark scores and performance metrics from the AA API response.

    AA API structure (confirmed from live API):
      {
        "id": "...", "name": "...", "slug": "...",
        "evaluations": {
            "artificial_analysis_intelligence_index": 20.8,
            "artificial_analysis_coding_index": 14.4,
            "artificial_analysis_math_index": 62.3,
            "mmlu_pro": 0.718, "gpqa": 0.611, "hle": 0.051,
            "livecodebench": 0.652, "scicode": 0.34,
            "aime_25": 0.623, "ifbench": 0.578,
            "lcr": 0.31, "terminalbench_hard": 0.045, "tau2": 0.503
        },
        "pricing": {
            "price_1m_input_tokens": 0.06,
            "price_1m_output_tokens": 0.20,
            "price_1m_blended_3_to_1": 0.095
        },
        "median_output_tokens_per_second": 273.54,
        "median_time_to_first_token_seconds": 0.498,
        "median_time_to_first_answer_token": 7.81
      }
    """
    result = {"aa_data": True}

    # ── Benchmark scores — nested under 'evaluations' ─────────────────────
    evals = aa_model.get("evaluations") or {}
    for aa_field, our_field in AA_BENCHMARK_FIELDS.items():
        val = evals.get(aa_field)
        result[our_field] = float(val) if val is not None else None

    # ── Performance metrics — top-level fields ────────────────────────────
    for aa_field, our_field in AA_PERF_FIELDS.items():
        val = aa_model.get(aa_field)
        result[our_field] = float(val) if val is not None else None

    # ── Pricing — nested under 'pricing' ─────────────────────────────────
    pricing = aa_model.get("pricing") or {}
    for aa_field, our_field in AA_PRICING_FIELDS.items():
        val = pricing.get(aa_field)
        result[our_field] = float(val) if val is not None else None

    return result


# ── Model registry builder ────────────────────────────────────────────────────

def build_registry(yaml_models: list[dict], aa_models: list[dict] | None) -> list[dict]:
    """
    Merge YAML specs with AA API data into one registry entry per model.

    Priority: AA API data > YAML manual specs
    (AA data is independently measured; YAML is our best guess for manual fields)
    """
    registry = []

    for m in yaml_models:
        model_id = m["id"]
        aa_slug  = m.get("aa_slug")

        print(f"\n  Processing: {model_id}")

        # Start with all YAML fields
        entry = {
            # Identity
            "id":       model_id,
            "aa_slug":  aa_slug,
            "provider": m.get("provider"),
            "license":  m.get("license"),

            # Cost flags
            "free":     m.get("free", False),

            # Manual specs from YAML
            "total_params_B":      m.get("total_params_B"),
            "active_params_B":     m.get("active_params_B"),
            "context_window_k":    m.get("context_window_k"),
            "effective_context_k": m.get("effective_context_k"),
            "supports_vision":     m.get("supports_vision", False),
            "supports_tools":      m.get("supports_tools", True),
            "is_reasoning":        m.get("is_reasoning", False),
            "has_thinking_mode":   m.get("has_thinking_mode", False),

            # Pricing from YAML (authoritative — AA may show different tiers)
            "price_input_per_1M":      m.get("price_input_per_1M", 0.0),
            "price_output_per_1M":     m.get("price_output_per_1M", 0.0),
            "price_cache_read_per_1M": m.get("price_cache_read_per_1M", 0.0),

            # AA data fields — populated below if available
            "aa_data": False,
        }

        # Merge AA API data if available
        if aa_slug and aa_models is not None:
            aa_match = find_aa_model(aa_models, aa_slug)
            if aa_match:
                aa_fields = extract_aa_fields(aa_match)
                entry.update(aa_fields)
                print(f"    ✓ AA data found — intelligence_index={aa_fields.get('aa_intelligence_index')}")

                # Use AA context window if YAML didn't specify or AA is more specific
                if aa_fields.get("aa_context_window_k") and not m.get("context_window_k"):
                    entry["context_window_k"] = aa_fields["aa_context_window_k"]
            else:
                print(f"    ⚠ No AA match found for slug '{aa_slug}'")
                entry["aa_data"] = False

        elif aa_slug is None:
            print(f"    ↷ No aa_slug — using YAML specs only")
        else:
            print(f"    ⚠ AA API unavailable — using YAML specs only")

        # Derived features — computed from the merged data
        # These are used directly as features in the PerfRouter XGBoost model

        # Cost per 1M tokens (blended 3:1 input/output ratio, standard AA convention)
        p_in  = entry.get("price_input_per_1M")  or 0.0
        p_out = entry.get("price_output_per_1M") or 0.0
        entry["price_blended_per_1M"] = round((3 * p_in + p_out) / 4, 6)

        # Context headroom: how much of the context window is practically usable
        ctx_total = entry.get("context_window_k")    or 0
        ctx_eff   = entry.get("effective_context_k") or 0
        entry["context_headroom_ratio"] = (
            round(ctx_eff / ctx_total, 3) if ctx_total > 0 else 0.5
        )

        # Parameter efficiency: active / total (MoE density)
        total  = entry.get("total_params_B")  or 0.0
        active = entry.get("active_params_B") or 0.0
        entry["moe_density"] = (
            round(active / total, 3) if total > 0 else 1.0
        )

        # Boolean flags as ints (XGBoost handles these fine but explicit is cleaner)
        entry["flag_reasoning"]    = int(entry.get("is_reasoning",      False))
        entry["flag_thinking"]     = int(entry.get("has_thinking_mode", False))
        entry["flag_vision"]       = int(entry.get("supports_vision",   False))
        entry["flag_tools"]        = int(entry.get("supports_tools",    True))
        entry["flag_free"]         = int(entry.get("free",              False))

        registry.append(entry)

    return registry


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch AA data and build PerfRouter model_registry.json"
    )
    parser.add_argument("--models",  default=str(_PROJECT_ROOT / "models.yaml"),
                        help="Path to models.yaml")
    parser.add_argument("--out",     default=str(_DATA_DIR / "model_registry.json"),
                        help="Output path")
    parser.add_argument("--api-key", default=None,
                        help="AA API key (default: reads AA_API_KEY env var)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-fetch even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be fetched without calling API")
    args = parser.parse_args()

    models_path = Path(args.models).expanduser()
    out_path    = Path(args.out).expanduser()

    # ── Load YAML ─────────────────────────────────────────────────────────────
    if not models_path.exists():
        print(f"ERROR: {models_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(models_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    yaml_models = config.get("models", [])
    print(f"Loaded {len(yaml_models)} models from {models_path}")

    # ── Check output exists ───────────────────────────────────────────────────
    if out_path.exists() and not args.force and not args.dry_run:
        print(f"Output already exists: {out_path}")
        print("Pass --force to re-fetch.")
        sys.exit(0)

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDRY RUN — models that would be fetched from AA API:")
        for m in yaml_models:
            slug = m.get("aa_slug")
            status = f"→ fetch slug '{slug}'" if slug else "→ skip (no aa_slug)"
            print(f"  {m['id']:<45} {status}")
        return

    # ── Fetch AA data ─────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("AA_API_KEY")
    aa_models = None

    slugs_needed = [m["aa_slug"] for m in yaml_models if m.get("aa_slug")]

    if not slugs_needed:
        print("No aa_slugs defined — skipping AA API fetch, using YAML specs only.")
    elif not api_key:
        print("WARN: AA_API_KEY not set — skipping AA API fetch, using YAML specs only.")
        print("      Set AA_API_KEY or pass --api-key to fetch benchmark data.")
    else:
        print(f"\nFetching AA data for {len(slugs_needed)} models...")
        aa_models = fetch_aa_models(api_key)
        if aa_models:
            print(f"  AA API returned {len(aa_models)} total models")
        else:
            print("  WARN: AA API fetch failed — proceeding with YAML specs only")

    # ── Build registry ────────────────────────────────────────────────────────
    print(f"\nBuilding model registry...")
    registry = build_registry(yaml_models, aa_models)

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "source":   str(models_path),
        "aa_fetch": aa_models is not None,
        "models":   registry,
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nSaved model_registry.json → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    aa_found    = sum(1 for m in registry if m.get("aa_data"))
    aa_missing  = sum(1 for m in registry if m.get("aa_slug") and not m.get("aa_data"))
    no_slug     = sum(1 for m in registry if not m.get("aa_slug"))

    print(f"\n── Registry summary ──────────────────────────────────────────────────")
    print(f"  Total models         : {len(registry)}")
    print(f"  AA data found        : {aa_found}")
    print(f"  AA slug but no data  : {aa_missing}")
    print(f"  No aa_slug (manual)  : {no_slug}")

    print(f"\n── Per-model status ──────────────────────────────────────────────────")
    for m in registry:
        aa_idx   = m.get("aa_intelligence_index")
        aa_str   = f"AA={aa_idx:.0f}" if aa_idx is not None else "AA=—"
        free_str = "free" if m.get("flag_free") else f"${m.get('price_blended_per_1M', 0):.3f}/1M"
        status   = "✓" if m.get("aa_data") else "⚠ manual only"
        print(f"  {m['id']:<48} {aa_str:<8} {free_str:<14} {status}")

    print(f"\nDone. Next step: python3 build_task_taxonomy.py")


if __name__ == "__main__":
    main()