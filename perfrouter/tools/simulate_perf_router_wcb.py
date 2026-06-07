#!/usr/bin/env python3
"""
simulate_perf_router_wcb.py
============================
Runs PerfRouter's routing simulation against WildClawBench task prompts
WITHOUT executing any task or calling any model API.

For each WCB task markdown file, it:
  1. Parses the ## Prompt section
  2. Runs PerfRouter's sentence-BERT classifier
  3. Records which model would be chosen and why

This lets you validate routing decisions in seconds instead of waiting
hours for the full benchmark to complete.

─────────────────────────────────────────────────────────────────────
Usage:
  # Simulate against all tasks in a WCB repo
  python3 simulate_perf_router_wcb.py \\
      --tasks-dir ~/workspace/WildClawBench/tasks \\
      --router    perf_router.pkl \\
      --taxonomy  task_taxonomy.json \\
      --registry  model_registry.json \\
      --features  model_features.csv \\
      --baseline  deepseek/deepseek-v4-pro \\
      --out       perf_router_wcb_simulation.csv

  # Simulate against a single category
  python3 simulate_perf_router_wcb.py \\
      --tasks-dir ~/workspace/WildClawBench/tasks/02_Code_Intelligence

  # Compare against TRouter's actual WCB results
  python3 simulate_perf_router_wcb.py \\
      --tasks-dir  ~/workspace/WildClawBench/tasks \\
      --wcb-csv    ~/workspace/WildClawBench/wcb_training_labeled.csv \\
      --out        perf_router_wcb_simulation.csv
─────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ── Markdown parser ───────────────────────────────────────────────────────────

def parse_task_markdown(md_path: Path) -> dict | None:
    """
    Parse a WildClawBench task markdown file.

    Expected structure:
      ---
      id: <task_id>
      name: <task_name>
      category: <category>
      ...
      ---
      ## Prompt
      <prompt text>
      ## Expected Behavior
      ...

    Returns dict with: id, name, category, prompt, md_path
    Returns None if the file can't be parsed.
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  WARN: could not read {md_path}: {e}")
        return None

    # ── Parse YAML frontmatter ────────────────────────────────────────────────
    frontmatter = {}
    fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                frontmatter[key.strip()] = val.strip()

    # ── Extract ## Prompt section ─────────────────────────────────────────────
    # Find everything between ## Prompt and the next ## heading
    prompt_match = re.search(
        r"##\s*Prompt\s*\n(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not prompt_match:
        return None

    prompt_text = prompt_match.group(1).strip()
    if not prompt_text:
        return None

    # Derive task_name and category from path if not in frontmatter
    # WCB path pattern: tasks/01_Productivity_Flow/task_1_arxiv_digest.md
    parts      = md_path.parts
    category   = frontmatter.get("category", "")
    task_id    = frontmatter.get("id", md_path.stem)
    task_name  = frontmatter.get("name", md_path.stem)

    if not category:
        # Try to infer from directory name
        for part in parts:
            if re.match(r"\d+_", part):
                category = part
                break

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "category":  category,
        "prompt":    prompt_text[:2000],  # first 2000 chars is enough for classification
        "md_path":   str(md_path),
    }


def find_task_files(tasks_dir: Path) -> list[Path]:
    """Recursively find all .md task files in the tasks directory."""
    md_files = sorted(tasks_dir.rglob("*.md"))
    # Filter out README files and non-task files
    return [
        f for f in md_files
        if not f.name.lower().startswith("readme")
        and not f.name.lower().startswith("index")
    ]


# ── Load WCB ground truth for comparison ─────────────────────────────────────

def load_wcb_results(csv_path: Path | None) -> dict:
    """
    Load actual WCB benchmark results for comparison.
    Returns {task_name → {model → score}} if csv_path provided.
    """
    if csv_path is None or not csv_path.exists():
        return {}

    results = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_name = row["task_name"]
            model     = row["model"]
            score     = float(row["overall_score"])
            results[task_name][model] = score
    return results


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(
    task_files:            list[Path],
    router,
    wcb_results:           dict,
    degradation_threshold: float = 0.0,
) -> list[dict]:
    """
    Run PerfRouter against all task files.
    Returns list of routing decision dicts.
    """
    rows = []

    for md_path in task_files:
        task = parse_task_markdown(md_path)
        if task is None:
            continue

        # Route using task prompt
        try:
            decision = router.route(
                task["prompt"],
                degradation_threshold=degradation_threshold
            )
        except Exception as e:
            print(f"  ERROR routing {task['task_id']}: {e}")
            continue

        # Look up actual WCB scores if available
        task_name    = task["task_name"]
        actual_scores = wcb_results.get(task_name, {})

        # What did the chosen model actually score in WCB?
        chosen_model_short = decision["decision_model"].split("/")[-1]
        actual_score_chosen = None
        for wcb_model, score in actual_scores.items():
            if wcb_model in decision["decision_model"] or \
               decision["decision_model"] in wcb_model or \
               chosen_model_short in wcb_model:
                actual_score_chosen = score
                break

        # What was the best possible score in WCB?
        best_actual_score = max(actual_scores.values()) if actual_scores else None

        row = {
            # Task info
            "task_id":          task["task_id"],
            "task_name":        task["task_name"],
            "category":         task["category"],

            # PerfRouter decision
            "decision_model":   decision["decision_model"],
            "task_type":        decision["task_type"],
            "top_task_type_1":  decision["top_k_task_types"][0]["task_type"],
            "top_sim_1":        round(decision["top_k_task_types"][0]["similarity"], 3),
            "top_task_type_2":  decision["top_k_task_types"][1]["task_type"] if len(decision["top_k_task_types"]) > 1 else "",
            "top_sim_2":        round(decision["top_k_task_types"][1]["similarity"], 3) if len(decision["top_k_task_types"]) > 1 else 0,
            "predicted_quality": decision["predicted_quality"],
            "cost_per_1M":      decision["cost_per_1M"],
            "cost_saved_pct":   decision["cost_saved_pct"],
            "utility":          decision["decision_utility"],
            "alpha":            decision["alpha"],
            "inference_ms":     decision["inference_ms"],

            # WCB ground truth (if available)
            "actual_score_chosen": actual_score_chosen,
            "best_actual_score":   best_actual_score,
            "routing_gap":         round(best_actual_score - actual_score_chosen, 4)
                                   if (actual_score_chosen is not None and best_actual_score is not None)
                                   else None,
        }
        rows.append(row)

    return rows


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]):
    if not rows:
        print("No rows to summarise.")
        return

    print(f"\n{'═'*72}")
    print(f"  PerfRouter × WildClawBench Simulation Summary")
    print(f"{'═'*72}")
    print(f"  Tasks simulated : {len(rows)}")

    # Model distribution
    model_counts = defaultdict(int)
    for r in rows:
        model_counts[r["decision_model"].split("/")[-1]] += 1

    print(f"\n── Model selection distribution ──────────────────────────────────────")
    total = len(rows)
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        bar = "█" * count
        print(f"  {model:<42} {count:>3} tasks ({pct:>5.1f}%)  {bar}")

    # Cost savings
    free_tasks    = sum(1 for r in rows if r["cost_per_1M"] == 0)
    total_savings = sum(r["cost_saved_pct"] for r in rows) / len(rows)
    print(f"\n── Cost analysis ─────────────────────────────────────────────────────")
    print(f"  Free model routes  : {free_tasks}/{total} ({free_tasks/total*100:.1f}%)")
    print(f"  Avg cost saved     : {total_savings:.1f}% vs baseline")

    # Task type distribution
    type_counts = defaultdict(int)
    for r in rows:
        type_counts[r["task_type"]] += 1

    print(f"\n── Task type classification distribution ─────────────────────────────")
    for task_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"  {task_type:<45} {count:>3}  {bar}")

    # WCB comparison (if available)
    rows_with_wcb = [r for r in rows if r["routing_gap"] is not None]
    if rows_with_wcb:
        avg_gap      = sum(r["routing_gap"] for r in rows_with_wcb) / len(rows_with_wcb)
        acceptable   = sum(1 for r in rows_with_wcb if r["routing_gap"] <= 0.10)
        avg_chosen   = sum(r["actual_score_chosen"] for r in rows_with_wcb) / len(rows_with_wcb)
        avg_best     = sum(r["best_actual_score"]   for r in rows_with_wcb) / len(rows_with_wcb)

        print(f"\n── WCB ground truth comparison ({len(rows_with_wcb)} tasks with data) ────────────")
        print(f"  Avg chosen model score : {avg_chosen:.4f}")
        print(f"  Avg best possible score: {avg_best:.4f}")
        print(f"  Avg routing gap        : {avg_gap:.4f}")
        print(f"  Acceptable routes (≤0.10 gap): {acceptable}/{len(rows_with_wcb)} "
              f"({acceptable/len(rows_with_wcb)*100:.1f}%)")

    # Per-category breakdown
    print(f"\n── Per-category routing ──────────────────────────────────────────────")
    cat_rows = defaultdict(list)
    for r in rows:
        cat_rows[r["category"]].append(r)

    for cat in sorted(cat_rows.keys()):
        cat_data = cat_rows[cat]
        models_chosen = defaultdict(int)
        for r in cat_data:
            models_chosen[r["decision_model"].split("/")[-1]] += 1
        top_model = max(models_chosen.items(), key=lambda x: x[1])
        free_pct  = sum(1 for r in cat_data if r["cost_per_1M"] == 0) / len(cat_data) * 100
        print(f"  {cat:<40} {len(cat_data):>2} tasks  "
              f"top→{top_model[0][:20]}  free={free_pct:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Simulate PerfRouter routing on WildClawBench tasks"
    )
    parser.add_argument("--tasks-dir",  required=True,
                        help="Path to WildClawBench tasks directory")
    parser.add_argument("--router",     default=str(_MODELS_DIR / "perf_router.pkl"))
    parser.add_argument("--taxonomy",   default=str(_DATA_DIR / "task_taxonomy.json"))
    parser.add_argument("--registry",   default=str(_DATA_DIR / "model_registry.json"))
    parser.add_argument("--features",   default=str(_DATA_DIR / "model_features.csv"))
    parser.add_argument("--baseline",   default="deepseek/deepseek-v4-pro",
                        help="Baseline model for cost comparison")
    parser.add_argument("--cost-weight",type=float, default=0.3)
    parser.add_argument("--wcb-csv",    default=None,
                        help="wcb_training_labeled.csv for ground truth comparison")
    parser.add_argument("--out",        default="perf_router_wcb_simulation.csv")
    parser.add_argument("--degradation-threshold", type=float, default=0.0,
                        help="Quality degradation threshold (e.g. 0.10 = accept "
                             "10%% drop in quality to get a cheaper model)")
    parser.add_argument("--min-similarity", type=float, default=0.20,
                        help="Min task-type similarity before routing to cheapest "
                             "model (default: 0.20). Queries below this threshold "
                             "are treated as ambiguous.")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir).expanduser()
    if not tasks_dir.exists():
        print(f"ERROR: --tasks-dir not found: {tasks_dir}")
        sys.exit(1)

    # ── Load PerfRouter ───────────────────────────────────────────────────────
    from perfrouter.inference.perf_router_inference import PerfRouterInference

    print("Initialising PerfRouter...")
    router = PerfRouterInference(
        router_path              = args.router,
        taxonomy_path            = args.taxonomy,
        registry_path            = args.registry,
        features_path            = args.features,
        cost_weight              = args.cost_weight,
        baseline_model           = args.baseline,
        min_similarity_threshold = args.min_similarity,
    )

    # ── Find task files ───────────────────────────────────────────────────────
    print(f"\nScanning tasks in {tasks_dir}...")
    task_files = find_task_files(tasks_dir)
    print(f"  Found {len(task_files)} task files")

    if not task_files:
        print("ERROR: no task .md files found")
        sys.exit(1)

    # ── Load WCB ground truth ─────────────────────────────────────────────────
    wcb_path    = Path(args.wcb_csv).expanduser() if args.wcb_csv else None
    wcb_results = load_wcb_results(wcb_path)
    if wcb_results:
        print(f"  Loaded WCB results for {len(wcb_results)} tasks")

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"\nSimulating routing for {len(task_files)} tasks "
          f"(degradation_threshold={args.degradation_threshold})...")
    rows = simulate(task_files, router, wcb_results,
                    degradation_threshold=args.degradation_threshold)
    print(f"  Completed {len(rows)} routing decisions")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if rows:
        out_path   = Path(args.out).expanduser()
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to: {out_path}")

        # ── Print per-task table ──────────────────────────────────────────────
        print(f"\n── Per-task routing decisions ────────────────────────────────────────")
        print(f"  {'task_name':<42} {'task_type':<32} {'model':<20} "
              f"{'cost%':>6} {'gap':>6}")
        print("  " + "─" * 110)
        for r in rows:
            short_model = r["decision_model"].split("/")[-1][:20]
            gap_str = f"{r['routing_gap']:+.3f}" if r["routing_gap"] is not None else "  — "
            print(f"  {r['task_name']:<42} {r['task_type']:<32} "
                  f"{short_model:<20} {r['cost_saved_pct']:>+5.0f}% {gap_str:>6}")

    # ── Final cost savings summary ────────────────────────────────────────────
    if rows:
        total_tasks  = len(rows)
        free_tasks   = sum(1 for r in rows if r["cost_per_1M"] == 0)
        paid_tasks   = total_tasks - free_tasks
        avg_saved    = sum(r["cost_saved_pct"] for r in rows) / total_tasks

        # Dollar estimate using ~5K tokens per task (WCB average)
        est_tokens   = 5_000
        factor       = est_tokens / 1_000_000
        baseline_usd = router._baseline_cost
        total_base   = total_tasks * baseline_usd * factor
        total_actual = sum(r["cost_per_1M"] * factor for r in rows)
        total_saved  = total_base - total_actual

        print(f"\n{'═'*72}")
        print(f"  Cost Savings Summary")
        print(f"{'═'*72}")
        print(f"  Tasks simulated       : {total_tasks}")
        print(f"  Free model routes     : {free_tasks}  ({free_tasks/total_tasks*100:.1f}%)")
        print(f"  Paid model routes     : {paid_tasks}  ({paid_tasks/total_tasks*100:.1f}%)")
        print(f"  Avg cost saved        : {avg_saved:+.1f}% vs baseline")
        print(f"")
        print(f"  Estimated at ~{est_tokens:,} tokens/task:")
        print(f"  Baseline total        : ${total_base:.4f}  "
              f"({total_tasks} × ${baseline_usd:.3f}/1M)")
        print(f"  PerfRouter total      : ${total_actual:.4f}")
        print(f"  Saved                 : ${total_saved:.4f}  "
              f"({total_saved/total_base*100:.1f}% reduction)")
        print(f"{'═'*72}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()