#!/usr/bin/env python3
"""
build_task_taxonomy.py
======================
Makes one LLM call to expand the 13 WildClawBench task types into a
richer ~32-type taxonomy that maps cleanly to Artificial Analysis
benchmark categories.

Why expand the taxonomy?
  - 13 types is too coarse for a feature-based router.
    "code.generation" covers everything from writing HTML to SAM3 inference.
  - The AA benchmarks distinguish capabilities at a finer level:
    HumanEval tests function-level code, TerminalBench tests shell/CLI,
    GPQA tests scientific reasoning, etc.
  - With 32 types, each benchmark maps to 2-4 task types, and each
    task type maps to 1-3 benchmarks — a clean many-to-many relationship.

The taxonomy is generated once and frozen. It becomes the shared
vocabulary between:
  - build_benchmark_weights.py  (benchmark → task type weights)
  - build_model_features.py     (task type affinity per model)
  - perf_router_inference.py    (classify incoming prompts)

Output: task_taxonomy.json
  {
    "task_types": [
      {
        "id": "code.generation",
        "name": "Code Generation",
        "parent": "code",
        "definition": "...",
        "examples": ["..."],
        "relevant_aa_benchmarks": ["livecodebench", "scicode"],
        "inherited_from_wcb": true   // was in original 13
      },
      ...
    ]
  }

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python3 build_task_taxonomy.py

  # Use a different model
  python3 build_task_taxonomy.py --model gpt-4o-mini --base-url https://api.openai.com/v1

  # Force regeneration even if taxonomy already exists
  python3 build_task_taxonomy.py --force
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"


# ── The 13 original WildClawBench task types ──────────────────────────────────
# These are preserved exactly — we're extending, not replacing.

WCB_TASK_TYPES = [
    "code.debugging",
    "code.generation",
    "code.puzzle_or_algorithm",
    "retrieval.document_extraction",
    "retrieval.repo_code_search",
    "retrieval.web_search",
    "safety.constraint_following",
    "social.conversation_analysis",
    "social.negotiation_or_communication",
    "synthesis.creative_media",
    "synthesis.report_or_summary",
    "synthesis.structured_data",
    "tool_use.file_or_system_operation",
    "tool_use.multi_step_workflow",
]

# ── AA benchmarks available in the model registry ────────────────────────────
# These are the benchmarks we can actually use as features.
# Each has a short description of what it measures.

AA_BENCHMARKS = {
    "aa_intelligence_index": "Overall intelligence — composite of reasoning, knowledge, coding",
    "aa_coding_index":       "Overall coding ability — composite of multiple coding benchmarks",
    "aa_math_index":         "Mathematical reasoning ability",
    "bench_terminal_bench":  "Agentic coding in terminal/shell environments — CLI tool use",
    "bench_tau2_bench":      "Multi-step tool use and API chaining",
    "bench_aa_lcr":          "Long context reasoning — understanding and reasoning over long docs",
    "bench_hle":             "Hard general reasoning — Humanity's Last Exam",
    "bench_gpqa":            "Scientific reasoning — graduate-level science questions",
    "bench_scicode":         "Scientific coding — code generation for science tasks",
    "bench_ifbench":         "Instruction following — adherence to complex instructions",
    "bench_aime":            "Mathematical competition reasoning — AIME problems",
    "bench_livecodebench":   "Practical code generation — real-world coding problems",
    "bench_mmmu_pro":        "Multimodal reasoning — visual + language understanding",
    "bench_math_500":        "Mathematical problem solving — MATH benchmark",
}

# ── Taxonomy generation prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in LLM evaluation and routing system design.
Your task is to design a task taxonomy for a neural routing system that routes
incoming LLM requests to the best model based on task type.

Reply with ONLY valid JSON, no markdown, no explanation, no preamble."""

def build_user_prompt() -> str:
    wcb_list = "\n".join(f"  - {t}" for t in WCB_TASK_TYPES)
    bench_list = "\n".join(
        f"  - {k}: {v}" for k, v in AA_BENCHMARKS.items()
    )

    return f"""Design an extended task taxonomy for an LLM router.

EXISTING TASK TYPES (from WildClawBench — must all be preserved exactly):
{wcb_list}

AVAILABLE BENCHMARKS (that measure model capabilities):
{bench_list}

REQUIREMENTS:
1. Preserve ALL 14 existing task types exactly as-is (mark them with "inherited_from_wcb": true)
2. Add 15-20 NEW task types that:
   - Are meaningfully distinct from existing ones
   - Map cleanly to 1-3 of the available benchmarks
   - Cover gaps in the existing taxonomy (e.g. mathematical reasoning, long-context, multimodal, instruction following, agentic real-world tasks)
3. Use the naming convention: domain.subcategory (e.g. "reasoning.mathematical")
4. Keep parent domains consistent: code, reasoning, retrieval, synthesis, social, safety, tool_use, instruction, multimodal

OUTPUT FORMAT — return exactly this JSON structure:
{{
  "task_types": [
    {{
      "id": "code.generation",
      "name": "Code Generation",
      "parent": "code",
      "definition": "Writing new code, scripts, or programs from a specification or description.",
      "examples": ["write a Python function that...", "implement a REST API endpoint for..."],
      "relevant_aa_benchmarks": ["bench_livecodebench", "bench_scicode", "aa_coding_index"],
      "benchmark_weights": {{"bench_livecodebench": 0.5, "bench_scicode": 0.3, "aa_coding_index": 0.2}},
      "inherited_from_wcb": true
    }},
    ...
  ]
}}

benchmark_weights must sum to 1.0 for each task type.
Only use benchmark IDs from the AVAILABLE BENCHMARKS list above.
Generate 30-34 task types total (14 existing + 16-20 new).
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(model: str, base_url: str, api_key: str) -> dict:
    """Call the LLM and return parsed taxonomy JSON."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"  Calling {model} at {base_url}...")
    print(f"  Generating taxonomy ({len(WCB_TASK_TYPES)} existing + ~18 new types)...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt()},
        ],
        max_tokens=8000,
        temperature=0.2,     # low temp for consistent structured output
        extra_body={"thinking": {"type": "disabled"}},  # disable thinking mode
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: LLM returned invalid JSON: {e}", file=sys.stderr)
        print(f"Raw response (first 500 chars):\n{raw[:500]}", file=sys.stderr)
        sys.exit(1)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_taxonomy(taxonomy: dict) -> list[str]:
    """
    Validate the generated taxonomy.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []
    task_types = taxonomy.get("task_types", [])

    if not task_types:
        return ["No task_types found in response"]

    # Check all WCB types are present
    found_ids = {t["id"] for t in task_types}
    for wcb_type in WCB_TASK_TYPES:
        if wcb_type not in found_ids:
            warnings.append(f"Missing WCB type: {wcb_type}")

    # Check benchmark weights sum to ~1.0
    valid_benchmarks = set(AA_BENCHMARKS.keys())
    for t in task_types:
        tid = t.get("id", "?")
        weights = t.get("benchmark_weights", {})

        # Check all referenced benchmarks exist
        for b in weights:
            if b not in valid_benchmarks:
                warnings.append(f"{tid}: unknown benchmark '{b}'")

        # Check weights sum to 1.0
        total = sum(weights.values())
        if weights and abs(total - 1.0) > 0.05:
            warnings.append(f"{tid}: benchmark_weights sum to {total:.3f}, not 1.0")

    return warnings


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate extended task taxonomy for PerfRouter"
    )
    parser.add_argument("--out",      default=str(_DATA_DIR / "task_taxonomy.json"))
    parser.add_argument("--model",    default="deepseek-v4-flash",
                        help="LLM model for taxonomy generation (default: deepseek-v4-flash)")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key",  default=None,
                        help="API key (default: DEEPSEEK_API_KEY env var)")
    parser.add_argument("--force",    action="store_true",
                        help="Regenerate even if output already exists")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()

    # ── Check if already exists ───────────────────────────────────────────────
    if out_path.exists() and not args.force:
        print(f"Taxonomy already exists: {out_path}")
        print("Pass --force to regenerate.")
        # Load and print summary
        data = json.loads(out_path.read_text())
        task_types = data.get("task_types", [])
        print(f"\nExisting taxonomy: {len(task_types)} task types")
        inherited = sum(1 for t in task_types if t.get("inherited_from_wcb"))
        new_types  = len(task_types) - inherited
        print(f"  Inherited from WCB : {inherited}")
        print(f"  New types          : {new_types}")
        return

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = (
        args.api_key or
        os.environ.get("DEEPSEEK_API_KEY") or
        os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("ERROR: No API key found. Set DEEPSEEK_API_KEY or pass --api-key",
              file=sys.stderr)
        sys.exit(1)

    # ── Generate taxonomy ─────────────────────────────────────────────────────
    print(f"Generating extended task taxonomy...")
    print(f"  Model    : {args.model}")
    print(f"  Base URL : {args.base_url}")
    print(f"  Output   : {out_path}")

    taxonomy = call_llm(args.model, args.base_url, api_key)

    # ── Validate ──────────────────────────────────────────────────────────────
    warnings = validate_taxonomy(taxonomy)
    if warnings:
        print(f"\n⚠ Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✓ Taxonomy validation passed")

    # ── Enrich with metadata ──────────────────────────────────────────────────
    task_types = taxonomy.get("task_types", [])
    all_benchmarks = set(AA_BENCHMARKS.keys())

    for t in task_types:
        # Ensure relevant_aa_benchmarks is consistent with benchmark_weights
        if "benchmark_weights" in t and "relevant_aa_benchmarks" not in t:
            t["relevant_aa_benchmarks"] = list(t["benchmark_weights"].keys())

        # Remove any unknown benchmark references
        weights = t.get("benchmark_weights", {})
        unknown = [b for b in weights if b not in all_benchmarks]
        for b in unknown:
            del weights[b]

        # Normalise weights after removing unknowns
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            t["benchmark_weights"] = {b: round(w/total, 4) for b, w in weights.items()}

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "total_types":    len(task_types),
        "inherited_wcb":  sum(1 for t in task_types if t.get("inherited_from_wcb")),
        "new_types":      sum(1 for t in task_types if not t.get("inherited_from_wcb")),
        "benchmarks_used": sorted(all_benchmarks),
        "task_types":     task_types,
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Task type distribution ────────────────────────────────────────────")
    parents = {}
    for t in task_types:
        p = t.get("parent", t["id"].split(".")[0])
        parents.setdefault(p, []).append(t["id"])

    for parent, types in sorted(parents.items()):
        bar = "█" * len(types)
        print(f"  {parent:<20} {len(types):>2}  {bar}")
        for tid in types:
            tag = " ← wcb" if tid in WCB_TASK_TYPES else " ← new"
            print(f"    {tid}{tag}")

    print(f"\n  Total: {len(task_types)} task types "
          f"({output['inherited_wcb']} from WCB + {output['new_types']} new)")
    print(f"\nSaved to: {out_path}")
    print(f"\nDone. Next step: python3 build_benchmark_weights.py")


if __name__ == "__main__":
    main()