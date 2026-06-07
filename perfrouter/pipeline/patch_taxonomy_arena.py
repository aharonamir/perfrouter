#!/usr/bin/env python3
"""
patch_taxonomy_arena.py
=======================
Patches task_taxonomy.json to add Arena ELO scores as benchmark weights
for task types that were previously using AA-Intel as a catch-all.

Run this AFTER fetch_arena_data.py and BEFORE build_benchmark_weights.py.

Usage:
  python3 patch_taxonomy_arena.py
  python3 build_benchmark_weights.py  # re-run to pick up new weights
  python3 build_model_features.py     # re-run to use updated weight matrix
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data"

# Updated benchmark weights per task type — incorporating Arena ELO signals
# where AA benchmarks had no good coverage.
#
# Arena ELO categories:
#   arena_text_elo     → conversation, creative writing, general text quality
#   arena_search_elo   → search and retrieval
#   arena_code_elo     → coding quality (human preference)
#   arena_vision_elo   → visual understanding
#   arena_document_elo → document understanding and extraction

ARENA_UPDATES = {
    # ── Social tasks — text Arena is the best available signal ────────────────
    "social.conversation_analysis": {
        "arena_text_elo":        0.7,
        "aa_intelligence_index": 0.3,
    },
    "social.negotiation_or_communication": {
        "arena_text_elo":        0.7,
        "aa_intelligence_index": 0.3,
    },
    "social.sentiment_analysis": {
        "arena_text_elo":        0.8,
        "aa_intelligence_index": 0.2,
    },

    # ── Creative synthesis — text Arena covers subjective quality ─────────────
    "synthesis.creative_media": {
        "arena_text_elo":        0.6,
        "aa_intelligence_index": 0.4,
    },
    "synthesis.creative_writing": {
        "arena_text_elo":        0.8,
        "aa_intelligence_index": 0.2,
    },
    "synthesis.report_or_summary": {
        "arena_document_elo": 0.5,
        "bench_aa_lcr":       0.3,
        "arena_text_elo":     0.2,
    },
    "synthesis.structured_data": {
        "arena_document_elo":    0.6,
        "aa_intelligence_index": 0.4,
    },
    "synthesis.data_analysis_report": {
        "arena_document_elo":    0.5,
        "bench_aa_lcr":          0.3,
        "aa_intelligence_index": 0.2,
    },

    # ── Retrieval — search Arena is the best signal ───────────────────────────
    "retrieval.web_search": {
        "arena_search_elo":      0.7,
        "aa_intelligence_index": 0.3,
    },
    "retrieval.knowledge_qa": {
        "arena_search_elo":      0.6,
        "aa_intelligence_index": 0.4,
    },
    "retrieval.document_extraction": {
        "arena_document_elo": 0.6,
        "bench_aa_lcr":       0.4,
    },

    # ── Multimodal — vision Arena supplements MMMU ────────────────────────────
    "multimodal.visual_reasoning": {
        "arena_vision_elo": 0.7,
        "bench_mmmu_pro":   0.3,
    },
    "multimodal.diagram_understanding": {
        "arena_vision_elo": 0.7,
        "bench_mmmu_pro":   0.3,
    },

    # ── Code — Arena code ELO as additional signal ────────────────────────────
    "code.generation": {
        "arena_code_elo":    0.3,
        "bench_livecodebench": 0.4,
        "bench_scicode":     0.2,
        "aa_coding_index":   0.1,
    },
    "code.debugging": {
        "arena_code_elo":    0.3,
        "bench_livecodebench": 0.5,
        "aa_coding_index":   0.2,
    },
    "code.code_review": {
        "arena_code_elo":    0.4,
        "aa_coding_index":   0.4,
        "bench_livecodebench": 0.2,
    },
    "code.scientific_coding": {
        "arena_code_elo":    0.2,
        "bench_scicode":     0.5,
        "bench_livecodebench": 0.3,
    },
}


def main():
    taxonomy_path = _DATA_DIR / "task_taxonomy.json"
    if not taxonomy_path.exists():
        print(f"ERROR: {taxonomy_path} not found. Run build_task_taxonomy.py first.")
        raise SystemExit(1)

    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = json.load(f)

    task_types = taxonomy.get("task_types", [])
    updated    = 0
    not_found  = []

    for t in task_types:
        tid = t["id"]
        if tid in ARENA_UPDATES:
            old_weights = t.get("benchmark_weights", {})
            new_weights = ARENA_UPDATES[tid]

            # Validate weights sum to 1.0
            total = sum(new_weights.values())
            if abs(total - 1.0) > 0.01:
                print(f"  WARN: {tid} weights sum to {total:.3f}, normalising...")
                new_weights = {k: round(v/total, 4) for k, v in new_weights.items()}

            t["benchmark_weights"]      = new_weights
            t["relevant_aa_benchmarks"] = list(new_weights.keys())
            updated += 1
            print(f"  ✓ {tid}")
            print(f"      was : {list(old_weights.keys())}")
            print(f"      now : {list(new_weights.keys())}")

    # Check if any ARENA_UPDATES keys weren't found in taxonomy
    found_ids = {t["id"] for t in task_types}
    for tid in ARENA_UPDATES:
        if tid not in found_ids:
            not_found.append(tid)

    if not_found:
        print(f"\n  WARN: These task types not found in taxonomy:")
        for tid in not_found:
            print(f"    - {tid}")

    taxonomy["arena_updated"] = True
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated {updated} task types in {taxonomy_path}")
    print(f"\nNext steps:")
    print(f"  python3 build_benchmark_weights.py  # regenerate weight matrix")
    print(f"  python3 build_model_features.py     # recompute model features")


if __name__ == "__main__":
    main()