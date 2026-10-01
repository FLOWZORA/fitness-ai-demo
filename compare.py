"""
compare.py — BEFORE/AFTER comparison (your proof of value).

Audits BOTH the broken system and the optimized system, then prints a
side-by-side metrics table. This is the single most important artifact in
your service: quantified, inarguable improvement.

Saves comparison.json for the report generator.
"""
import json
from audit import run_audit
from broken_system import ask_broken
from optimized_system import ask_optimized
from llm_client import mode_banner


def _row(label, before, after, higher_better=True, is_score=False):
    if before is None:
        before = 0
    if after is None:
        after = 0
    unit = "" if is_score else "%"
    arrow = "^" if after > before else ("v" if after < before else "=")
    delta = after - before
    delta_str = f"{'+' if delta >= 0 else ''}{round(delta, 2)}{unit}"
    return {
        "metric": label,
        "before": f"{before}{unit}",
        "after": f"{after}{unit}",
        "change": f"{arrow} {delta_str}",
        "_before_num": before,
        "_after_num": after,
    }


def build_comparison():
    before = run_audit(ask_broken, "BEFORE")["summary"]
    after = run_audit(ask_optimized, "AFTER")["summary"]

    rows = [
        _row("Overall pass rate", before["overall_pass_rate_pct"],
             after["overall_pass_rate_pct"]),
        _row("Safety screening", before["safety_screening_pct"],
             after["safety_screening_pct"]),
        _row("Hallucination avoidance", before["hallucination_avoidance_pct"],
             after["hallucination_avoidance_pct"]),
        _row("Context use", before["context_use_pct"],
             after["context_use_pct"]),
        _row("Disclaimer presence", before["disclaimer_presence_pct"],
             after["disclaimer_presence_pct"]),
        _row("Consistency score", before["avg_consistency_score"],
             after["avg_consistency_score"], is_score=True),
    ]
    return {"before": before, "after": after, "rows": rows}


def print_table(comp):
    rows = comp["rows"]
    print("\n" + "=" * 64)
    print("  BEFORE / AFTER  —  AI SYSTEM AUDIT & OPTIMIZATION")
    print("=" * 64)
    print(f"  {'Metric':<26}{'Before':>10}{'After':>10}{'Change':>16}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r['metric']:<26}{r['before']:>10}{r['after']:>10}"
              f"{r['change']:>16}")
    print("=" * 64)


if __name__ == "__main__":
    print(mode_banner())
    comp = build_comparison()
    print_table(comp)
    with open("comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
    print("\nSaved -> comparison.json")
