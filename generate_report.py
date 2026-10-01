"""
generate_report.py — Collects all audit + comparison data and writes
report_data.json, then calls the Node generator to produce the client .docx.

This is the DELIVERABLE step. For a real client you'd swap in their name and
their system; everything else is identical.
"""
import json
import subprocess
import os
from datetime import date

import config
from audit import run_audit
from broken_system import ask_broken
from optimized_system import ask_optimized
from compare import build_comparison

CLIENT_NAME = config.CLIENT_NAME
AUDITOR_NAME = config.AUDITOR_NAME


def collect_findings(broken_audit):
    """Turn raw audit cases into human-readable findings for the report."""
    findings = []
    for c in broken_audit["cases"]:
        if c["category"] == "consistency":
            continue
        checks = c.get("checks", {})
        problems = []
        if checks.get("screened_danger") is False:
            problems.append("Failed to screen a safety-critical request "
                            "(injury/pregnancy/disordered eating).")
        if checks.get("avoided_dangerous_advice") is False:
            toks = checks.get("_dangerous_tokens_found", [])
            problems.append(f"Gave dangerous advice containing: "
                            f"{', '.join(toks)}.")
        if checks.get("avoided_hallucination") is False:
            toks = checks.get("_false_claims_found", [])
            problems.append(f"Stated false specifics: {', '.join(toks)}.")
        if checks.get("asked_for_context") is False:
            problems.append("Produced generic advice without asking for the "
                            "user's goal, experience, or constraints.")
        if checks.get("had_disclaimer") is False:
            problems.append("Missing required safety disclaimer.")
        if problems:
            findings.append({
                "id": c["id"],
                "category": c["category"],
                "input": c["input"],
                "problems": problems,
                "actual_output": (c["response"][:300] + "...")
                                 if len(c["response"]) > 300 else c["response"],
            })
    return findings


def main():
    print(config.banner())
    print("\nGenerating client report...")

    broken_audit = run_audit(ask_broken, "BEFORE")
    comp = build_comparison()
    findings = collect_findings(broken_audit)

    # severity rollup
    danger_findings = [f for f in findings if f["category"] == "danger"]
    halluc_findings = [f for f in findings if f["category"] == "hallucination"]

    cost_line = ""
    if config.LIVE_MODE:
        import api_client
        cost_line = f" | API cost this run: ${api_client.usage.estimated_cost_usd}"

    report_data = {
        "client_name": CLIENT_NAME,
        "auditor_name": AUDITOR_NAME,
        "date": date.today().strftime("%B %d, %Y"),
        "mode": ("LIVE — real API calls" + cost_line) if config.LIVE_MODE
                 else "MOCK MODE (illustrative, offline)",
        "summary_before": comp["before"],
        "summary_after": comp["after"],
        "comparison_rows": comp["rows"],
        "findings": findings,
        "counts": {
            "total_findings": len(findings),
            "critical_safety": len(danger_findings),
            "hallucinations": len(halluc_findings),
        },
    }

    with open("report_data.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved -> report_data.json")

    # Generate the .docx via Node
    result = subprocess.run(
        ["node", "build_report_docx.js"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Report generation error:")
        print(result.stderr)
    else:
        print("Saved -> AI_Audit_Report.docx")


if __name__ == "__main__":
    main()
