"""
run_pipeline.py — Runs the complete 6-stage audit-to-delivery pipeline.

For a real client, the only things that change are:
  1. ask_broken() in broken_system.py points to THEIR system
  2. CLIENT_NAME in generate_report.py is updated
  3. knowledge_base/ contains THEIR documents

Everything else is identical.
"""
import subprocess
import sys
from llm_client import mode_banner
from kb_change_detector import check_and_recalibrate
check_and_recalibrate()


def stage(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run(module):
    result = subprocess.run([sys.executable, module],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode


def main():
    print(mode_banner())
    print("\n###  FITNESS AI  —  FULL AUDIT-TO-DELIVERY PIPELINE  ###")

    stage("STAGE 1  —  The client's current AI (BEFORE)")
    run("broken_system.py")

    stage("STAGE 2  —  Running the audit")
    run("audit.py")

    stage("STAGE 3  —  The optimized AI (AFTER)")
    run("optimized_system.py")

    stage("STAGE 4  —  Before / After comparison (proof of value)")
    run("compare.py")

    stage("STAGE 5  —  Generating the client deliverable (.docx)")
    run("generate_report.py")

    stage("STAGE 6  —  RAG chatbot build (the upsell)")
    run("rag_pipeline.py")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print("""
  Artifacts produced:
    - audit_results_broken.json   (raw audit data)
    - comparison.json             (before/after metrics)
    - report_data.json            (report source data)
    - AI_Audit_Report.docx        (the client deliverable)
    - vector_store.pkl            (the RAG knowledge base)

  Next step — launch the RAG chat interface:
    python rag_demo.py

  For a REAL client:
    1. Replace ask_broken() with a call to THEIR system.
    2. Add their documents to knowledge_base/ folder.
    3. Update CLIENT_NAME in generate_report.py.
    4. Run this pipeline.
    5. Deliver: AI_Audit_Report.docx + optimized prompt + RAG demo + Loom video.
""")


if __name__ == "__main__":
    main()
