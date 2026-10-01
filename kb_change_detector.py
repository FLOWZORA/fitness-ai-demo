"""
kb_change_detector.py -- Auto-detect knowledge base changes and re-calibrate.

WHY THIS EXISTS
---------------
When a client adds or updates documents in their knowledge base, three things
need to happen in order:
  1. The vector store must be rebuilt (old embeddings are stale)
  2. The relevance threshold must be recalibrated (score distributions shift
     when content changes -- a threshold calibrated on 21 chunks may no longer
     be correct after 35 chunks are added)
  3. The new threshold must be written back to config.py automatically

Without this, a client adding a new document would silently degrade retrieval
quality -- the old threshold might now block real answers or let weak matches
through, with no warning and no automatic correction.

HOW IT WORKS
------------
1. Computes a SHA-256 hash of all .txt files in the knowledge base
2. Compares against the last recorded hash stored in kb_hash.txt
3. If changed (or no record exists): rebuilds vector store + re-calibrates
4. If unchanged: skips both steps (fast path, no API calls, no cost)
5. Records the new hash after a successful rebuild+calibration cycle

WHEN TO CALL IT
---------------
Call check_and_recalibrate() at the top of:
  - run_pipeline.py  (before the audit stages)
  - app.py           (before build_vector_store on startup)
  - Any script that uses the RAG pipeline

The function is fast when nothing changed (hash compare is instant).
It only makes API calls when the knowledge base has actually changed.

WHAT IT COSTS
-------------
A recalibration run (10 positive + 5 negative queries) costs roughly
$0.002 in embedding API calls. This is negligible -- it only runs when
the knowledge base actually changes, not on every startup.

Run standalone:  python kb_change_detector.py
"""
import hashlib
import json
from pathlib import Path
import config

HASH_FILE = config.PROJECT_ROOT / "kb_hash.txt"
CALIBRATION_RESULTS_FILE = config.PROJECT_ROOT / "calibration_results.json"


def compute_kb_hash() -> str:
    """Hash all .txt files in the knowledge base, sorted for determinism."""
    hasher = hashlib.sha256()
    kb_files = sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt"))
    for f in kb_files:
        # Hash filename + content so renames are also detected
        hasher.update(f.name.encode())
        hasher.update(f.read_bytes())
    return hasher.hexdigest()


def load_last_hash() -> str | None:
    """Load the previously recorded hash, or None if no record exists."""
    if HASH_FILE.exists():
        return HASH_FILE.read_text().strip()
    return None


def save_hash(hash_value: str):
    """Persist the current hash for the next comparison."""
    HASH_FILE.write_text(hash_value)


def save_calibration_results(results: dict):
    """Persist calibration results so they can be referenced later."""
    CALIBRATION_RESULTS_FILE.write_text(json.dumps(results, indent=2))


def kb_changed() -> bool:
    """Return True if the knowledge base has changed since the last check."""
    current_hash = compute_kb_hash()
    last_hash = load_last_hash()
    return current_hash != last_hash


def check_and_recalibrate(force: bool = False, verbose: bool = True) -> dict:
    """
    Check whether the knowledge base has changed. If so (or if force=True):
      1. Rebuild the vector store from scratch
      2. Re-run threshold calibration
      3. Update MIN_RELEVANCE in config.py automatically
      4. Save the new hash and calibration results

    Returns a dict with:
      - changed: bool -- whether a rebuild/recalibration was triggered
      - recommended: float -- the calibrated MIN_RELEVANCE value
      - min_positive: float
      - max_negative: float
      - gap: float
      - driver: str -- which formula drove the recommendation

    Safe to call on every startup -- fast path (hash match) takes <1ms.
    """
    if not config.LIVE_MODE:
        if verbose:
            print("[kb_change_detector] MOCK MODE -- skipping calibration.")
        return {"changed": False, "recommended": config.MIN_RELEVANCE}

    from pipeline_logger import log

    current_hash = compute_kb_hash()
    last_hash = load_last_hash()

    if not force and current_hash == last_hash:
        if verbose:
            print("[kb_change_detector] Knowledge base unchanged -- "
                  "using existing vector store and threshold.")
        return {"changed": False, "recommended": config.MIN_RELEVANCE}

    # Log the knowledge base change event before rebuilding
    files_before = (load_last_hash() and
                    [f.name for f in sorted(
                     config.KNOWLEDGE_BASE_DIR.glob("*.txt"))]) or []
    files_after = [f.name for f in sorted(
                   config.KNOWLEDGE_BASE_DIR.glob("*.txt"))]
    log.kb_change(last_hash, current_hash, files_before, files_after)

    if verbose:
        if force:
            print("[kb_change_detector] Force rebuild requested.")
        else:
            print("[kb_change_detector] Knowledge base CHANGED -- "
                  "rebuilding vector store and recalibrating threshold...")

    # Step 1: Force rebuild the vector store
    import shutil
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        if verbose:
            print("  Cleared stale vector store.")

    from rag_pipeline import build_vector_store, retrieve
    store = build_vector_store()
    if verbose:
        print("  Vector store rebuilt.")

    # Step 2: Run calibration queries
    # These are the same queries from calibrate_threshold.py
    # In a real client project, update these to match their domain
    from calibrate_threshold import POSITIVE_QUERIES, NEGATIVE_QUERIES
    SAFETY_MARGIN = 0.05

    positive_scores = []
    for q in POSITIVE_QUERIES:
        chunks = retrieve(store, q, k=1)
        score = chunks[0]["relevance"] if chunks else 0.0
        positive_scores.append(score)

    negative_scores = []
    for q in NEGATIVE_QUERIES:
        chunks = retrieve(store, q, k=1)
        score = chunks[0]["relevance"] if chunks else 0.0
        negative_scores.append(score)

    min_positive = min(positive_scores)
    max_negative = max(negative_scores)
    gap = round(min_positive - max_negative, 3)

    # Step 3: Compute recommended threshold using the two-constraint formula
    pct_based = round(min_positive * 0.60, 3)
    floor_based = round(max_negative + SAFETY_MARGIN, 3)
    recommended = max(pct_based, floor_based)
    driver = "60% heuristic" if pct_based >= floor_based else "safety floor"

    results = {
        "changed": True,
        "min_positive": round(min_positive, 3),
        "max_negative": round(max_negative, 3),
        "gap": gap,
        "recommended": recommended,
        "driver": driver,
        "margin_below_positive": round(min_positive - recommended, 3),
        "margin_above_negative": round(recommended - max_negative, 3),
    }

    # Step 4: Update MIN_RELEVANCE in config.py automatically
    _update_config_threshold(recommended, results)

    # Step 5: Persist hash and results, log the calibration event
    save_hash(current_hash)
    save_calibration_results(results)
    log.calibration(results)

    if verbose:
        print(f"\n  Calibration complete:")
        print(f"    positive floor  : {min_positive:.3f}")
        print(f"    negative ceiling: {max_negative:.3f}")
        print(f"    gap             : {gap:.3f}")
        print(f"    recommended     : {recommended}  [{driver}]")
        print(f"    config.py updated: MIN_RELEVANCE = {recommended}")

    return results


def _update_config_threshold(new_value: float, results: dict):
    """
    Rewrite the MIN_RELEVANCE line in config.py with the new calibrated value.
    Preserves all other content in the file exactly.
    """
    config_path = config.PROJECT_ROOT / "config.py"
    content = config_path.read_text(encoding="utf-8")

    # Find and replace the MIN_RELEVANCE = ... line
    import re
    new_line = (
        f"MIN_RELEVANCE = {new_value}"
        f"   # calibrated: {config.EMBEDDING_MODEL} | "
        f"pos_floor={results['min_positive']} "
        f"neg_ceil={results['max_negative']} "
        f"gap={results['gap']} "
        f"driver={results['driver']}"
    )
    updated = re.sub(
        r"^MIN_RELEVANCE\s*=.*$",
        new_line,
        content,
        flags=re.MULTILINE,
    )
    config_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    print(config.banner())
    print("\nRunning knowledge base change detection...\n")

    # Show current state
    files = sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt"))
    print(f"Knowledge base: {len(files)} files")
    for f in files:
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")

    current_hash = compute_kb_hash()
    last_hash = load_last_hash()
    print(f"\nCurrent hash : {current_hash[:16]}...")
    print(f"Recorded hash: {last_hash[:16] if last_hash else 'none (first run)'}...")
    print(f"Changed      : {current_hash != last_hash}")
    print()

    results = check_and_recalibrate(verbose=True)

    if not results["changed"]:
        print("\nNo changes detected. Pipeline is already calibrated.")
        print(f"Current MIN_RELEVANCE: {config.MIN_RELEVANCE}")
    else:
        print(f"\nDone. config.py updated with MIN_RELEVANCE = {results['recommended']}")
        print("Run the pipeline normally -- it will use the new threshold.")
