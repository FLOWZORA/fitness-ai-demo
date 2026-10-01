"""
calibrate_threshold.py -- Calibrate MIN_RELEVANCE for a new embedding model
or a new client knowledge base.

WHY THIS EXISTS
---------------
MIN_RELEVANCE = 0.25 was calibrated for text-embedding-3-small on the
FitCoach Pro knowledge base. Cosine similarity scores are NOT universal --
they depend on both the embedding model and the content distribution of
the knowledge base. If you:
  - switch to a different OpenAI embedding model
  - switch to an open-source embedding model
  - load a client whose content is very different in vocabulary and structure

...then 0.25 might be too strict (lots of false "not in knowledge base"
refusals) or too loose (weak matches slip through and cause hallucination).

HOW IT WORKS
------------
1. Loads the knowledge base and builds the vector store (same as production)
2. Runs two sets of queries:
   - POSITIVE queries: questions whose answer IS in the knowledge base
   - NEGATIVE queries: questions whose answer is NOT in the knowledge base
3. Records the top-1 cosine similarity score for each query
4. Suggests a threshold:
   - Should be ABOVE all negative scores (to block off-topic matches)
   - Should be BELOW all positive scores (to allow real answers through)
   - Recommended: 60% of the lowest positive score
5. Prints a calibration report and the exact line to update in config.py

HOW TO USE
----------
1. Update POSITIVE_QUERIES with 8-12 questions you KNOW are answered
   by the client's knowledge base.
2. Update NEGATIVE_QUERIES with 5-8 questions clearly NOT in the content.
3. Run:   python calibrate_threshold.py
4. Update MIN_RELEVANCE in config.py with the recommended value.

When to re-run:
  - Any time you change EMBEDDING_MODEL in config.py
  - Any time you load a new client's knowledge base
  - If MIN_RELEVANCE starts producing too many false refusals or
    too many weak-match answers in production

Run:  python calibrate_threshold.py
"""
import config
config.require_live_mode()

from rag_pipeline import build_vector_store, retrieve

# SAMPLE SIZE GUIDANCE
# --------------------
# Current default: 10 positive + 5 negative queries.
#
# This is appropriate for demos and simple knowledge bases.
# For production client deployments, scale based on content
# complexity:
#
#   Simple KB   (1-2 topics,  <30 chunks) : 10+ pos, 5+ neg
#   Moderate KB (3-5 topics, 30-100 chunks): 25+ pos, 10+ neg
#   Complex KB  (5+ topics, 100+ chunks)  : 50+ pos, 20+ neg
#
# KEY RULE: positive queries must cover every major topic in
# the knowledge base. If a topic has no positive query, the
# calibration cannot detect whether that topic's chunks score
# unusually low -- and the threshold may silently block real
# answers on that topic in production.
#
# For a new client: read through their knowledge base, list
# every distinct topic, and write at least one positive query
# per topic before running calibration.
#
# Negative queries should reflect realistic off-topic inputs
# their actual users might send -- not obviously unrelated
# questions like "what is the stock price of Apple" but
# plausibly adjacent ones like "what supplements help with
# anxiety" (if the KB is fitness-only, not mental health).
#
# These are the FitCoach Pro calibration queries (defaults).
# Replace with client-specific questions for each engagement.
POSITIVE_QUERIES = [
    # Questions whose answers ARE in knowledge_base/
    "How much creatine should I take daily?",
    "What protein intake is needed to build muscle?",
    "How many days per week should a beginner train?",
    "How much sleep does an adult need for recovery?",
    "Is it okay to skip the warm-up before lifting?",
    "What is Zone 2 cardio and why does it matter?",
    "How often should I take a deload week?",
    "What supplement helps with sleep quality?",
    "How much protein is in a cooked chicken breast?",
    "Is high protein safe for someone with kidney disease?",
]

NEGATIVE_QUERIES = [
    # Questions whose answers are NOT in knowledge_base/
    "What is the best cryptocurrency to invest in?",
    "How do I fix a leaking tap?",
    "What are the latest AI research papers on transformers?",
    "How do I file my taxes in India?",
    "What is the fastest car in the world?",
]


def calibrate():
    print(config.banner())
    print("\n" + "=" * 60)
    print("  THRESHOLD CALIBRATION")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Knowledge base  : {config.KNOWLEDGE_BASE_DIR}")
    print(f"  Current MIN_RELEVANCE: {config.MIN_RELEVANCE}")
    print("=" * 60)

    print("\nBuilding vector store...")
    store = build_vector_store()

    print("\n[POSITIVE queries -- answer IS in knowledge base]")
    positive_scores = []
    for q in POSITIVE_QUERIES:
        chunks = retrieve(store, q, k=1)
        score = chunks[0]["relevance"] if chunks else 0.0
        positive_scores.append(score)
        flag = " [WARN: unexpectedly low]" if score < 0.30 else ""
        print(f"  {score:.3f}  {q[:60]}{flag}")

    print("\n[NEGATIVE queries -- answer NOT in knowledge base]")
    negative_scores = []
    for q in NEGATIVE_QUERIES:
        chunks = retrieve(store, q, k=1)
        score = chunks[0]["relevance"] if chunks else 0.0
        negative_scores.append(score)
        flag = " [WARN: unexpectedly high]" if score > 0.25 else ""
        print(f"  {score:.3f}  {q[:60]}{flag}")

    # --- Calibration logic ---
    min_positive = min(positive_scores)
    max_negative = max(negative_scores)

    # Safety margin to keep above the negative ceiling regardless of
    # what the percentage formula produces.
    SAFETY_MARGIN = 0.05

    # Recommended threshold: the HIGHER of:
    #   (a) 60% of the lowest positive score  -- stays well below real answers
    #   (b) negative ceiling + safety margin  -- always above weak matches
    #
    # Using only (a) is fragile: if the negative ceiling is high (e.g. 0.38)
    # and min_positive is 0.50, then 60% = 0.30 < 0.38 -- lets bad matches
    # through. The max() ensures the threshold is always above the negative
    # ceiling even when the 60% heuristic would otherwise fall short.
    pct_based = round(min_positive * 0.60, 3)
    floor_based = round(max_negative + SAFETY_MARGIN, 3)
    recommended = max(pct_based, floor_based)

    print("\n" + "=" * 60)
    print("  CALIBRATION REPORT")
    print("=" * 60)
    print(f"\n  Lowest positive score : {min_positive:.3f}")
    print(f"  Highest negative score: {max_negative:.3f}")
    print(f"  Margin (gap)          : {min_positive - max_negative:.3f}")

    if max_negative >= min_positive:
        print("\n  [PROBLEM] Positive and negative scores OVERLAP.")
        print("  This means the knowledge base may have content that")
        print("  attracts off-topic queries, OR the embedding model")
        print("  has poor discrimination on this content.")
        print("  Consider: revising chunk boundaries, or using a")
        print("  stronger embedding model before setting a threshold.")
    else:
        # Show which formula drove the recommendation
        driver = "60% of positive floor" if pct_based >= floor_based \
                 else f"negative ceiling + {SAFETY_MARGIN} safety margin"
        print(f"\n  Recommended MIN_RELEVANCE: {recommended}  [{driver}]")
        if pct_based < floor_based:
            print(f"  [NOTE] The 60% heuristic ({pct_based}) fell below the")
            print(f"  negative ceiling + margin ({floor_based}).")
            print(f"  The safety floor was applied to prevent weak-match leakage.")
        print(f"\n  Update config.py line:")
        print(f"  MIN_RELEVANCE = {recommended}"
              f"  # calibrated for {config.EMBEDDING_MODEL}")
        print(f"\n  This gives a {min_positive - recommended:.3f} margin below the"
              f" lowest positive score")
        print(f"  and a {recommended - max_negative:.3f} margin above the"
              f" highest negative score.")

        # Check whether the current configured threshold is still safe
        current = config.MIN_RELEVANCE
        margin_to_positive = min_positive - current
        margin_to_negative = current - max_negative

        print(f"\n  Current MIN_RELEVANCE in config.py: {current}")
        if current > min_positive:
            print(f"  [FAIL] Current threshold is ABOVE the positive floor.")
            print(f"  This would block real in-KB answers. Lower to {recommended}.")
        elif margin_to_positive < 0.05:
            print(f"  [WARN] Current threshold is dangerously close to the")
            print(f"  positive floor (only {margin_to_positive:.3f} margin).")
            print(f"  Lower to {recommended} for a safer buffer.")
        elif current < max_negative:
            print(f"  [FAIL] Current threshold is BELOW the negative ceiling.")
            print(f"  Weak off-topic matches are slipping through. Raise to {recommended}.")
        else:
            print(f"  Current threshold is valid for this score distribution.")
            if margin_to_positive < 0.10:
                print(f"  [WARN] Margin to positive floor is narrow ({margin_to_positive:.3f})")
                print(f"  -- consider lowering to {recommended} for safety.")

        if max_negative >= recommended:
            print(f"\n  [WARN] Recommended threshold ({recommended}) is below the")
            print(f"  negative ceiling ({max_negative:.3f}). Use at least"
                  f" {round(max_negative + 0.02, 3)} instead.")

    print("\n  Rule of thumb: if users report too many 'not in knowledge")
    print("  base' refusals on real questions, lower the threshold.")
    print("  If weak-match answers start slipping through, raise it.")
    print("  Re-run this script whenever you change the embedding model.")


if __name__ == "__main__":
    calibrate()
