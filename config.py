"""
config.py — Central configuration for the API (live) version of the pipeline.

This is the ONE file you edit when:
  - switching between your own key and a client's key
  - changing which client's system you're auditing
  - tuning model choices without touching pipeline code

Everything else in the project reads from here.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------
# API / MODEL CONFIG
# ---------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
LIVE_MODE = OPENAI_API_KEY.startswith("sk-")

# Models — change these per-project if a client wants a specific model
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Cost reference (USD, approx, June 2026 pricing) — printed in reports
CHAT_MODEL_COST_PER_1M_INPUT = 0.15
CHAT_MODEL_COST_PER_1M_OUTPUT = 0.60
EMBEDDING_COST_PER_1M_TOKENS = 0.02

# ---------------------------------------------------------------
# CLIENT IDENTITY — change these per real engagement
# ---------------------------------------------------------------
CLIENT_NAME = os.environ.get("CLIENT_NAME", "FitCoach Pro (Demo Client)")
AUDITOR_NAME = os.environ.get("AUDITOR_NAME", "Vaibhav Tiwari — AI Systems Audit & Optimization")

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
LOCAL_STORE_PATH = PROJECT_ROOT / "vector_store.pkl"  # legacy offline fallback

# ---------------------------------------------------------------
# RAG TUNING
# ---------------------------------------------------------------
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4

# MIN_RELEVANCE -- CALIBRATED PER CLIENT, NOT A UNIVERSAL CONSTANT.
#
# This threshold controls the relevance gate: if even the best retrieved
# chunk scores below this value, the system refuses to answer rather than
# risk hallucinating a weak-match response.
#
# 0.299 was calibrated for:
#   - Embedding model : text-embedding-3-small
#   - Knowledge base  : FitCoach Pro (fitness/nutrition, ~21 chunks)
#   - Positive floor  : 0.499 (lowest score a real in-KB answer scores)
#   - Negative ceiling: 0.165 (highest score a clearly off-topic query gets)
#   - Gap             : 0.334
#
# WHEN TO RECALIBRATE (run calibrate_threshold.py):
#   - You change EMBEDDING_MODEL (scores not portable across models)
#   - You load a new client knowledge base (different vocab shifts scores)
#   - Users report too many false refusals on valid questions  -> lower it
#   - Weak-match answers are slipping through               -> raise it
#
# HOW TO RECALIBRATE:
#   python calibrate_threshold.py
#   Then update this line with the recommended value from the output.
MIN_RELEVANCE = 0.302   # calibrated: text-embedding-3-small | pos_floor=0.504 neg_ceil=0.165 gap=0.339 driver=60% heuristic

# ROADMAP: THREE-ZONE THRESHOLD ARCHITECTURE
# -------------------------------------------
# The current binary gate (answer vs refuse) is the right
# starting point: simple, transparent, and explainable.
#
# When real client usage reveals frustration with borderline
# refusals -- queries that are partially answered by the KB
# but score just below MIN_RELEVANCE -- consider upgrading
# to a three-zone approach:
#
#   score >= MIN_RELEVANCE              → answer normally
#   score >= MIN_RELEVANCE * 0.75       → answer with caveat
#   score <  MIN_RELEVANCE * 0.75       → refuse
#
# The middle zone would prepend to the answer:
#   "I found partial information on this in the knowledge
#    base. Here is what I have, but you may want to verify
#    with an authoritative source:"
#
# BUILD TRIGGER: implement this when a real client reports
# that specific valid questions are being refused. Do not
# build it speculatively -- the binary gate is defensible
# and the three-zone approach requires careful prompt
# engineering to avoid the caveat feeling like a hedged
# hallucination.
#
# DESIGN NOTE: the 0.75 multiplier is a starting point.
# Calibrate the medium-zone boundary the same way you
# calibrate MIN_RELEVANCE -- measure actual score
# distributions and set it empirically, not arbitrarily.

# ---------------------------------------------------------------
# RATE LIMITING / SAFETY
# ---------------------------------------------------------------
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30


def banner() -> str:
    if LIVE_MODE:
        return (f"[LIVE MODE] Using OpenAI API — chat: {CHAT_MODEL}, "
                f"embeddings: {EMBEDDING_MODEL}")
    return "[MOCK MODE] No OPENAI_API_KEY found — using offline canned responses"


def require_live_mode():
    """Call this at the start of any script that MUST have a real key."""
    if not LIVE_MODE:
        raise RuntimeError(
            "This script requires a real OPENAI_API_KEY.\n"
            "Set it with:  $env:OPENAI_API_KEY = \"sk-...\"   (PowerShell)\n"
            "          or:  export OPENAI_API_KEY=\"sk-...\"   (bash)"
        )
