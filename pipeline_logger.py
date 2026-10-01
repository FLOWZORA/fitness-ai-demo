"""
pipeline_logger.py -- Silent internal audit trail for all pipeline events.

WHY THIS EXISTS
---------------
In health and wellness, if a client ever claims the AI gave dangerous advice,
you need an indisputable audit trail to determine what actually happened:
  - Did the model hallucinate? (faithfulness failure)
  - Did the client add a poorly worded or contradictory document the night
    before? (knowledge base change -- logged with exact file hashes)
  - Was the relevance threshold misconfigured? (calibration event logged)
  - Was the query so off-topic it should have been refused? (retrieval logged)

This logger writes everything to a hidden internal .log file that:
  - Is NEVER shown to the client (completely silent to the user)
  - Is ONLY readable by you (the architect/operator)
  - Uses structured JSON Lines format (one event per line, easily parseable)
  - Rotates automatically when the file exceeds MAX_LOG_SIZE_MB
  - Never crashes the pipeline if logging fails (safe try/except everywhere)

WHAT IS LOGGED
--------------
  STARTUP       -- every pipeline startup with config snapshot
  KB_CHANGE     -- knowledge base file additions, edits, deletions (with hashes)
  CALIBRATION   -- every threshold recalibration with full score distribution
  QUERY         -- every user query with retrieved chunks and relevance scores
  ANSWER        -- every generated answer (truncated for storage efficiency)
  REFUSAL       -- every relevance-gate refusal (below MIN_RELEVANCE)
  ERROR         -- any exception caught in the pipeline
  AUDIT_SUMMARY -- weekly digest of query volume, refusal rate, avg relevance

LOG LOCATION
------------
  .pipeline_audit.log  (in the project root, hidden file on Unix)
  On Windows this is just a regular file named .pipeline_audit.log

USAGE
-----
  from pipeline_logger import log

  log.startup()
  log.kb_change(old_hash, new_hash, files_before, files_after)
  log.calibration(results_dict)
  log.query(user_query, chunks, relevance_score)
  log.answer(user_query, answer_text, sources)
  log.refusal(user_query, best_score, threshold)
  log.error(context, exception)

All methods are safe to call even if the log file is locked or disk is full --
they catch all exceptions silently so the pipeline never crashes due to logging.
"""
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler

import config

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
LOG_FILE = config.PROJECT_ROOT / ".pipeline_audit.log"
MAX_LOG_SIZE_MB = 10
BACKUP_COUNT = 5       # keeps .1 through .5 rotated files
ANSWER_MAX_CHARS = 500 # truncate long answers in the log


# ---------------------------------------------------------------
# Internal logger setup
# ---------------------------------------------------------------
def _build_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline_audit")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # already configured (module reloaded)

    handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    # Raw JSON Lines format -- one JSON object per line, no extra formatting
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    # Suppress propagation so these never appear on stdout
    logger.propagate = False
    return logger


_logger = _build_logger()


def _write(event_type: str, payload: dict):
    """Write one structured event to the audit log. Never raises."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "client": config.CLIENT_NAME,
            "model": config.EMBEDDING_MODEL,
            **payload,
        }
        _logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        pass  # logging must never crash the pipeline


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
class PipelineLogger:
    """Structured audit logger. All methods are silent and exception-safe."""

    def startup(self):
        """Log every pipeline startup with a config snapshot."""
        _write("STARTUP", {
            "live_mode": config.LIVE_MODE,
            "chat_model": config.CHAT_MODEL if hasattr(config, "CHAT_MODEL")
                          else "gpt-4o-mini",
            "embedding_model": config.EMBEDDING_MODEL,
            "min_relevance": config.MIN_RELEVANCE,
            "top_k": config.TOP_K,
            "chunk_size": config.CHUNK_SIZE,
            "kb_files": [
                {"name": f.name, "bytes": f.stat().st_size}
                for f in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt"))
                if f.exists()
            ],
        })

    def kb_change(self, old_hash: str | None, new_hash: str,
                  files_before: list[str], files_after: list[str]):
        """Log a knowledge base change event with exact file hashes."""
        added = [f for f in files_after if f not in files_before]
        removed = [f for f in files_before if f not in files_after]
        modified = [f for f in files_after
                    if f in files_before and f not in added]

        _write("KB_CHANGE", {
            "old_hash": old_hash or "none",
            "new_hash": new_hash,
            "files_before": files_before,
            "files_after": files_after,
            "added": added,
            "removed": removed,
            "modified": modified,
            "note": ("This event precedes a vector store rebuild and "
                     "threshold recalibration. If the client later claims "
                     "the AI changed behaviour, cross-reference this timestamp."),
        })

    def calibration(self, results: dict):
        """Log a full threshold calibration result."""
        _write("CALIBRATION", {
            "min_positive": results.get("min_positive"),
            "max_negative": results.get("max_negative"),
            "gap": results.get("gap"),
            "recommended": results.get("recommended"),
            "driver": results.get("driver"),
            "margin_below_positive": results.get("margin_below_positive"),
            "margin_above_negative": results.get("margin_above_negative"),
            "previous_min_relevance": config.MIN_RELEVANCE,
        })

    def query(self, user_query: str, chunks: list, best_score: float):
        """Log a retrieval event with the top chunks and their scores."""
        _write("QUERY", {
            "query": user_query,
            "best_relevance": best_score,
            "threshold": config.MIN_RELEVANCE,
            "above_threshold": best_score >= config.MIN_RELEVANCE,
            "chunks_retrieved": [
                {
                    "source": c.get("source"),
                    "relevance": c.get("relevance"),
                    "preview": c.get("content", "")[:120],
                }
                for c in chunks[:4]
            ],
        })

    def answer(self, user_query: str, answer_text: str, sources: list[str]):
        """Log a generated answer (truncated). Sources are logged in full."""
        _write("ANSWER", {
            "query": user_query,
            "sources": sources,
            "answer_preview": answer_text[:ANSWER_MAX_CHARS],
            "answer_truncated": len(answer_text) > ANSWER_MAX_CHARS,
        })

    def refusal(self, user_query: str, best_score: float, threshold: float):
        """
        Log a relevance-gate refusal. This is the most forensically important
        event -- if a client claims the AI refused to answer a valid question,
        this log shows exactly what score it received vs the threshold.
        """
        _write("REFUSAL", {
            "query": user_query,
            "best_relevance": best_score,
            "threshold": threshold,
            "gap_to_threshold": round(threshold - best_score, 4),
            "note": ("Query scored below MIN_RELEVANCE. System correctly "
                     "refused rather than hallucinating a weak-match answer."),
        })

    def error(self, context: str, exc: Exception):
        """Log any caught exception with full traceback."""
        _write("ERROR", {
            "context": context,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })


# Singleton -- import and use anywhere in the pipeline
log = PipelineLogger()


# ---------------------------------------------------------------
# Log reader utility (for you, not the client)
# ---------------------------------------------------------------
def tail_log(n: int = 20, event_type: str | None = None):
    """
    Print the last N log entries, optionally filtered by event type.
    Usage:  python -c "from pipeline_logger import tail_log; tail_log(20)"
    Or:     python -c "from pipeline_logger import tail_log; tail_log(10, 'REFUSAL')"
    """
    if not LOG_FILE.exists():
        print("No audit log found yet.")
        return

    lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
    events = []
    for line in lines:
        try:
            e = json.loads(line)
            if event_type is None or e.get("event") == event_type:
                events.append(e)
        except json.JSONDecodeError:
            continue

    recent = events[-n:]
    print(f"=== Last {len(recent)} events"
          + (f" [{event_type}]" if event_type else "")
          + f" from {LOG_FILE.name} ===\n")
    for e in recent:
        ts = e.get("ts", "?")[:19].replace("T", " ")
        ev = e.get("event", "?")
        client = e.get("client", "?")
        detail = ""
        if ev == "QUERY":
            detail = (f"score={e.get('best_relevance')} "
                      f"q={e.get('query', '')[:50]}")
        elif ev == "REFUSAL":
            detail = (f"score={e.get('best_relevance')} "
                      f"threshold={e.get('threshold')} "
                      f"q={e.get('query', '')[:40]}")
        elif ev == "CALIBRATION":
            detail = (f"recommended={e.get('recommended')} "
                      f"gap={e.get('gap')} driver={e.get('driver')}")
        elif ev == "KB_CHANGE":
            detail = (f"added={e.get('added')} "
                      f"removed={e.get('removed')}")
        elif ev == "ANSWER":
            detail = (f"sources={e.get('sources', [])} "
                      f"q={e.get('query', '')[:40]}")
        elif ev == "STARTUP":
            detail = f"min_relevance={e.get('min_relevance')} live={e.get('live_mode')}"
        print(f"  {ts}  [{ev:12}]  {client}  {detail}")


if __name__ == "__main__":
    tail_log(30)
