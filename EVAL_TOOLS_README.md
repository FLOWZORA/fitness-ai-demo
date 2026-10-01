# Evaluation Tools — Upgraded Audit Layer

Three industry-standard evaluation tools now layered onto your pipeline.
Each serves a distinct purpose and runs independently.

## What each tool does and when to use it

### 1. DeepEval  →  audit_deepeval.py
**Upgrades your audit from string-matching to LLM-judged scoring.**

Your original audit.py checks if literal text like "deadlift" appears in a
response. That misses failures when phrasing varies ("Romanian deadlift",
"stiff-leg DL"). DeepEval's GEval uses an LLM judge to score by MEANING:
"does this recommend a contraindicated exercise?" — caught regardless of wording.

Run:
```
$env:OPENAI_API_KEY = "sk-..."
python audit_deepeval.py
```
Scores both broken and optimized systems on: safety screening, dangerous-advice
avoidance, and factual accuracy. Costs a few cents (LLM judge calls).

**Use for:** every real client audit. This is your credibility upgrade.

---

### 2. Promptfoo  →  promptfooconfig.yaml
**A repeatable, shareable test suite with a web-UI report.**

Runs entirely as a Node tool driven by YAML — completely separate from your
Python code, so it can never break your pipeline. Uses llm-rubric assertions
(LLM-graded) for robust, phrasing-independent checks. Produces a clean web
report you can screen-record or screenshot for a client deliverable.

Setup + run:
```
npm install -g promptfoo
$env:OPENAI_API_KEY = "sk-..."
promptfoo eval -c promptfooconfig.yaml
promptfoo view      # opens the results web UI
```

**Use for:** producing a polished, professional audit report artifact, and for
running larger systematic test suites without writing Python.

---

### 3. Ragas  →  ragas_eval.py   (STANDALONE — run in isolation)
**Scores RAG answer quality, not just retrieval distance.**

Your pipeline's MIN_RELEVANCE threshold scores retrieval distance only. Ragas
scores the GENERATED ANSWER with the industry-standard RAG triad:
- faithfulness:     does the answer stay grounded in retrieved context?
- answer_relevancy: does it address the question?
- context_recall:   did retrieval find the needed info?

This catches a failure your current pipeline can't: retrieval was correct, but
the LLM still drifted or added unsupported claims.

⚠️ Run standalone — Ragas pins older langchain versions. Don't wire it into
run_pipeline.py.
```
python ragas_eval.py
```

**Use for:** validating a real client's RAG build before delivery, and as
proof-of-quality in your RAG project deliverables.

---

## Dependency note (important)

DeepEval and Promptfoo are clean — no conflicts.
Ragas pins older langchain versions. The tested-compatible set is in
requirements-eval.txt. If you want maximum safety, run ragas_eval.py in a
separate Python virtual environment so its pins never touch your main pipeline:

```
python -m venv ragas-env
ragas-env\Scripts\activate     # Windows
pip install ragas==0.2.10 langchain-community==0.3.7 datasets
python ragas_eval.py
deactivate
```

## The honest framing for clients

"I run a multi-layered evaluation: DeepEval for LLM-judged safety and accuracy
scoring, Promptfoo for repeatable regression testing, and Ragas for RAG answer
faithfulness. This is the same evaluation stack used by production AI teams."

That is true, and it's a materially stronger pitch than hand-rolled checks.
