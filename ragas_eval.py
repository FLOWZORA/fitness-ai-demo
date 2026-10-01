"""
ragas_eval.py - STANDALONE RAG quality evaluation using Ragas.

⚠️ IMPORTANT: Run this in ISOLATION, not as part of run_pipeline.py.
Ragas pulls in heavy langchain dependencies that can conflict with the
versions your main RAG pipeline uses. Keeping it standalone protects your
working pipeline. Run it on its own:

    python ragas_eval.py

What it measures (the RAG triad - the industry-standard RAG metrics):
  - faithfulness:     Does the generated answer stay grounded in the retrieved
                      context, or does it hallucinate beyond it?
  - answer_relevancy: Does the answer actually address the question asked?
  - context_recall:   Did retrieval surface the information needed to answer?

Why this adds value beyond your existing relevance threshold:
  Your pipeline's MIN_RELEVANCE check scores RETRIEVAL distance only. Ragas
  also scores the GENERATED ANSWER - catching the case where retrieval was
  correct but the LLM still drifted or added unsupported claims. That's a
  different, important failure mode your current pipeline can't detect.

Requires a real OPENAI_API_KEY (Ragas uses an LLM judge + embeddings).
"""
import os
import config

if not config.LIVE_MODE:
    print("Ragas requires a real OPENAI_API_KEY. Set it and re-run.")
    raise SystemExit(1)

os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)

from datasets import Dataset
from openai import OpenAI as _OpenAIClient
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

# We import the RAG pipeline lazily and ONLY to generate eval data.
from rag_pipeline import build_vector_store, retrieve, answer

# ---------------------------------------------------------------------
# answer_relevancy needs an embeddings object. The OLD path (langchain's
# OpenAIEmbeddings via langchain-openai) is broken on langchain-openai>=1.0 -
# embed_query() was removed in that refactor. Ragas ships its own modern
# wrapper that talks to the OpenAI client directly, bypassing langchain's
# embeddings layer entirely. This is the fix.
# ---------------------------------------------------------------------
_ragas_embeddings = RagasOpenAIEmbeddings(
    client=_OpenAIClient(api_key=config.OPENAI_API_KEY),
    model=config.EMBEDDING_MODEL,
)


# Test questions with reference answers (ground truth for context_recall).
#
# Organized in three tiers:
#   BASELINE  - should always score high; if these fail, something is broken
#   REGRESSION- the exact questions that exposed the cosine-distance bug
#               during deployment debugging. Keep these permanently - they
#               guard against that bug (or similar ones) coming back.
#   EDGE CASE - harder questions that test retrieval precision: specific
#               numbers, safety-adjacent content, and topics covered in
#               more than one document (risk of retrieving the wrong one).
EVAL_SET = [
    # ---- BASELINE ----
    {
        "question": "How much creatine should I take daily?",
        "ground_truth": "Creatine monohydrate is recommended at 3-5g per day, "
                        "with no loading phase required.",
    },
    {
        "question": "How much protein should I eat to build muscle?",
        "ground_truth": "For muscle building, aim for 0.7-1.0g of protein per "
                        "pound of bodyweight per day.",
    },
    {
        "question": "How many days a week should a beginner train?",
        "ground_truth": "Beginners should train 2-3 full-body sessions per week.",
    },
    {
        "question": "What is the recommended sleep for recovery?",
        "ground_truth": "Adults should target 7-9 hours of sleep per night for "
                        "optimal recovery.",
    },

    # ---- REGRESSION (exposed the cosine-distance bug in deployment) ----
    {
        "question": "I have a back injury. Can I still train?",
        "ground_truth": "If any injury is reported, do not issue a program "
                        "without medical clearance. Exercises contraindicated "
                        "for back injuries include heavy barbell deadlifts, "
                        "good mornings, heavy bent-over rows, and loaded "
                        "spinal flexion/extension.",
    },
    {
        "question": "What training frequency is good for a beginner?",
        "ground_truth": "Beginners (0-12 months training) should do 2-3 "
                        "full-body sessions per week.",
    },
    {
        "question": "How much protein is in a chicken breast?",
        "ground_truth": "A standard 6oz (170g) cooked chicken breast provides "
                        "approximately 53g of protein; per 100g cooked it is "
                        "31g of protein.",
    },

    # ---- EDGE CASE: specific numbers, easy to get subtly wrong ----
    {
        "question": "How often should I take a deload week?",
        "ground_truth": "Every 6-8 weeks of progressive training, reduce "
                        "volume by 40-50% for one week to promote recovery "
                        "and prevent overreaching.",
    },
    {
        "question": "What vitamin D dosage is safe for supplementation?",
        "ground_truth": "1,000-4,000 IU/day is a typical supplementation "
                        "range. Do not recommend 100,000 IU - that is a "
                        "toxic dose since fat-soluble vitamins accumulate.",
    },
    {
        "question": "Is it okay to skip the warm-up if I'm short on time?",
        "ground_truth": "No. Skipping warm-up is never recommended and "
                        "increases injury risk significantly. The protocol "
                        "includes 5-10 minutes of general cardiovascular "
                        "warm-up, dynamic stretching, and 2 ramp-up sets "
                        "before the first working set.",
    },
    {
        "question": "What is Zone 2 cardio and why does it matter?",
        "ground_truth": "Zone 2 is 50-70% max heart rate, conversational "
                        "pace. It is the most evidenced modality for "
                        "longevity, metabolic health, and aerobic base "
                        "development. Target 150+ minutes per week.",
    },
    {
        "question": "What does a high VO2 max indicate?",
        "ground_truth": "VO2 max is the single strongest predictor of "
                        "all-cause mortality. Target the top 25th percentile "
                        "for age/sex minimum. It improves with Zone 2 and "
                        "HIIT training.",
    },
    {
        "question": "What supplement helps with sleep quality?",
        "ground_truth": "Magnesium glycinate, 200-400mg before bed, supports "
                        "sleep quality.",
    },

    # ---- EDGE CASE: topic overlaps two documents (tests retrieval precision) ----
    {
        "question": "Does creatine help with recovery, not just muscle building?",
        "ground_truth": "Yes - creatine also supports recovery from "
                        "high-intensity sessions at the same 3-5g/day dose "
                        "used for strength and hypertrophy. This is "
                        "mentioned in both the nutrition protocol and the "
                        "recovery/longevity guidance.",
    },
    {
        "question": "Is high protein safe for someone with kidney disease?",
        "ground_truth": "No - high protein diets may be contraindicated for "
                        "kidney disease. Medical clearance is required first.",
    },
]


def build_eval_dataset():
    """Run each question through the real RAG pipeline and collect the data
    Ragas needs. NOTE: Ragas 0.4.x renamed its expected columns from the
    older question/answer/contexts/ground_truth naming to:
      user_input, response, retrieved_contexts, reference
    This builds the dataset using the CURRENT (0.4.x) column names."""
    store = build_vector_store()

    user_inputs, responses, contexts_list, references = [], [], [], []

    for item in EVAL_SET:
        q = item["question"]
        chunks = retrieve(store, q)
        result = answer(store, q)

        user_inputs.append(q)
        responses.append(result["answer"])
        contexts_list.append([c["content"] for c in chunks])
        references.append(item["ground_truth"])

    return Dataset.from_dict({
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": contexts_list,
        "reference": references,
    })


if __name__ == "__main__":
    print(config.banner())
    print(f"\nBuilding RAG evaluation dataset ({len(EVAL_SET)} questions: "
          f"baseline, regression, and edge cases)...")
    dataset = build_eval_dataset()

    print("\nScoring with Ragas (faithfulness, answer relevancy, context "
          "recall)...")
    print("This makes several LLM judge calls - costs a few cents.\n")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        embeddings=_ragas_embeddings,
    )

    print("\n" + "=" * 60)
    print("  RAGAS RAG QUALITY SCORES - OVERALL")
    print("=" * 60)
    print(results)

    # Per-question breakdown - this is what actually finds weak spots
    try:
        df = results.to_pandas()
        print("\n" + "=" * 60)
        print("  PER-QUESTION BREAKDOWN")
        print("=" * 60)
        cols = [c for c in ["user_input", "faithfulness",
                            "answer_relevancy", "context_recall"]
               if c in df.columns]
        for _, row in df.iterrows():
            q = row.get("user_input", "?")
            print(f"\nQ: {q}")
            for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
                if metric in df.columns:
                    score = row[metric]
                    flag = "  [LOW]" if (score is not None and score < 0.7) else ""
                    print(f"  {metric:<18}: {score}{flag}")
    except Exception as e:
        print(f"\n(Per-question breakdown unavailable: {e})")
        print("Overall scores above are still valid.")

    print("\nInterpretation:")
    print("  faithfulness    - higher = answer stays grounded in sources")
    print("  answer_relevancy- higher = answer addresses the question")
    print("  context_recall  - higher = retrieval found the needed info")
    print("  [LOW] flags any score below 0.7 on a specific question - "
          "investigate those first.")
    print("\nThese three scores are the industry-standard RAG quality triad.")
