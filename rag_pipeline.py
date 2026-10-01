"""
rag_pipeline.py — The RAG chatbot build (API / LIVE version).

This is the UPSELL you offer after the audit reveals the client's AI is
answering from general training data instead of their own content.

LIVE MODE (OPENAI_API_KEY set):
  - Real OpenAI text-embedding-3-small embeddings
  - ChromaDB persistent vector store (proper cosine similarity search)
  - Real GPT-4o-mini grounded answer generation
  - Minimum relevance threshold — refuses to answer on weak matches instead
    of returning an irrelevant chunk (this fixes the bug you just saw with
    the offline TF-IDF version)

MOCK MODE (no key):
  - Falls back to the offline TF-IDF retriever for free practice

Architecture:
  1. Load the client's knowledge base documents (txt files)
  2. Chunk them into overlapping segments
  3. Embed each chunk (real embeddings in live mode)
  4. Store in ChromaDB, persisted to disk
  5. At query time: embed the question, retrieve top-K by cosine similarity,
     reject weak matches below MIN_RELEVANCE
  6. Pass retrieved context + question to LLM -> grounded, cited answer
"""
import re
import math
import pickle
from pathlib import Path
from collections import defaultdict

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from llm_client import mode_banner  # mock fallback still available

TOP_K = config.TOP_K
MIN_RELEVANCE = config.MIN_RELEVANCE


# ============================================================
# LIVE MODE: ChromaDB + real OpenAI embeddings
# ============================================================

def _build_chroma_store(force_rebuild: bool = False):
    import chromadb
    from chromadb.utils import embedding_functions
    import api_client

    config.CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    collection_name = "knowledge_base"
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing and not force_rebuild:
        print("Loading existing ChromaDB collection...")
        return client.get_collection(collection_name)

    if collection_name in existing:
        client.delete_collection(collection_name)

    print("Building ChromaDB collection with real OpenAI embeddings...")
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},  # MUST be set explicitly -- default
    )                                        # is L2, which breaks relevance scores

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks, all_sources, all_ids = [], [], []
    for file in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for files saved with non-UTF-8 encoding (e.g. UTF-16
            # from PowerShell's echo command, or Latin-1 from older editors).
            # errors="replace" substitutes undecodable bytes with ? rather
            # than crashing -- the content is still usable for embedding.
            text = file.read_text(encoding="utf-8-sig", errors="replace")
            print(f"  [WARN] {file.name}: non-UTF-8 encoding detected, "
                  f"read with fallback. Re-save as UTF-8 for best results.")
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            all_chunks.append(piece)
            all_sources.append(file.name)
            all_ids.append(f"{file.stem}_{i}")
        print(f"  Loaded: {file.name} -> {len(pieces)} chunks")

    # Embed in one batch call (real API, real cost — tracked by api_client)
    print(f"  Embedding {len(all_chunks)} chunks via {config.EMBEDDING_MODEL}...")
    vectors = api_client.embed(all_chunks)

    collection.add(
        ids=all_ids,
        embeddings=vectors,
        documents=all_chunks,
        metadatas=[{"source": s} for s in all_sources],
    )
    print(f"  Stored {len(all_chunks)} chunks in ChromaDB at {config.CHROMA_DIR}")
    return collection


def _retrieve_chroma(collection, query: str, k: int) -> list:
    import api_client
    query_vec = api_client.embed([query])[0]
    results = collection.query(query_embeddings=[query_vec], n_results=k)

    chunks = []
    for doc, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        # ChromaDB returns distance (lower = closer); convert to similarity-like score
        relevance = round(max(0.0, 1.0 - dist), 3)
        chunks.append({"content": doc, "source": meta["source"],
                       "relevance": relevance})
    return chunks


# ============================================================
# MOCK MODE: offline TF-IDF retriever (free, no API key needed)
# ============================================================

def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class LocalVectorStore:
    """TF-IDF retriever for offline practice. No neural net, no API."""

    def __init__(self):
        self.chunks = []
        self.tfidf = []
        self.idf = {}

    def add_documents(self, docs: list):
        self.chunks = docs
        df = defaultdict(int)
        tokenized = []
        for doc in docs:
            tokens = _tokenize(doc["content"])
            tokenized.append(tokens)
            for t in set(tokens):
                df[t] += 1
        N = max(len(docs), 1)
        self.idf = {t: math.log(N / (1 + f)) for t, f in df.items()}
        self.tfidf = []
        for tokens in tokenized:
            tf = defaultdict(float)
            for t in tokens:
                tf[t] += 1.0 / max(len(tokens), 1)
            self.tfidf.append({t: tf[t] * self.idf.get(t, 0) for t in tf})

    def _cosine(self, a: dict, b: dict) -> float:
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in b)
        mag_a = math.sqrt(sum(v ** 2 for v in a.values())) or 1e-9
        mag_b = math.sqrt(sum(v ** 2 for v in b.values())) or 1e-9
        return dot / (mag_a * mag_b)

    def similarity_search(self, query: str, k: int = TOP_K) -> list:
        tokens = _tokenize(query)
        tf = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0 / max(len(tokens), 1)
        qvec = {t: tf[t] * self.idf.get(t, 0) for t in tf}
        scores = [(self._cosine(qvec, dv), i)
                  for i, dv in enumerate(self.tfidf)]
        scores.sort(reverse=True)
        return [
            {"content": self.chunks[i]["content"], "source": self.chunks[i]["source"],
             "relevance": round(score, 3)}
            for score, i in scores[:k]
        ]

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _build_local_store(force_rebuild: bool = False) -> LocalVectorStore:
    if config.LOCAL_STORE_PATH.exists() and not force_rebuild:
        print("Loading existing local vector store...")
        return LocalVectorStore.load(config.LOCAL_STORE_PATH)

    print("Building offline TF-IDF vector store (mock mode)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = []
    for file in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file.read_text(encoding="utf-8-sig", errors="replace")
            print(f"  [WARN] {file.name}: non-UTF-8 encoding, read with fallback.")
        for piece in splitter.split_text(text):
            raw_chunks.append({"content": piece, "source": file.name})
        print(f"  Loaded: {file.name}")

    store = LocalVectorStore()
    store.add_documents(raw_chunks)
    store.save(config.LOCAL_STORE_PATH)
    print(f"  Built store: {len(raw_chunks)} chunks")
    return store


# ============================================================
# PUBLIC API — same interface regardless of mode
# ============================================================

def build_vector_store(force_rebuild: bool = False):
    """Returns a ChromaDB collection (live) or LocalVectorStore (mock)."""
    if config.LIVE_MODE:
        return _build_chroma_store(force_rebuild)
    return _build_local_store(force_rebuild)


def retrieve(store, query: str, k: int = TOP_K) -> list:
    if config.LIVE_MODE:
        return _retrieve_chroma(store, query, k)
    return store.similarity_search(query, k=k)


RAG_SYSTEM_PROMPT = """You are the FitCoach Pro AI assistant. You answer ONLY from
the provided context passages, which come from FitCoach Pro's verified coaching
documents. Do not use general knowledge — if the context doesn't contain the
answer, say so clearly.

Rules:
- Always cite the source document name after specific facts.
- Apply safety screening: if injury, pregnancy, or disordered eating is mentioned,
  flag it and refer to a professional instead of giving a plan.
- Never invent numbers. Use only figures that appear in the context.
- Keep answers concise and cite sources."""


def _mock_rag_response(query: str, chunks: list) -> str:
    if not chunks:
        return "This isn't covered in FitCoach Pro's current knowledge base."
    top = chunks[0]["content"][:400]
    source = chunks[0]["source"].replace(".txt", "").replace("_", " ").title()
    return (f"Based on FitCoach Pro's {source}:\n\n{top}\n\n"
            f"[Source: {source} | Relevance: {chunks[0]['relevance']}]")


def _real_rag_response(query: str, chunks: list) -> str:
    import api_client
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )
    return api_client.chat(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=f"Context passages:\n{context}\n\nQuestion: {query}",
        temperature=0.2,
    )


def answer(store, query: str) -> dict:
    """Full RAG pipeline: retrieve -> threshold check -> generate -> cite."""
    from pipeline_logger import log
    chunks = retrieve(store, query, k=TOP_K)
    best_score = chunks[0]["relevance"] if chunks else 0.0

    # Log every query with its retrieval scores
    log.query(query, chunks, best_score)

    # Minimum relevance gate — this is the fix for the bad-match bug.
    # If even the BEST chunk is below threshold, don't fabricate an answer.
    if not chunks or best_score < MIN_RELEVANCE:
        # Log the refusal — forensically important for liability
        log.refusal(query, best_score, MIN_RELEVANCE)
        return {
            "query": query,
            "answer": ("I don't have a confident answer for this in FitCoach "
                      "Pro's current knowledge base. This may need to be added "
                      "to the content library."),
            "sources": [],
            "chunks_used": 0,
            "below_threshold": True,
            "top_relevance": best_score,
        }

    response = (_real_rag_response(query, chunks) if config.LIVE_MODE
               else _mock_rag_response(query, chunks))

    sources = list(dict.fromkeys(c["source"] for c in chunks))
    log.answer(query, response, sources)

    return {
        "query": query,
        "answer": response,
        "sources": [c["source"] for c in chunks],
        "chunks_used": len(chunks),
        "below_threshold": False,
        "top_relevance": chunks[0]["relevance"],
    }


def compare_generic_vs_rag(store):
    from broken_system import ask_broken
    queries = [
        "How much protein is in a chicken breast?",
        "How much creatine should I take?",
        "I have a back injury. Can I still train?",
        "What training frequency for a beginner?",
    ]
    print("\n" + "=" * 72)
    print("  GENERIC AI  vs  RAG (grounded in FitCoach Pro content)")
    print("=" * 72)
    for q in queries:
        generic = ask_broken(q)
        rag_result = answer(store, q)
        print(f"\nQ: {q}")
        print(f"\n  GENERIC:  {generic[:180]}...")
        print(f"\n  RAG ({'relevance ' + str(rag_result['top_relevance'])}):"
              f"  {rag_result['answer'][:280]}...")
        if rag_result["sources"]:
            print(f"  Sources:  {', '.join(set(rag_result['sources']))}")
        print("\n" + "-" * 72)


if __name__ == "__main__":
    print(config.banner())
    print("\n=== RAG CHATBOT BUILD ===\n")
    vs = build_vector_store()
    compare_generic_vs_rag(vs)
    if config.LIVE_MODE:
        import api_client
        api_client.print_usage_report()
    print("\nRAG system ready. Run: python rag_demo.py")
