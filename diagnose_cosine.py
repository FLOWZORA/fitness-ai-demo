"""
diagnose_cosine.py — Before/after comparison of L2 vs. cosine distance metrics.

Creates two temporary ChromaDB collections (l2 and cosine) from the same
knowledge-base chunks, embeds them with the same OpenAI model, then queries
both with the three test questions so you can see the raw distance and
converted relevance side by side.

Run:  python diagnose_cosine.py
Requires: OPENAI_API_KEY in environment (live mode only).
"""
import sys
import chromadb

import config
import api_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

config.require_live_mode()

TEST_QUERIES = [
    "How much creatine should I take?",
    "I have a back injury, can I still train?",
    "What training frequency for a beginner?",
]

DIAG_L2     = "diag_l2_tmp"
DIAG_COSINE = "diag_cosine_tmp"


def load_chunks() -> tuple[list[str], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts, ids = [], []
    for file in sorted(config.KNOWLEDGE_BASE_DIR.glob("*.txt")):
        for i, piece in enumerate(splitter.split_text(file.read_text(encoding="utf-8"))):
            texts.append(piece)
            ids.append(f"{file.stem}_{i}")
    return texts, ids


def build_collections(client: chromadb.PersistentClient, texts: list[str],
                      ids: list[str], vectors: list) -> tuple:
    for name in (DIAG_L2, DIAG_COSINE):
        if name in [c.name for c in client.list_collections()]:
            client.delete_collection(name)

    coll_l2 = client.create_collection(DIAG_L2)  # default = L2
    coll_cos = client.create_collection(DIAG_COSINE, metadata={"hnsw:space": "cosine"})

    coll_l2.add(ids=ids, embeddings=vectors, documents=texts)
    coll_cos.add(ids=ids, embeddings=vectors, documents=texts)
    return coll_l2, coll_cos


def query_and_print(label: str, collection, query_vec: list, query: str):
    results = collection.query(query_embeddings=[query_vec], n_results=1)
    dist = results["distances"][0][0]
    relevance = round(max(0.0, 1.0 - dist), 3)
    snippet = results["documents"][0][0][:80].replace("\n", " ")
    print(f"  [{label:6s}]  raw_dist={dist:.4f}  relevance={relevance:.3f}  "
          f"top_chunk: \"{snippet}...\"")


def cleanup(client: chromadb.PersistentClient):
    for name in (DIAG_L2, DIAG_COSINE):
        try:
            client.delete_collection(name)
        except Exception:
            pass


def main():
    print(f"\n{config.banner()}")
    print("\n=== DISTANCE METRIC DIAGNOSTIC (L2 vs. COSINE) ===\n")

    print("Loading knowledge-base chunks...")
    texts, ids = load_chunks()
    print(f"  {len(texts)} chunks loaded.\n")

    print(f"Embedding {len(texts)} chunks via {config.EMBEDDING_MODEL}...")
    vectors = api_client.embed(texts)
    print("  Done.\n")

    db_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll_l2, coll_cos = build_collections(db_client, texts, ids, vectors)
    print("Temporary collections built (L2 and cosine).\n")

    print("-" * 72)
    for query in TEST_QUERIES:
        print(f"Q: {query}")
        q_vec = api_client.embed([query])[0]
        query_and_print("BEFORE", coll_l2,  q_vec, query)
        query_and_print("AFTER",  coll_cos, q_vec, query)
        print()
    print("-" * 72)

    print("\nNotes:")
    print("  BEFORE (L2):     1.0 - euclidean_dist — wrong formula for L2 distance")
    print("  AFTER  (cosine): 1.0 - cosine_dist   — equals cosine similarity directly")
    print("\nWith cosine metric, scores near 1.0 = strong semantic match.")
    print("The rag_pipeline.py fix will force-rebuild the live collection on next run.\n")

    cleanup(db_client)
    api_client.print_usage_report()


if __name__ == "__main__":
    main()
