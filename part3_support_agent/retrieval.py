from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = (
    ROOT_DIR
    / "part3_support_agent"
    / "policies"
)

INDEX_PATH = (
    POLICY_DIR
    / "policy_index.faiss"
)

CHUNKS_PATH = (
    POLICY_DIR
    / "policy_chunks.json"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.45

_INDEX = None
_CHUNKS = None
_EMBEDDER = None


def load_index():
    global _INDEX

    if _INDEX is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}"
            )

        _INDEX = faiss.read_index(
            str(INDEX_PATH)
        )

    return _INDEX


def load_chunks():
    global _CHUNKS

    if _CHUNKS is None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {CHUNKS_PATH}"
            )

        with open(
            CHUNKS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            _CHUNKS = json.load(file)

    return _CHUNKS


def load_embedder():
    global _EMBEDDER

    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _EMBEDDER


def retrieve_policy_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):
    query = query.strip()

    if not query:
        return []

    index = load_index()
    chunks = load_chunks()
    embedder = load_embedder()

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):
        if index_id < 0:
            continue

        chunk = chunks[index_id]

        results.append(
            {
                "chunk_id": chunk[
                    "chunk_id"
                ],
                "document_id": chunk[
                    "document_id"
                ],
                "title": chunk[
                    "title"
                ],
                "text": chunk[
                    "text"
                ],
                "similarity": float(
                    score
                ),
            }
        )

    return results


def retrieve_policy_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):
    chunk_results = retrieve_policy_chunks(
        query=query,
        top_k=top_k,
    )

    documents = []
    seen = set()

    for result in chunk_results:
        document_id = result[
            "document_id"
        ]

        if document_id in seen:
            continue

        seen.add(document_id)

        documents.append(
            {
                "document_id": document_id,
                "title": result["title"],
                "best_similarity": result[
                    "similarity"
                ],
                "supporting_chunks": [
                    {
                        "chunk_id": result[
                            "chunk_id"
                        ],
                        "text": result[
                            "text"
                        ],
                        "similarity": result[
                            "similarity"
                        ],
                    }
                ],
            }
        )

    return documents


def is_grounded(
    results,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
):
    if not results:
        return False

    best_similarity = max(
        result["similarity"]
        for result in results
    )

    return best_similarity >= threshold


def best_similarity(results):
    if not results:
        return 0.0

    return max(
        result["similarity"]
        for result in results
    )


def main():
    test_queries = [
        "What is the return window for shoes?",
        "How long does a COD refund take?",
        "Can I get a reverse pickup?",
        "What happens if my delivery is late?",
        "What is the return policy for electronics?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        results = retrieve_policy_chunks(
            query,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{result['document_id']} | "
                f"{result['title']} | "
                f"similarity="
                f"{result['similarity']:.4f}"
            )

            print(
                f"   {result['text']}"
            )

        print(
            f"\nGrounded with threshold "
            f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}: "
            f"{is_grounded(results)}"
        )

        print(
            f"Best similarity: "
            f"{best_similarity(results):.4f}"
        )


if __name__ == "__main__":
    main()