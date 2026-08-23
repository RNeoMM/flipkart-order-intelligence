from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT_DIR / "part3_support_agent" / "policies" / "policy_documents.json"
INDEX_DIR = ROOT_DIR / "part3_support_agent" / "policies"
INDEX_PATH = INDEX_DIR / "policy_index.faiss"
CHUNKS_PATH = INDEX_DIR / "policy_chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["documents"]


def split_sentences(text):
    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def build_chunks(documents):
    chunks = []

    for document in documents:
        sentences = split_sentences(
            document["text"]
        )

        for chunk_number, sentence in enumerate(
            sentences,
            start=1,
        ):
            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}_"
                        f"C{chunk_number}"
                    ),
                    "document_id": document[
                        "document_id"
                    ],
                    "title": document[
                        "title"
                    ],
                    "text": sentence,
                }
            )

    return chunks


def build_index(chunks):
    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return index, embeddings


def main():
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading policy documents..."
    )

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    chunks = build_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} sentence-level chunks."
    )

    index, embeddings = build_index(
        chunks
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal}"
    )

    print(
        f"Saved index to: "
        f"{INDEX_PATH}"
    )

    print(
        f"Saved chunk metadata to: "
        f"{CHUNKS_PATH}"
    )

    print(
        "\nFirst five chunks:"
    )

    for chunk in chunks[:5]:
        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['document_id']} | "
            f"{chunk['text']}"
        )


if __name__ == "__main__":
    main()