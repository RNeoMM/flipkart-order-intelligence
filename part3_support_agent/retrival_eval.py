from pathlib import Path
import json

from .retrieval import retrieve_policy_chunks


ROOT_DIR = Path(__file__).resolve().parent.parent

POLICY_PATH = (
    ROOT_DIR
    / "part3_support_agent"
    / "policies"
    / "policy_documents.json"
)


def load_evaluation_queries():
    with open(
        POLICY_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["retrieval_evaluation"]


def unique_document_ids(results):
    document_ids = []

    for result in results:
        document_id = result[
            "document_id"
        ]

        if document_id not in document_ids:
            document_ids.append(
                document_id
            )

    return document_ids


def precision_at_3(
    retrieved_documents,
    relevant_documents,
):
    top_three = retrieved_documents[:3]

    relevant_count = sum(
        document_id in relevant_documents
        for document_id in top_three
    )

    return relevant_count / 3


def recall_at_3(
    retrieved_documents,
    relevant_documents,
):
    top_three = retrieved_documents[:3]

    relevant_count = sum(
        document_id in relevant_documents
        for document_id in top_three
    )

    if not relevant_documents:
        return 0.0

    return (
        relevant_count
        / len(relevant_documents)
    )


def main():
    evaluation_queries = (
        load_evaluation_queries()
    )

    precision_values = []
    recall_values = []

    print(
        "\nRetrieval Evaluation"
    )

    print(
        "=" * 70
    )

    for item in evaluation_queries:
        query_id = item[
            "query_id"
        ]

        query = item[
            "query"
        ]

        relevant_documents = item[
            "relevant_documents"
        ]

        results = retrieve_policy_chunks(
            query,
            top_k=3,
        )

        retrieved_documents = (
            unique_document_ids(
                results
            )
        )

        precision = precision_at_3(
            retrieved_documents,
            relevant_documents,
        )

        recall = recall_at_3(
            retrieved_documents,
            relevant_documents,
        )

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        relevant_hits = [
            document_id
            for document_id in retrieved_documents[:3]
            if document_id
            in relevant_documents
        ]

        print(
            f"\n{query_id}: {query}"
        )

        print(
            f"Relevant documents: "
            f"{relevant_documents}"
        )

        print(
            f"Retrieved documents: "
            f"{retrieved_documents}"
        )

        print(
            f"Relevant hits in top-3: "
            f"{relevant_hits}"
        )

        print(
            f"Precision@3 = "
            f"{len(relevant_hits)}/3 "
            f"= {precision:.4f}"
        )

        print(
            f"Recall@3 = "
            f"{len(relevant_hits)}/"
            f"{len(relevant_documents)} "
            f"= {recall:.4f}"
        )

    average_precision = (
        sum(precision_values)
        / len(precision_values)
    )

    average_recall = (
        sum(recall_values)
        / len(recall_values)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Average Precision@3: "
        f"{average_precision:.4f}"
    )

    print(
        "Average Recall@3: "
        f"{average_recall:.4f}"
    )


if __name__ == "__main__":
    main()