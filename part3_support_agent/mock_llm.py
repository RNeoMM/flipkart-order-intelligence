import json

from .prompt_config import format_system_prompt


def build_intent_examples():
    return [
        {
            "user": "What is the return window for footwear?",
            "intent": "policy",
        },
        {
            "user": "Is order ORD1001 likely to be returned?",
            "intent": "return_risk",
        },
        {
            "user": "What category is data/sample_images/07_sneaker.png?",
            "intent": "product_category",
        },
    ]


def classify_intent(text: str) -> str:
    normalized = text.lower().strip()

    policy_terms = [
        "return policy",
        "return window",
        "refund",
        "delivery",
        "delivery date",
        "reverse pickup",
        "pickup",
        "eligible",
        "non-returnable",
        "replacement",
        "cod",
        "cash on delivery",
        "what can i do about that order",
        "what should i do about that order",
        "what about that order",
        "that order",
    ]

    risk_terms = [
        "return risk",
        "likely to be returned",
        "probability of return",
        "risk of return",
        "will this order be returned",
        "return probability",
        "risk bucket",
    ]

    image_terms = [
        "product category",
        "classify image",
        "classify this image",
        "what product is this",
        "what category is this image",
        "what category is this product image",
        "what category is this product",
        "this product image",
        ".png",
        "sneaker image",
        "product photo",
    ]

    if any(
        term in normalized
        for term in risk_terms
    ):
        return "return_risk"

    if any(
        term in normalized
        for term in image_terms
    ):
        return "product_category"

    if any(
        term in normalized
        for term in policy_terms
    ):
        return "policy"

    return "policy"


def generate_response(
    intent: str,
    retrieved_chunks=None,
    return_risk_result=None,
    image_result=None,
    grounded=True,
    state=None,
):
    if retrieved_chunks is None:
        retrieved_chunks = []

    if state is None:
        state = {}

    user_message = state.get(
        "user_message",
        "",
    )

    normalized_message = (
        user_message.lower().strip()
    )

    order_id = state.get(
        "order_id"
    )

    if (
        intent == "policy"
        and not order_id
        and (
            "that order" in normalized_message
            or "this order" in normalized_message
            or "what about that order" in normalized_message
            or "what can i do about that order"
            in normalized_message
            or "what should i do about that order"
            in normalized_message
        )
    ):
        return {
            "answer": (
                "I don't have an order ID in this "
                "conversation. Please provide the order ID "
                "so I can help with that specific order."
            ),
            "source": "policy_kb",
            "confidence": 0.0,
        }

    if intent == "policy":
        if not grounded:
            return {
                "answer": (
                    "I don't have sufficiently similar "
                    "policy information in the knowledge base "
                    "to answer that safely."
                ),
                "source": "policy_kb",
                "confidence": 0.0,
            }

        if not retrieved_chunks:
            return {
                "answer": (
                    "I could not find supporting policy "
                    "information in the knowledge base."
                ),
                "source": "policy_kb",
                "confidence": 0.0,
            }

        primary = retrieved_chunks[0]

        supporting_text = primary[
            "text"
        ]

        if order_id:
            answer = (
                f"For order {order_id}: "
                f"{supporting_text} "
                "Please check the specific order page "
                "for the latest order-specific update."
            )
        else:
            answer = (
                f"{supporting_text} "
                "Please check the specific order page "
                "for any item-specific eligibility shown there."
            )

        confidence = min(
            max(
                float(
                    primary["similarity"]
                ),
                0.0,
            ),
            1.0,
        )

        return {
            "answer": answer,
            "source": "policy_kb",
            "confidence": round(
                confidence,
                4,
            ),
        }

    if intent == "return_risk":
        if not return_risk_result:
            return {
                "answer": (
                    "I need the order features to "
                    "calculate return risk."
                ),
                "source": "return_risk_tool",
                "confidence": 0.0,
            }

        probability = (
            return_risk_result[
                "return_probability"
            ]
        )

        bucket = (
            return_risk_result[
                "risk_bucket"
            ]
        )

        if order_id:
            answer = (
                f"For order {order_id}, "
                f"the predicted return probability is "
                f"{probability:.2%}, which falls into the "
                f"{bucket} risk bucket."
            )
        else:
            answer = (
                f"The predicted return probability is "
                f"{probability:.2%}, which falls into the "
                f"{bucket} risk bucket."
            )

        return {
            "answer": answer,
            "source": "return_risk_tool",
            "confidence": round(
                probability,
                4,
            ),
        }

    if intent == "product_category":
        if not image_result:
            return {
                "answer": (
                    "I need a valid product image path "
                    "to classify the product."
                ),
                "source": "image_classifier_tool",
                "confidence": 0.0,
            }

        label = image_result[
            "label"
        ]

        confidence = image_result[
            "confidence"
        ]

        answer = (
            f"The product image is classified as "
            f"{label} with "
            f"{confidence:.2%} confidence."
        )

        return {
            "answer": answer,
            "source": "image_classifier_tool",
            "confidence": round(
                confidence,
                4,
            ),
        }

    return {
        "answer": (
            "I can help with return policies, "
            "return risk, and product categories."
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }


def response_as_json(
    response: dict,
) -> str:
    return json.dumps(
        {
            "answer": response[
                "answer"
            ],
            "source": response[
                "source"
            ],
            "confidence": response[
                "confidence"
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def main():
    print(
        "\nSystem prompt:"
    )

    print(
        format_system_prompt()
    )

    print(
        "\nIntent examples:"
    )

    for example in build_intent_examples():
        predicted = classify_intent(
            example["user"]
        )

        print(
            f"User: {example['user']}"
        )

        print(
            f"Expected: {example['intent']}"
        )

        print(
            f"Predicted: {predicted}"
        )

    policy_response = generate_response(
        intent="policy",
        retrieved_chunks=[
            {
                "text": (
                    "Apparel and footwear items are "
                    "eligible for return within 7 days "
                    "of delivery when the item is unused, "
                    "undamaged, and has its original tags "
                    "and packaging."
                ),
                "similarity": 0.4753,
            }
        ],
        grounded=True,
        state={
            "order_id": None,
            "user_message": (
                "What is the return window "
                "for footwear?"
            ),
        },
    )

    risk_response = generate_response(
        intent="return_risk",
        return_risk_result={
            "return_probability": 0.5260934525924563,
            "risk_bucket": "Medium",
        },
        state={
            "order_id": "ORD1001",
            "user_message": (
                "Is order ORD1001 likely to be returned?"
            ),
        },
    )

    image_response = generate_response(
        intent="product_category",
        image_result={
            "label": "Sneaker",
            "confidence": 0.9985365867614746,
        },
        state={
            "order_id": None,
            "user_message": (
                "What category is this product image?"
            ),
        },
    )

    blocked_policy_response = generate_response(
        intent="policy",
        retrieved_chunks=[],
        grounded=False,
        state={
            "order_id": None,
            "user_message": (
                "What is the refund policy for "
                "interplanetary spacecraft?"
            ),
        },
    )

    state_policy_response = generate_response(
        intent="policy",
        retrieved_chunks=[
            {
                "text": (
                    "When an order passes its displayed "
                    "estimated delivery date, the customer "
                    "can use the order support flow to request "
                    "an update."
                ),
                "similarity": 0.4704,
            }
        ],
        grounded=True,
        state={
            "order_id": "ORD2007",
            "user_message": (
                "What can I do about that order?"
            ),
        },
    )

    fresh_policy_response = generate_response(
        intent="policy",
        retrieved_chunks=[
            {
                "text": (
                    "When an order passes its displayed "
                    "estimated delivery date, the customer "
                    "can use the order support flow to request "
                    "an update."
                ),
                "similarity": 0.4704,
            }
        ],
        grounded=True,
        state={
            "order_id": None,
            "user_message": (
                "What can I do about that order?"
            ),
        },
    )

    print(
        "\nPolicy response:"
    )

    print(
        response_as_json(
            policy_response
        )
    )

    print(
        "\nReturn-risk response:"
    )

    print(
        response_as_json(
            risk_response
        )
    )

    print(
        "\nImage response:"
    )

    print(
        response_as_json(
            image_response
        )
    )

    print(
        "\nUngrounded response:"
    )

    print(
        response_as_json(
            blocked_policy_response
        )
    )

    print(
        "\nState-aware response:"
    )

    print(
        response_as_json(
            state_policy_response
        )
    )

    print(
        "\nFresh-conversation response:"
    )

    print(
        response_as_json(
            fresh_policy_response
        )
    )


if __name__ == "__main__":
    main()