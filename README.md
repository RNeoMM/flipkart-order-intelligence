# Flipkart Order Intelligence & Support Assistant

An end-to-end machine learning and agentic AI project for Flipkart support operations.

The project combines:

1. Return-risk prediction for orders
2. Product-image classification
3. A LangGraph support assistant using both models and a policy knowledge base

---

## Repository Structure

```text
flipkart-order-intelligence/

├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   ├── return_risk_model.pkl
│   ├── product_classifier.pt
│   └── product_classifier_meta.json
├── data/
│   └── sample_images/
├── part1_return_risk/
├── part2_image_classifier/
├── part3_support_agent/
└── transcripts/
```

# Part 1 - Return Risk

Part 1 builds a return-risk prediction pipeline from order data.

The final model is a tuned Random Forest saved as:

```text
models/return_risk_model.pkl
```

The Random Forest F1-maximising threshold is:

```text
t*_rf = 0.50
```

The Part 3 risk buckets use this threshold:

```text
Low: < 0.50
Medium: 0.50 to < 0.65
High: >= 0.65
```

The saved pipeline is loaded directly by the Part 3 return-risk tool.

# Part 2 - Product Image Categoriser

Part 2 uses the real Fashion-MNIST dataset and a pretrained ResNet-18 model.

## Dataset

The pipeline uses:

- 55,000 training images
- 5,000 validation images
- 10,000 test images
- 10 product classes

Source:

https://github.com/zalandoresearch/fashion-mnist

## Preprocessing

The original grayscale images are:

- converted from 1 channel to 3 channels
- resized to 224x224
- normalized using ImageNet mean and standard deviation

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

## Model

A pretrained ResNet-18 backbone is frozen during feature extraction.

The extracted 512-dimensional features are cached and used to train a new 10-class classifier head.

Training uses Adam with a learning rate of `0.001` and a batch size of `256`.

## Result

Final test accuracy:

```text
88.61%
```

The model exceeded the required 80% test accuracy.

The largest real confusion pairs were:

- Shirt -> T-shirt/top: 126 errors
- Shirt -> Coat: 104 errors

The saved model is:

```text
models/product_classifier.pt
```

Ten real Fashion-MNIST test images are exported to:

```text
data/sample_images/
```

# Part 3 - Flipkart Support Agent

Part 3 combines the two saved models with a local policy knowledge base into one LangGraph support assistant.

## Flow

```text
User
  ↓
Input Guardrail
  ↓
Intent Classification
  ↓
Conditional Routing
  ├── Policy → RAG
  ├── Return Risk → Part 1 Model
  └── Product Category → Part 2 Model
  ↓
Response Generation
  ↓
Structured JSON
```

The graph contains:

- input guardrail
- intent node
- policy retrieval node
- return-risk tool
- product-image tool
- response node

## Policy Knowledge Base

The knowledge base contains 14 project-authored policy documents.

The documents are split sentence-wise into 28 chunks.

Each chunk keeps its parent `document_id` for document-level retrieval evaluation.

Main topics include:

- apparel and footwear returns
- electronics returns
- home returns
- COD refunds
- delivery timelines
- reverse pickup
- damaged or wrong products
- non-returnable items
- replacements

## Embeddings and Search

Chunks are embedded locally with:

```text
all-MiniLM-L6-v2
```

and indexed using:

```text
FAISS
```

No paid API or account is required.

## Real Tools

### Return Risk

The agent loads:

```text
models/return_risk_model.pkl
```

Verified example:

```text
Probability: 52.61%
Risk bucket: Medium
```

### Product Image

The agent loads:

```text
models/product_classifier.pt
```

and works with the real PNG files in `data/sample_images/`.

Verified example:

```text
07_sneaker.png
Prediction: Sneaker
Confidence: 99.85%
```

## Prompt Design

The response generator follows the 4S principles:

- Specific
- Short
- Surround
- Single

It also uses role prompting and few-shot intent examples for:

- policy
- return risk
- product category

Final responses use:

```json
{
  "answer": "string",
  "source": "policy_kb | return_risk_tool | image_classifier_tool",
  "confidence": 0.0
}
```

## MOCK_LLM

`MOCK_LLM` is the default mode.

It is deterministic and requires:

- no API key
- no paid LLM
- no external LLM call

## Guardrails

Input-side filtering blocks prompt-injection patterns such as:

```text
ignore previous instructions
ignore all rules
pretend you are...
```

For policy questions, the groundedness threshold is:

```text
0.45
```

An unrelated test produced:

```text
Best similarity: 0.2450
Threshold: 0.45
Grounded: False
```

The agent correctly refused to invent a policy answer.

## Conversational State

The agent keeps short-term state within a conversation.

Example:

```text
Turn 1:
My order ORD2007 is delayed.

Turn 2:
What can I do about that order?
```

The order ID remains available as `ORD2007`.

A new conversation starts without that state and asks for the order ID again.

## Retrieval Evaluation

Retrieval is evaluated at the document level after deduplicating chunks.

| Query | Precision@3 | Recall@3 |
|---|---:|---:|
| Q1 | 0.3333 | 1.0000 |
| Q2 | 0.3333 | 0.5000 |
| Q3 | 0.3333 | 1.0000 |
| Q4 | 0.3333 | 1.0000 |
| Q5 | 0.6667 | 1.0000 |
| Average | **0.4000** | **0.9000** |

## Transcripts

The required test conversations are saved in:

```text
transcripts/
```

They cover:

- policy questions
- return-risk prediction
- product-image classification
- multi-turn state
- fresh conversation
- prompt injection
- ungrounded policy refusal

## Running the Project

From the repository root:

```bash
python -m part3_support_agent.graph
```

Run retrieval evaluation:

```bash
python -m part3_support_agent.retrival_eval
```

Run MOCK_LLM tests:

```bash
python -m part3_support_agent.mock_llm
```

Regenerate transcripts:

```bash
python -m part3_support_agent.save_transcripts
```

The complete system runs locally without requiring a paid LLM API.