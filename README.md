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

Part 1 generates a deterministic order dataset and trains a return-risk model using scikit-learn.

The final tuned Random Forest pipeline is saved as:

```text
models/return_risk_model.pkl
```

Dataset summary:

```text
Rows: 6000
Columns: 13
Return rate: 22.75%
Missing rating_given: 13.05%
Missingness: MAR
```

The final Random Forest F1-maximising threshold is:

```text
t*_rf = 0.46
```

Part 3 uses this threshold for risk buckets:

```text
Low: < 0.46
Medium: 0.46 to < 0.61
High: >= 0.61
```

The saved Random Forest is loaded directly by the Part 3 return-risk tool.

### Run Part 1

Generate the dataset:

```bash
python part1_return_risk/generate_orders.py
```

Train and evaluate the return-risk pipeline:

```bash
python part1_return_risk/return_risk.py
```

# Part 2 - Product Image Categoriser

Part 2 uses the Fashion-MNIST dataset with a pretrained ResNet-18 model.

## Dataset

```text
Training:   55,000
Validation: 5,000
Test:       10,000
Classes:    10
```

Source: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Preprocessing

Images are:

* converted from 1 channel to 3 channels
* resized to 224x224
* normalized with ImageNet mean/std

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

A pretrained ResNet-18 backbone is used for feature extraction, with a new 10-class classifier head trained using Adam.

Final test accuracy:

```text
88.61%
```

The main real confusion pairs were:

```text
Shirt -> T-shirt/top: 126 errors
Shirt -> Coat:        104 errors
```

The trained model is saved as:

```text
models/product_classifier.pt
```

Real test images are exported to:

```text
data/sample_images/
```

### Run Part 2

```bash
python part2_image_classifier/train_classifier.py
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

The graph includes an intent node, retrieval node, tool-calling logic, response generation, and conditional routing.

## Policy Knowledge Base

The project contains 14 authored policy documents split into 28 sentence-level chunks.

Topics include:

* apparel and footwear returns
* electronics and home returns
* COD refunds
* delivery timelines
* reverse pickup
* damaged or incorrect products
* non-returnable items
* replacements

## Embeddings and Search

Chunks are embedded locally using:

```text
all-MiniLM-L6-v2
```

and indexed with:

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

and uses the saved Random Forest probability output with:

```text
t*_rf = 0.46
```

### Product Image

The agent loads:

```text
models/product_classifier.pt
```

and predicts from the real PNG files in:

```text
data/sample_images/
```

## Prompt Design

The response generator uses the 4S principles:

* Specific
* Short
* Surround
* Single

It also uses role prompting and few-shot intent examples.

Final responses follow this JSON structure:

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

* no API key
* no paid LLM
* no external LLM call

## Guardrails

Input-side filtering blocks common prompt-injection patterns such as:

```text
ignore previous instructions
ignore all rules
pretend you are...
```

Policy answers also use a groundedness threshold of:

```text
0.45
```

An unrelated test produced:

```text
Best similarity: 0.2450
Threshold: 0.45
Grounded: False
```

The agent correctly refused to answer without sufficient supporting information.

## Conversational State

Short-term state is maintained within a conversation.

Example:

```text
Turn 1:
My order ORD2007 is delayed.

Turn 2:
What can I do about that order?
```

The order ID remains available as `ORD2007`.

A fresh conversation starts with no previous order ID.

## Retrieval Evaluation

Evaluation is performed at the document level after mapping and deduplicating retrieved chunks.

| Query       | Precision@3 |   Recall@3 |
| ----------- | ----------: | ---------: |
| Q1          |      0.3333 |     1.0000 |
| Q2          |      0.3333 |     0.5000 |
| Q3          |      0.3333 |     1.0000 |
| Q4          |      0.3333 |     1.0000 |
| Q5          |      0.6667 |     1.0000 |
| **Average** |  **0.4000** | **0.9000** |

## Transcripts

The required test conversations are available here:

* [Policy - Apparel](transcripts/01_policy_apparel.txt)
* [Policy - COD Refund](transcripts/02_policy_cod_refund.txt)
* [Return Risk](transcripts/03_return_risk.txt)
* [Product Category](transcripts/04_product_category.txt)
* [Multi-turn State](transcripts/05_multiturn_state.txt)
* [Fresh Conversation](transcripts/06_fresh_conversation.txt)
* [Prompt Injection](transcripts/07_prompt_injection.txt)
* [Ungrounded Policy](transcripts/08_ungrounded_policy.txt)

## Running the Project

From the repository root:

Run the full support agent:

```bash
python -m part3_support_agent.graph
```

Run retrieval evaluation:

```bash
python -m part3_support_agent.retrival_eval
```

Run the MOCK_LLM module:

```bash
python -m part3_support_agent.mock_llm
```

Regenerate transcripts:

```bash
python -m part3_support_agent.save_transcripts
```

The complete project runs locally using the saved models, FAISS knowledge base, and deterministic `MOCK_LLM` mode without requiring a paid LLM API.

## Verification

The support agent was tested locally in default `MOCK_LLM` mode.

Return-risk tool verification:

* `t*_rf`: 0.46
* Medium cutoff: 0.46
* High cutoff: 0.61

The agent also verifies multi-turn state, fresh-conversation reset, prompt-injection blocking, and refusal of ungrounded policy questions.
