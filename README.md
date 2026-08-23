# Flipkart Order Intelligence & Support Assistant

An end-to-end machine learning and agentic AI system for Flipkart support operations.

The project connects three components:

1. Return-risk scoring for orders
2. Product-image classification
3. A support assistant that can use both trained models and a grounded policy knowledge base

---

## Repository Structure

```text
flipkart-order-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── return_risk_model.pkl
├── part1_return_risk/
│   ├── generate_orders.py
│   ├── orders_dataset.csv
│   └── return_risk.py
├── part2_image_classifier/
└── part3_support_agent/

# Part 2 - Product Image Categoriser

Part 2 uses the real Fashion-MNIST benchmark dataset and a pretrained ResNet-18 transfer-learning pipeline.

## Dataset

Fashion-MNIST is downloaded automatically with torchvision.

The pipeline uses:

- 55,000 training images
- 5,000 validation images
- 10,000 untouched test images
- 10 classes

The dataset source is:

https://github.com/zalandoresearch/fashion-mnist

## Preprocessing

The original Fashion-MNIST images are grayscale 28x28 images.

They are processed by:

- Replicating the grayscale channel to 3 channels
- Resizing to 224x224
- ImageNet normalization

ImageNet mean:

```text
[0.485, 0.456, 0.406]


# Part 3 - Flipkart Support Agent

Part 3 connects the Part 1 return-risk model, the Part 2 product-image classifier, and a local policy knowledge base into one LangGraph support assistant.

## Part 3 Architecture

The assistant uses the following flow:

User message
→ input guardrail
→ intent classification
→ conditional routing
→ policy retrieval or model tool
→ grounded response generation
→ structured JSON response

The graph contains six nodes:

- input_guardrail
- intent
- policy_retrieval
- return_risk_tool
- product_image_tool
- response

Conditional edges route requests according to intent:

- policy → policy retrieval
- return_risk → Part 1 return-risk tool
- product_category → Part 2 image-classifier tool

## Policy Knowledge Base

The knowledge base contains 14 project-authored policy documents.

Each policy document contains multiple sentences and is split sentence-wise into 28 chunks.

Each chunk retains its parent `document_id` so retrieval evaluation can be performed at the document level.

The main policy topics include:

- apparel and footwear returns
- electronics returns
- home-product returns
- COD refunds
- prepaid refunds
- delivery SLAs
- delayed delivery
- reverse pickup
- damaged products
- wrong products
- non-returnable items
- refund after reverse pickup
- replacement eligibility
- return-condition requirements

## Embedding and Vector Search

Policy chunks are embedded locally using:

```text
all-MiniLM-L6-v2