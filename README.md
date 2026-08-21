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