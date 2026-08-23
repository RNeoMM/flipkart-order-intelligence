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