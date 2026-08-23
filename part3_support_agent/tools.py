from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import json
import subprocess
import sys

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


ROOT_DIR = Path(__file__).resolve().parent.parent

RETURN_RISK_MODEL_PATH = (
    ROOT_DIR / "models" / "return_risk_model.pkl"
)

PRODUCT_CLASSIFIER_PATH = (
    ROOT_DIR / "models" / "product_classifier.pt"
)

RETURN_RISK_THRESHOLD = 0.50
HIGH_RISK_OFFSET = 0.15

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

FEATURE_DIM = 512
NUM_CLASSES = 10

RETURN_RISK_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
    "product_category",
    "payment_method",
]

_PRODUCT_MODEL = None
_RETURN_MODEL = None


def load_return_risk_model():
    global _RETURN_MODEL

    if _RETURN_MODEL is None:
        _RETURN_MODEL = joblib.load(
            RETURN_RISK_MODEL_PATH
        )

    return _RETURN_MODEL


def check_return_risk(
    order_features: dict,
) -> dict:
    required = set(
        RETURN_RISK_FEATURES
    )

    received = set(
        order_features.keys()
    )

    missing = sorted(
        required - received
    )

    if missing:
        raise ValueError(
            "Missing required order features: "
            + ", ".join(missing)
        )

    model = load_return_risk_model()

    if "classifier" in model.named_steps:
       model.named_steps["classifier"].n_jobs = 1

    row = {
        feature: order_features[feature]
        for feature in RETURN_RISK_FEATURES
    }

    frame = pd.DataFrame(
        [row]
    )

    probabilities = model.predict_proba(
        frame
    )

    return_probability = float(
        probabilities[0][1]
    )

    medium_cutoff = (
        RETURN_RISK_THRESHOLD
        + HIGH_RISK_OFFSET
    )

    if return_probability < RETURN_RISK_THRESHOLD:
        risk_bucket = "Low"
    elif return_probability >= medium_cutoff:
        risk_bucket = "High"
    else:
        risk_bucket = "Medium"

    return {
        "return_probability": return_probability,
        "risk_bucket": risk_bucket,
        "threshold_t_rf": RETURN_RISK_THRESHOLD,
        "medium_cutoff": RETURN_RISK_THRESHOLD,
        "high_cutoff": medium_cutoff,
    }


def load_product_classifier():
    global _PRODUCT_MODEL

    if _PRODUCT_MODEL is None:
        model = resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
        )

        state_dict = torch.load(
            PRODUCT_CLASSIFIER_PATH,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        _PRODUCT_MODEL = model

    return _PRODUCT_MODEL


PRODUCT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)

def classify_product_image(
    image_path: str,
) -> dict:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    worker = (
        ROOT_DIR
        / "part3_support_agent"
        / "image_worker.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    output_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    if not output_lines:
        raise RuntimeError(
            "Image classifier worker returned no output."
        )

    return json.loads(
        output_lines[-1]
    )


def main():
    print(
        "Return-risk model:"
    )

    print(
        load_return_risk_model()
    )

    print(
        "\nRandom Forest "
        "threshold t*_rf:"
    )

    print(
        RETURN_RISK_THRESHOLD
    )

    print(
        "\nRisk bucket cut points:"
    )

    print(
        f"Low: probability < "
        f"{RETURN_RISK_THRESHOLD:.2f}"
    )

    print(
        f"Medium: "
        f"{RETURN_RISK_THRESHOLD:.2f} "
        f"<= probability < "
        f"{RETURN_RISK_THRESHOLD + HIGH_RISK_OFFSET:.2f}"
    )

    print(
        f"High: probability >= "
        f"{RETURN_RISK_THRESHOLD + HIGH_RISK_OFFSET:.2f}"
    )

    sample_order = {
        "price_inr": 2499,
        "discount_pct": 20,
        "customer_tenure_days": 900,
        "num_previous_orders": 18,
        "num_previous_returns": 2,
        "delivery_distance_km": 12,
        "delivery_days": 4,
        "is_weekend_order": 0,
        "rating_given": 1,
        "product_category": "Apparel",
        "payment_method": "COD",
    }

    risk_result = check_return_risk(
        sample_order
    )

    print(
        "\nReturn-risk test:"
    )

    print(
        risk_result
    )

    sample_image = (
        ROOT_DIR
        / "data"
        / "sample_images"
        / "07_sneaker.png"
    )

    image_result = classify_product_image(
        str(sample_image)
    )

    print(
        "\nProduct-image test:"
    )

    print(
        image_result
    )


if __name__ == "__main__":
    main()