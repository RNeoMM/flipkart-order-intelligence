import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "product_classifier.pt"
)

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

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def main():
    image_path = Path(sys.argv[1])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    torch.set_num_threads(1)

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        512,
        10,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(
        image_path
    ).convert("L")

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(tensor),
            dim=1,
        )[0]

    class_id = int(
        probabilities.argmax().item()
    )

    result = {
        "label": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": float(
            probabilities[class_id].item()
        ),
        "image_path": str(
            image_path
        ),
    }

    print(
        json.dumps(result)
    )


if __name__ == "__main__":
    main()