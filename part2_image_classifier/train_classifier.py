from pathlib import Path
import copy
import json

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
SAMPLE_DIR = DATA_DIR / "sample_images"

MODEL_PATH = MODEL_DIR / "product_classifier.pt"
META_PATH = MODEL_DIR / "product_classifier_meta.json"
FEATURE_CACHE = DATA_DIR / "resnet18_features_cache.npz"

BATCH_SIZE = 128
HEAD_BATCH_SIZE = 256
HEAD_LR = 1e-3
HEAD_EPOCHS = 10

FINE_TUNE_LR = 1e-4
FINE_TUNE_EPOCHS = 5

VAL_SIZE = 5000
NUM_CLASSES = 10
FEATURE_DIM = 512

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

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("PART 2 - PRODUCT IMAGE CATEGORISER")
print("=" * 70)
print(f"Device: {DEVICE}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

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


def build_frozen_backbone():
    backbone = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone.fc = nn.Identity()
    backbone = backbone.to(DEVICE)
    backbone.eval()

    return backbone


def extract_features(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    features = []

    model.eval()

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(DEVICE)

            batch_features = model(images)

            features.append(
                batch_features.detach().cpu().numpy()
            )

            if batch_index % 50 == 0:
                print(
                    f"Feature batch "
                    f"{batch_index}/{len(loader)}"
                )

    return np.concatenate(
        features,
        axis=0,
    ).astype(np.float32)


def evaluate_head(classifier, loader):
    classifier.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = classifier(features)
            preds = logits.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    return accuracy_score(
        true_labels,
        predictions,
    )


def evaluate_full_model(model, loader):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            logits = model(images)
            preds = logits.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.numpy()
            )

    return accuracy_score(
        true_labels,
        predictions,
    )


def main():
    print("\n[Task 1] Loading real Fashion-MNIST...")

    train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        transform=transform,
        download=True,
    )

    targets = np.asarray(
        train_dataset.targets
    )

    all_indices = np.arange(
        len(train_dataset)
    )

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VAL_SIZE,
        stratify=targets,
        random_state=SEED,
    )

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    val_subset = Subset(
        train_dataset,
        val_indices,
    )

    print("\nSplit sizes:")
    print(
        f"Train      : "
        f"{len(train_subset):,}"
    )
    print(
        f"Validation : "
        f"{len(val_subset):,}"
    )
    print(
        f"Test       : "
        f"{len(test_dataset):,}"
    )

    print("\nClasses:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i}: {name}")

    print("\n[Task 2] Preprocessing:")
    print("Grayscale 1-channel -> 3 channels")
    print("Resize: 224 x 224")
    print(
        "ImageNet mean: "
        "[0.485, 0.456, 0.406]"
    )
    print(
        "ImageNet std : "
        "[0.229, 0.224, 0.225]"
    )

    print(
        "\n[Task 3] "
        "Loading pretrained ResNet-18..."
    )

    backbone = build_frozen_backbone()

    print(
        f"Feature dimension: "
        f"{FEATURE_DIM}"
    )

    print(
        "Backbone frozen: yes"
    )

    y_train = targets[
        train_indices
    ].astype(np.int64)

    y_val = targets[
        val_indices
    ].astype(np.int64)

    if FEATURE_CACHE.exists():
        print(
            "\nLoading cached "
            "ResNet-18 features..."
        )

        cached = np.load(
            FEATURE_CACHE
        )

        X_train_features = cached[
            "X_train_features"
        ]

        y_train = cached[
            "y_train"
        ]

        X_val_features = cached[
            "X_val_features"
        ]

        y_val = cached[
            "y_val"
        ]

    else:
        print(
            "\nExtracting frozen "
            "ResNet-18 features..."
        )

        print(
            "Test set remains untouched."
        )

        X_train_features = extract_features(
            backbone,
            train_subset,
        )

        X_val_features = extract_features(
            backbone,
            val_subset,
        )

        np.savez_compressed(
            FEATURE_CACHE,
            X_train_features=X_train_features,
            y_train=y_train,
            X_val_features=X_val_features,
            y_val=y_val,
        )

        print(
            f"Cached features to: "
            f"{FEATURE_CACHE}"
        )

    print(
        "\nFeature shapes:",
        X_train_features.shape,
        X_val_features.shape,
    )

    print(
        "\nTraining new "
        "10-class classifier head..."
    )

    head = nn.Linear(
        FEATURE_DIM,
        NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=HEAD_LR,
    )

    train_tensor_dataset = TensorDataset(
        torch.from_numpy(
            X_train_features
        ),
        torch.from_numpy(
            y_train
        ),
    )

    val_tensor_dataset = TensorDataset(
        torch.from_numpy(
            X_val_features
        ),
        torch.from_numpy(
            y_val
        ),
    )

    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Optimizer: Adam | "
        f"LR={HEAD_LR} | "
        f"Batch={HEAD_BATCH_SIZE} | "
        f"Epochs={HEAD_EPOCHS}"
    )

    best_head_state = copy.deepcopy(
        head.state_dict()
    )

    best_val_accuracy = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):
        head.train()

        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = head(features)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_accuracy = evaluate_head(
            head,
            val_loader,
        )

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{HEAD_EPOCHS} "
            f"| loss={avg_loss:.4f} "
            f"| val_acc="
            f"{val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = (
                val_accuracy
            )

            best_head_state = (
                copy.deepcopy(
                    head.state_dict()
                )
            )

            patience_counter = 0

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    "Early stopping "
                    "head training."
                )
                break

    head.load_state_dict(
        best_head_state
    )

    print(
        "\nFeature-extraction "
        "validation accuracy: "
        f"{best_val_accuracy:.4f} "
        f"({best_val_accuracy * 100:.2f}%)"
    )

    fine_tuned = False

    fine_tune_before = (
        best_val_accuracy
    )

    fine_tune_after = None

    final_model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    final_model.fc = nn.Linear(
        FEATURE_DIM,
        NUM_CLASSES,
    )

    with torch.no_grad():
        final_model.fc.weight.copy_(
            head.weight.detach().cpu()
        )

        final_model.fc.bias.copy_(
            head.bias.detach().cpu()
        )

    if best_val_accuracy < 0.80:
        print(
            "\n[Task 4] "
            "Validation accuracy < 80%."
        )

        print(
            "Unfreezing "
            "ResNet-18 layer4 "
            "and classifier head."
        )

        fine_tuned = True

        for parameter in (
            final_model.parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            final_model.layer4.parameters()
        ):
            parameter.requires_grad = True

        for parameter in (
            final_model.fc.parameters()
        ):
            parameter.requires_grad = True

        final_model = (
            final_model.to(DEVICE)
        )

        optimizer_ft = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                final_model.parameters(),
            ),
            lr=FINE_TUNE_LR,
        )

        train_image_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        val_image_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        best_ft_state = copy.deepcopy(
            final_model.state_dict()
        )

        best_ft_accuracy = (
            best_val_accuracy
        )

        for epoch in range(
            1,
            FINE_TUNE_EPOCHS + 1,
        ):
            final_model.train()

            for images, labels in (
                train_image_loader
            ):
                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )

                optimizer_ft.zero_grad()

                logits = final_model(
                    images
                )

                loss = criterion(
                    logits,
                    labels,
                )

                loss.backward()
                optimizer_ft.step()

            val_accuracy = (
                evaluate_full_model(
                    final_model,
                    val_image_loader,
                )
            )

            print(
                f"Fine-tune epoch "
                f"{epoch:02d}/"
                f"{FINE_TUNE_EPOCHS} "
                f"| val_acc="
                f"{val_accuracy:.4f}"
            )

            if val_accuracy > (
                best_ft_accuracy
            ):
                best_ft_accuracy = (
                    val_accuracy
                )

                best_ft_state = (
                    copy.deepcopy(
                        final_model.state_dict()
                    )
                )

        final_model.load_state_dict(
            best_ft_state
        )

        fine_tune_after = (
            best_ft_accuracy
        )

        print(
            "\nBefore fine-tuning: "
            f"{fine_tune_before * 100:.2f}%"
        )

        print(
            "After fine-tuning: "
            f"{fine_tune_after * 100:.2f}%"
        )

    else:
        print(
            "\n[Task 4] "
            "Feature extraction "
            "is sufficient."
        )

        print(
            "Before fine-tuning: "
            f"{best_val_accuracy * 100:.2f}%"
        )

        print(
            "After fine-tuning: "
            "N/A (not required)"
        )

    final_model = (
        final_model.to(DEVICE)
    )

    final_model.eval()

    print(
        "\n" + "=" * 70
    )

    print(
        "[Task 5] "
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for images, labels in (
            test_loader
        ):
            images = images.to(
                DEVICE
            )

            logits = final_model(
                images
            )

            predictions = (
                logits.argmax(dim=1)
            )

            all_test_predictions.extend(
                predictions.cpu().numpy()
            )

            all_test_labels.extend(
                labels.numpy()
            )

    all_test_predictions = np.asarray(
        all_test_predictions
    )

    all_test_labels = np.asarray(
        all_test_labels
    )

    test_accuracy = accuracy_score(
        all_test_labels,
        all_test_predictions,
    )

    print(
        "\nFinal test accuracy: "
        f"{test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        "\nPer-class "
        "precision / recall / F1:"
    )

    print(
        classification_report(
            all_test_labels,
            all_test_predictions,
            target_names=CLASS_NAMES,
        )
    )

    cm = confusion_matrix(
        all_test_labels,
        all_test_predictions,
    )

    print(
        "\n10 x 10 Confusion Matrix"
    )

    print(
        "(rows=true, "
        "columns=predicted)\n"
    )

    print(
        "True\\Pred "
        + " ".join(
            f"{i:5d}"
            for i in range(
                NUM_CLASSES
            )
        )
    )

    for i, row in enumerate(cm):
        print(
            f"{i:10d} "
            + " ".join(
                f"{value:5d}"
                for value in row
            )
        )

    off_diagonal = cm.copy()

    np.fill_diagonal(
        off_diagonal,
        0,
    )

    flat_indices = np.argsort(
        off_diagonal.ravel()
    )[::-1]

    top_pairs = []

    for flat_index in flat_indices:
        count = (
            off_diagonal.ravel()[
                flat_index
            ]
        )

        if count <= 0:
            break

        true_class = (
            flat_index //
            NUM_CLASSES
        )

        predicted_class = (
            flat_index %
            NUM_CLASSES
        )

        top_pairs.append(
            (
                true_class,
                predicted_class,
                int(count),
            )
        )

        if len(top_pairs) == 2:
            break

    print(
        "\nTop two confusion pairs "
        "from real test predictions:"
    )

    for (
        true_class,
        predicted_class,
        count,
    ) in top_pairs:

        print(
            f"- "
            f"{CLASS_NAMES[true_class]} "
            f"-> "
            f"{CLASS_NAMES[predicted_class]}: "
            f"{count} errors"
        )

    print(
        "\n[Task 7] "
        "Saving model artifact..."
    )

    torch.save(
        final_model.state_dict(),
        MODEL_PATH,
    )

    print(
        f"Saved model to: "
        f"{MODEL_PATH}"
    )

    metadata = {
        "backbone": "ResNet-18",
        "pretrained": True,
        "dataset": "Fashion-MNIST",
        "dataset_source": (
            "https://github.com/"
            "zalandoresearch/"
            "fashion-mnist"
        ),
        "train_size": len(
            train_subset
        ),
        "validation_size": len(
            val_subset
        ),
        "test_size": len(
            test_dataset
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "feature_extraction_validation_accuracy": float(
            best_val_accuracy
        ),
        "fine_tuned": fine_tuned,
        "fine_tune_before_accuracy": float(
            fine_tune_before
        ),
        "fine_tune_after_accuracy": (
            None
            if fine_tune_after is None
            else float(
                fine_tune_after
            )
        ),
        "input_size": [
            224,
            224,
        ],
        "imagenet_mean": [
            0.485,
            0.456,
            0.406,
        ],
        "imagenet_std": [
            0.229,
            0.224,
            0.225,
        ],
        "head_optimizer": "Adam",
        "head_learning_rate": HEAD_LR,
        "head_batch_size": HEAD_BATCH_SIZE,
        "head_max_epochs": HEAD_EPOCHS,
        "fine_tune_learning_rate": FINE_TUNE_LR,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "class_names": CLASS_NAMES,
        "confusion_pairs": [
            {
                "true": CLASS_NAMES[a],
                "predicted": CLASS_NAMES[b],
                "count": count,
            }
            for a, b, count in top_pairs
        ],
    }

    with open(
        META_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved metadata to: "
        f"{META_PATH}"
    )

    print(
        "\n[Task 8] "
        "Exporting real test images..."
    )

    raw_test_dataset = (
        datasets.FashionMNIST(
            root=str(DATA_DIR),
            train=False,
            transform=None,
            download=False,
        )
    )

    raw_targets = np.asarray(
        raw_test_dataset.targets
    )

    saved_files = []

    for (
        class_id,
        class_name,
    ) in enumerate(CLASS_NAMES):

        indices = np.where(
            raw_targets == class_id
        )[0]

        image_index = int(
            indices[0]
        )

        image, label = (
            raw_test_dataset[
                image_index
            ]
        )

        filename = (
            f"{class_id:02d}_"
            f"{class_name.lower()}"
            .replace(
                "/",
                "_",
            )
            .replace(
                " ",
                "_",
            )
            + ".png"
        )

        output_path = (
            SAMPLE_DIR /
            filename
        )

        image.save(
            output_path
        )

        saved_files.append(
            output_path
        )

        print(
            f"Saved "
            f"{output_path.name} "
            f"(true label: "
            f"{CLASS_NAMES[label]})"
        )

    print(
        f"\nExported "
        f"{len(saved_files)} "
        f"real Fashion-MNIST "
        f"test images."
    )

    print(
        "\nPart 2 training complete."
    )

    return saved_files


_INFERENCE_MODEL = None


def load_product_classifier(
    model_path: str = str(MODEL_PATH),
):
    global _INFERENCE_MODEL

    if _INFERENCE_MODEL is None:

        model = resnet18(
            weights=None
        )

        model.fc = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
        )

        state_dict = torch.load(
            model_path,
            map_location="cpu",
        )

        model.load_state_dict(
            state_dict
        )

        model.eval()

        _INFERENCE_MODEL = model

    return _INFERENCE_MODEL


def classify_product_image(
    image_path: str,
    model_path: str = str(MODEL_PATH),
) -> dict:

    model = load_product_classifier(
        model_path
    )

    image = Image.open(
        image_path
    ).convert("L")

    image = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(
            image
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    class_id = int(
        probabilities.argmax().item()
    )

    return {
        "label": CLASS_NAMES[
            class_id
        ],
        "class_id": class_id,
        "confidence": float(
            probabilities[
                class_id
            ].item()
        ),
        "all_probs": {
            name: float(
                probabilities[i].item()
            )
            for i, name in enumerate(
                CLASS_NAMES
            )
        },
    }


if __name__ == "__main__":
    sample_files = main()

    print(
        "\n[Smoke test] "
        "Testing classify_product_image()"
    )

    for image_path in sample_files[:5]:
        result = classify_product_image(
            str(image_path)
        )

        print(
            f"{image_path.name} -> "
            f"{result['label']} "
            f"(confidence="
            f"{result['confidence']:.4f})"
        )

    print(
        "\nPart 2 completed successfully."
    )