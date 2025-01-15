from __future__ import annotations

import os
from pathlib import Path

import torch
import typer

# Dynamically determine project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()


# Define paths relative to the project root
DATA_PATH = PROJECT_ROOT / "data/raw/corruptmnist_v1"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"

# Other code remains the same
train_paths_list = [
    "train_images_0.pt",
    "train_images_1.pt",
    "train_images_2.pt",
    "train_images_3.pt",
    "train_images_4.pt",
    "train_images_5.pt",
]

train_target_list = [
    "train_target_0.pt",
    "train_target_1.pt",
    "train_target_2.pt",
    "train_target_3.pt",
    "train_target_4.pt",
    "train_target_5.pt",
]


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    # Load and concatenate training data
    train_images, train_target = [], []
    for train_path, target_path in zip(train_paths_list, train_target_list):
        train_images.append(torch.load(os.path.join(DATA_PATH, train_path)))
        train_target.append(torch.load(os.path.join(DATA_PATH, target_path)))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Define and load test data
    test_image_path = "test_images.pt"
    test_target_path = "test_target.pt"
    test_images: torch.Tensor = torch.load(os.path.join(DATA_PATH, test_image_path))
    test_target: torch.Tensor = torch.load(os.path.join(DATA_PATH, test_target_path))

    # Reshape and convert data types
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize data
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save(train_images, os.path.join(PROCESSED_DIR, "train_images.pt"))
    torch.save(train_target, os.path.join(PROCESSED_DIR, "train_target.pt"))
    torch.save(test_images, os.path.join(PROCESSED_DIR, "test_images.pt"))
    torch.save(test_target, os.path.join(PROCESSED_DIR, "test_target.pt"))


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load(os.path.join(PROCESSED_DIR, "train_images.pt"))
    train_target = torch.load(os.path.join(PROCESSED_DIR, "train_target.pt"))
    test_images = torch.load(os.path.join(PROCESSED_DIR, "test_images.pt"))
    test_target = torch.load(os.path.join(PROCESSED_DIR, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
