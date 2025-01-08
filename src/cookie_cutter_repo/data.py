from __future__ import annotations

import matplotlib.pyplot as plt  # only needed for plotting
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
import random
import os

DATA_PATH = "data/raw/corruptmnist_v1"
PROCESSED_DIR = "data/processed"
test_path = "/test_images.pt"
test_target_path = "/test_target.pt"

train_paths_list = ["/train_images_0.pt",
                    "/train_images_1.pt",
                    "/train_images_2.pt",
                    "/train_images_3.pt",
                    "/train_images_4.pt",
                    "/train_images_5.pt"]

train_target_list = ["/train_target_0.pt",
                     "/train_target_1.pt",
                     "/train_target_2.pt",
                     "/train_target_3.pt",
                     "/train_target_4.pt",
                     "/train_target_5.pt"]

# 0 must not be rotated
# 1 must be rotated by -45 degrees
# 2 must be rotated by -40 degrees
# 3 must be rotated by 45 degrees
# 4 must not be rotated
# 5 must not be rotated

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()

def preprocess_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(DATA_PATH + train_paths_list[i]))
        train_target.append(torch.load(DATA_PATH + train_target_list[i]))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(DATA_PATH + test_path)
    test_target: torch.Tensor = torch.load(DATA_PATH + test_target_path)  # Updated name

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{PROCESSED_DIR}/train_images.pt")
    torch.save(train_target, f"{PROCESSED_DIR}/train_target.pt")
    torch.save(test_images, f"{PROCESSED_DIR}/test_images.pt")
    torch.save(test_target, f"{PROCESSED_DIR}/test_target.pt")

    return train_set, test_set


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    preprocess_data()