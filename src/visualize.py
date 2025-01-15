import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from model import FredNet  # Import FredNet from the local model.py

# Define project root and directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Adjust to reach the project root
MODEL_DIR = PROJECT_ROOT / "models"
PLOT_DIR = PROJECT_ROOT / "reports/figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


def show_random_images_grid(dataset: torch.utils.data.Dataset, grid_size: int = 10, save_path: Path = None) -> None:
    """Display a grid of random images and their labels from a dataset."""
    images, labels = dataset.tensors  # Unpack images and labels from the TensorDataset
    total_images = grid_size * grid_size  # Total number of images to display
    indices = random.sample(range(len(images)), total_images)  # Randomly select indices

    selected_images = images[indices]
    selected_labels = labels[indices]

    # Plot the selected images and their labels in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(selected_images[i].squeeze(), cmap="gray")
        ax.set_title(f"{selected_labels[i].item()}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Grid plot saved at: {save_path}")
    plt.show()

def visualize(
    model_checkpoint: str = str(MODEL_DIR / "model.pt"),  # Default checkpoint path
    grid_size: int = 10,  # Grid size for random images
    figure_name: str = "random_images_grid.png"  # Default plot file name
) -> None:
    """Visualize a grid of random images and their labels."""
    # Correct the path to point to the correct processed data folder
    test_images_path = PROJECT_ROOT / "data/processed/test_images.pt"
    test_target_path = PROJECT_ROOT / "data/processed/test_target.pt"

    # Load test data
    test_images = torch.load(test_images_path)
    test_target = torch.load(test_target_path)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    # Display a grid of random images
    plot_path = PLOT_DIR / figure_name
    show_random_images_grid(test_dataset, grid_size=grid_size, save_path=plot_path)


if __name__ == "__main__":
    typer.run(visualize)
