import os
from pathlib import Path

import torch
import typer
from model import FredNet

from data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Adjust to match project structure

def evaluate(model_checkpoint: str = os.path.join(PROJECT_ROOT, "models/model.pt")) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Model checkpoint: {model_checkpoint}")

    model = FredNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
