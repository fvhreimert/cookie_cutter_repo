from pathlib import Path

import torch
import typer
from matplotlib import pyplot as plt
from model import FredNet
from torch import nn
from tqdm import tqdm
import wandb
from data import corrupt_mnist

app = typer.Typer()

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define project root and directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Adjust based on your project structure
MODEL_DIR = PROJECT_ROOT / "models"
PLOT_DIR = PROJECT_ROOT / "reports/figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


def plot_training_statistics(train_loss, train_accuracy, save_path: Path):
    """Create and save a publication-quality plot of training statistics."""
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, train_loss, label="Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracy, label="Accuracy", color="orange", linewidth=2)
    ax2.set_ylabel("Accuracy", fontsize=14, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    fig.tight_layout()
    ax1.legend(loc="upper left", fontsize=12)
    ax2.legend(loc="upper right", fontsize=12)

    plt.title("Training Statistics", fontsize=16)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Training plot saved to {save_path}")

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10):
    """Train a model on MNIST and save training statistics."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Initialize WandB
    wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    model = FredNet().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    train_accuracy = []

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for img, target in tqdm(train_dataloader, leave=False):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            running_loss += loss.item()
            running_accuracy += (y_pred.argmax(dim=1) == target).float().mean().item()
            
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
        
        avg_loss = running_loss / len(train_dataloader)
        avg_accuracy = running_accuracy / len(train_dataloader)

        train_loss.append(avg_loss)
        train_accuracy.append(avg_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    print("Training complete.")
    model_path = MODEL_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

    # Log the model as an artifact in WandB
    artifact = wandb.Artifact("trained_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print("Model logged as artifact in WandB.")

    plot_path = PLOT_DIR / "training_statistics.png"
    plot_training_statistics(train_loss, train_accuracy, save_path=plot_path)

    # End WandB run
    wandb.finish()
    
if __name__ == "__main__":
    typer.run(train)
