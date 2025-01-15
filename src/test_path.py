from __future__ import annotations

import os
from pathlib import Path

import torch
import typer

# Dynamically determine project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Adjust to match project structure

# Define paths relative to the project root
DATA_PATH = PROJECT_ROOT / "data/raw/corruptmnist_v1"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"


print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_PATH: {DATA_PATH}")
print(f"PROCESSED_DIR: {PROCESSED_DIR}")
