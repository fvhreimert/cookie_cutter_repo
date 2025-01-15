import os

# Dynamically determine project root directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Adjust to match project structure
_TEST_ROOT = os.path.join(_PROJECT_ROOT, "tests/")  # root of test folder
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data/", "processed/")  # root of data
