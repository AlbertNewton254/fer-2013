"""
src/data/fetch_data.py

Downloads the FER2013 dataset from Kaggle and saves it to data/raw/.
"""

from pathlib import Path
from src.config import get_config

CONFIG = get_config()

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / CONFIG["data"]["raw_data_dir"]

DATASET_NAME = CONFIG["data"]["kaggle_dataset"]


def fetch_data() -> None:
    """Download the FER2013 dataset via the Kaggle API."""
    try:
        import kaggle

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset {DATASET_NAME} from Kaggle...")
        kaggle.api.dataset_download_files(DATASET_NAME, path=RAW_DATA_DIR, unzip=True)
        print(f"Dataset saved to {RAW_DATA_DIR}")

    except Exception as e:
        print(f"Error fetching dataset: {e}")


if __name__ == "__main__":
    fetch_data()