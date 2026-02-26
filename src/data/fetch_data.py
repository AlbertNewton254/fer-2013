"""
fetch_data.py

Fetch data from Kaggle and save it to the data directory
"""

from pathlib import Path
import os

import kaggle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

DATASET_NAME = 'msambare/fer2013'

def fetch_data():
    """
    Fetch data from Kaggle and save it to the data directory
    """
    try:
        if not RAW_DATA_DIR.exists():
            RAW_DATA_DIR.mkdir(parents=True)

        # Check if the dataset is already downloaded
        if (RAW_DATA_DIR / 'fer2013.csv').exists():
            print("Dataset already exists. Skipping download.")
            return

        # Download the dataset using Kaggle API
        print(f'Downloading dataset {DATASET_NAME} from Kaggle...')
        kaggle.api.dataset_download_files(DATASET_NAME, path=RAW_DATA_DIR, unzip=True)
        print('Dataset downloaded and saved to data/raw/fer2013.csv')

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")

if __name__ == '__main__':
    fetch_data()