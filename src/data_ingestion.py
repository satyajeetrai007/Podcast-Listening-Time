import os
import zipfile
import json
import logging
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["KAGGLE_USERNAME"] = (os.getenv("KAGGLE_USERNAME") or "").strip()
os.environ["KAGGLE_KEY"] = (os.getenv("KAGGLE_KEY") or "").strip()

def download_data(download_path: str, api: KaggleApi) -> None:
    """Downloads competition files from Kaggle."""
    try:
        os.makedirs(download_path, exist_ok=True)
        api.competition_download_files("playground-series-s5e4", path=download_path)
        logger.info(f"Download to: {download_path}")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise

def unzip(download_path: str) -> None:
    """Extracts the downloaded zip file."""
    zip_file: str = os.path.join(download_path, "playground-series-s5e4.zip")
    try:
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            logger.info("Unzipped successfully.")
            os.remove(zip_file)
            logger.info("ZIP file deleted.")
        else:
            logger.warning(f"ZIP file not found: {zip_file}")
    except zipfile.BadZipFile as e:
        logger.error(f"Corrupted zip file: {e}")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")

def save_categories(raw_data_path: str, output_path: str) -> None:
    """Extracts unique categories for UI components."""
    try:
        csv_path: str = os.path.join(raw_data_path, "train.csv")
        df: pd.DataFrame = pd.read_csv(csv_path)
        
        categories: dict[str, list[Any]] = {
            "genres": sorted(df['Genre'].dropna().unique().tolist()),
            "days": sorted(df['Publication_Day'].dropna().unique().tolist()),
            "podcasts": sorted(df['Podcast_Name'].dropna().unique().tolist()),
            "titles": sorted(df['Episode_Title'].dropna().unique().tolist())
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(categories, f, indent=4)
        logger.info(f"Categories saved to: {output_path}")
    except FileNotFoundError:
        logger.error(f"train.csv not found at {raw_data_path}")
    except Exception as e:
        logger.error(f"Failed to save categories: {e}")

if __name__ == "__main__":
    try:
        api: KaggleApi = KaggleApi()
        api.authenticate()

        download_dir: str = os.path.join(os.getcwd(), "data", "raw")
        download_data(download_dir, api)
        unzip(download_dir)

        cat_file: str = os.path.join(os.getcwd(), "model", "categories.json")
        save_categories(download_dir, cat_file)
        
        test_data_path: str = os.path.join(download_dir, "test.csv")
        sample_sub_path: str = os.path.join(download_dir, "sample_submission.csv")

        for file_path in [test_data_path, sample_sub_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"{file_path} deleted successfully")
            else:
                logger.info(f"{file_path} not found")
    except Exception as e:
        logger.critical(f"Script failed: {e}")