from dotenv import load_dotenv
import os

load_dotenv()
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi # kaggle reads env variable on time of module import so make sure to use "load_dotenv()" before this.

print("KAGGLE_USERNAME =", os.getenv("KAGGLE_USERNAME"))
print("KAGGLE_KEY =", os.getenv("KAGGLE_KEY"))

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME").strip()
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY").strip()


def download_data(download_path, api) ->None:

    os.makedirs(download_path, exist_ok=True)

    api.competition_download_files("playground-series-s5e4",
                                path = download_path)

    print(f" Download to : {download_path}")


def unzip(download_path) ->None:
    zip_file:str = os.path.join(download_path, "playground-series-s5e4.zip")
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Unzipped successfully.")

        os.remove(zip_file)
        print("ZIP file deleted.")
    else:
        print("ZIP file not found.")



if __name__ == "__main__":

    api = KaggleApi()
    api.authenticate()

    download_path = os.path.join(os.getcwd(), "data", "raw")
    download_data(download_path, api)
    unzip(download_path)