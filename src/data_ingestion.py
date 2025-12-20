from dotenv import load_dotenv
import os

load_dotenv()
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi # kaggle reads env variable on time of module import so make sure to use "load_dotenv()" before this.

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
     
    # DELETING THESE FILES BECAUSE THEY ARE USEFUL IN CASE OF KAGGLE COMPETITION SUBMISSION NOT FOR THIS TASK. 
    test_data_path = os.path.join(download_path,"test.csv")
    sample_submission_data_path = os.path.join(download_path,"sample_submission.csv")

    for file_path in [test_data_path, sample_submission_data_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"{file_path} deleted successfully")
        else:
            print(f"{file_path} not found")

