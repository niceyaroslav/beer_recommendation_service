import os
import zipfile
from pathlib import Path
import subprocess
from dotenv import load_dotenv

load_dotenv()


def download_kaggle_dataset_if_missing(
    dataset="ruthgn/beer-profile-and-ratings-data-set",
    data_dir="data"
):
    data_path = Path(data_dir)
    csv_file = data_path / "beer_profile_and_ratings.csv"

    if csv_file.exists():
        print("Dataset already exists.")
        return

    print("Dataset not found. Downloading from Kaggle...")

    # Load credentials
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise ValueError(
            "Kaggle credentials not found in .env file.\n"
            "Please set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    # Set env variables for subprocess
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = kaggle_username
    env["KAGGLE_KEY"] = kaggle_key

    data_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(data_path)
        ],
        check=True,
        env=env
    )

    # Extract
    for zip_file in data_path.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(data_path)
        zip_file.unlink()

    print("Download complete.")