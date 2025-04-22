import os
import urllib.request
from pathlib import Path

def download_dataset(url, save_path):
    os.makedirs(save_path, exist_ok=True)
    filename = url.split("/")[-1]
    full_path = Path(save_path) / filename
    print(f"Downloading {filename} to {full_path}...")
    urllib.request.urlretrieve(url, full_path)
    print("Done.")


if __name__ == "__main__":
    url = "temp/emoji_dataset_128x128.zip"
    save_path = "data/"
    download_dataset(url, save_path)
