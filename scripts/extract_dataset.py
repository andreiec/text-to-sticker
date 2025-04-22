import zipfile


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


if __name__ == "__main__":
    zip_path = "data/emoji_dataset_128x128.zip"
    extract_to = "data/emoji_dataset_128x128/"
    extract_zip(zip_path, extract_to)
