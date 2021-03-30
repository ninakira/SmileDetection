import zipfile
import os
from pathlib import Path

zip_path = "/data/celeba.zip"


def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        path = Path(zip_path)
        name = os.path.basename(path).split('.')[0]
        zip_ref.extractall(str(path.parent) + "/unzipped_" + name)


extract_zip(zip_path)
