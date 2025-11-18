# download_data.py

import os
import json
import urllib.request
import zipfile

# 1. URLs dữ liệu
train_data_url = "https://dl-challenge.zalo.ai/2025/Drone/observing.zip"
public_test_data_url = "https://dl-challenge.zalo.ai/2025/Drone/public_test.zip"

# 2. Thư mục lưu trữ
# Sử dụng đường dẫn local thay vì Google Drive
base_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(base_dir, exist_ok=True)

train_zip_path = os.path.join(base_dir, "observing.zip")
public_test_zip_path = os.path.join(base_dir, "public_test.zip")

train_unzip_dir = os.path.join(base_dir, "observing_unzipped")
public_test_unzip_dir = os.path.join(base_dir, "public_test_unzipped")

os.makedirs(train_unzip_dir, exist_ok=True)
os.makedirs(public_test_unzip_dir, exist_ok=True)

# 3. Download dữ liệu (Python thuần, thay vì !wget)
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    else:
        print(f"File {save_path} already exists, skipping download.")

download_file(train_data_url, train_zip_path)
download_file(public_test_data_url, public_test_zip_path)

# 4. Giải nén file ZIP
def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

unzip_file(train_zip_path, train_unzip_dir)
unzip_file(public_test_zip_path, public_test_unzip_dir)

# 5. Đọc file JSON annotations
annotations_path = os.path.join(train_unzip_dir, "train/annotations/annotations.json")
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# 6. Test in ra 1 entry đầu tiên
print(annotations[0])
