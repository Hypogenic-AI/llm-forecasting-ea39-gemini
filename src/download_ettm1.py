import pandas as pd
import os
import urllib.request
import zipfile

# Download ETTm1 directly from a reliable source if HF fails
os.makedirs("datasets/ett", exist_ok=True)
url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
output_path = "datasets/ett/ETTm1.csv"

if not os.path.exists(output_path):
    print(f"Downloading ETTm1 from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
else:
    print("ETTm1 already exists.")
