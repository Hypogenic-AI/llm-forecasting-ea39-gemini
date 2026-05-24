from datasets import load_dataset
import os
import pandas as pd

if not os.path.exists("datasets/ts"):
    os.makedirs("datasets/ts")

try:
    ds = load_dataset("ts-arena/ettm1")
    for split in ds.keys():
        df = ds[split].to_pandas()
        csv_path = f"datasets/ts/ettm1_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} to {csv_path}")
except Exception as e:
    print(f"Error: {e}")
