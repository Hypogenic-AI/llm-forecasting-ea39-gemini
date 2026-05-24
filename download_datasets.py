from datasets import load_dataset
import os
import pandas as pd

if not os.path.exists("datasets"):
    os.makedirs("datasets")

# 1. ForecastBench (Long-term Low-data)
print("Downloading ForecastBench datasets...")
try:
    # ForecastBench has several subsets, let's try the main ones
    # Based on GitHub, subsets might be 'questions', 'forecasts', 'human_forecasts'
    # I'll try to load 'forecastbench-datasets' directly
    ds = load_dataset("forecastingresearch/forecastbench-datasets")
    for split in ds.keys():
        df = ds[split].to_pandas()
        csv_path = f"datasets/forecastbench_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} to {csv_path} (Size: {len(df)})")
except Exception as e:
    print(f"Error downloading ForecastBench: {e}")

# 2. ETT (Short-term High-data)
print("Downloading ETT datasets...")
try:
    # ETT is often used in TS papers. I'll use ETTm1 (minutes level)
    # dataset = load_dataset("thuml/ETT", "ETTm1") # Path might be different
    # Alternative: use a more common repo
    ds = load_dataset("all-about-time-series/ett", "ETTm1")
    for split in ds.keys():
        df = ds[split].to_pandas()
        csv_path = f"datasets/ettm1_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split} to {csv_path} (Size: {len(df)})")
except Exception as e:
    print(f"Error downloading ETT: {e}")

