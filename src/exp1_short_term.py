import pandas as pd
import numpy as np
import json
import random
from openai import OpenAI
import os
import re

# Set random seed
random.seed(42)
np.random.seed(42)

# Load ETTm1
print("Loading ETTm1 dataset...")
df = pd.read_csv("datasets/ett/ETTm1.csv")

# We'll focus on 'OT' (Oil Temperature)
target_col = 'OT'
data = df[target_col].values

# Parameters
window_size = 96  # 96 hours (4 days of hourly data)
pred_size = 24    # predict next 24 hours (1 day)
num_samples = 10  # number of samples to test

# Generate random samples
samples = []
for _ in range(num_samples):
    start_idx = random.randint(0, len(data) - window_size - pred_size - 1)
    x = data[start_idx : start_idx + window_size]
    y_true = data[start_idx + window_size : start_idx + window_size + pred_size]
    samples.append((x, y_true))

client = OpenAI()

def parse_predictions(response_text, expected_len):
    # Try to find a list of numbers
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
    # Convert to float
    numbers = [float(n) for n in numbers]
    if len(numbers) >= expected_len:
        return np.array(numbers[-expected_len:])
    else:
        # Fallback if not enough numbers
        if len(numbers) > 0:
            return np.pad(numbers, (0, expected_len - len(numbers)), mode='edge')
        else:
            return np.zeros(expected_len)

print("Running LLM baseline for short-term numerical forecasting...")
results = []
for i, (x, y_true) in enumerate(samples):
    x_str = ", ".join([f"{val:.3f}" for val in x])
    
    prompt = f"""You are an expert time-series forecasting model.
I will provide you with a sequence of {window_size} hourly numerical data points (Oil Temperature from a transformer).
Your task is to predict the next {pred_size} hourly values.

Input sequence:
{x_str}

Please output ONLY a comma-separated list of the {pred_size} predicted numerical values, and nothing else.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        pred_text = response.choices[0].message.content
        y_pred_llm = parse_predictions(pred_text, pred_size)
    except Exception as e:
        print(f"Error on sample {i}: {e}")
        y_pred_llm = np.zeros(pred_size)
        
    # Baseline 1: Naive (repeat last value)
    y_pred_naive = np.full(pred_size, x[-1])
    
    # Baseline 2: Moving Average (last 24 values)
    ma_val = np.mean(x[-24:])
    y_pred_ma = np.full(pred_size, ma_val)
    
    # Metrics
    mse_llm = np.mean((y_true - y_pred_llm)**2)
    mse_naive = np.mean((y_true - y_pred_naive)**2)
    mse_ma = np.mean((y_true - y_pred_ma)**2)
    
    results.append({
        "sample": i,
        "mse_llm": mse_llm,
        "mse_naive": mse_naive,
        "mse_ma": mse_ma
    })
    print(f"Sample {i}: LLM MSE={mse_llm:.4f}, Naive MSE={mse_naive:.4f}, MA MSE={mse_ma:.4f}")

df_res = pd.DataFrame(results)
print("\n--- Summary for Short-Term High-Data (ETTm1) ---")
print(df_res.mean())

os.makedirs("results", exist_ok=True)
df_res.to_csv("results/exp1_ettm1_results.csv", index=False)
print("Saved exp1 results.")
