import json
import pandas as pd
import numpy as np
import os
import random
from openai import OpenAI
import re

random.seed(42)

# Load questions
with open('datasets/forecastbench/datasets/question_sets/2024-07-21-llm.json') as f:
    questions_data = json.load(f)
questions = {}
for q in questions_data['questions']:
    qid = q['id']
    if isinstance(qid, list):
        qid = qid[0]
    questions[qid] = q

# Load resolutions
with open('datasets/forecastbench/datasets/resolution_sets/2024-07-21_resolution_set.json') as f:
    res_data = json.load(f)
resolutions = {}
for r in res_data['resolutions']:
    if r['resolved']:
        rid = r['id']
        if isinstance(rid, list):
            rid = rid[0]
        resolutions[rid] = r['resolved_to']

# Match questions with resolutions
valid_ids = [qid for qid in questions.keys() if qid in resolutions]
print(f"Found {len(valid_ids)} valid questions with resolutions out of {len(questions)}")

# Sub-sample to save time and API costs (e.g., 20 questions)
if len(valid_ids) > 20:
    valid_ids = random.sample(valid_ids, 20)
print(f"Using {len(valid_ids)} questions for experiment.")

client = OpenAI()

def parse_probability(text):
    # Try to find a percentage or decimal
    matches = re.findall(r"0\.\d+|\b[1-9]?\d(?:\.\d+)?%", text)
    if matches:
        val = matches[-1]
        if '%' in val:
            return float(val.replace('%', '')) / 100.0
        return float(val)
    # Default fallback
    return 0.5

print("Running LLM on event-based forecasting...")
llm_forecasts = {}
for qid in valid_ids:
    q = questions[qid]
    prompt = f"""You are an expert superforecaster analyzing global events.
Question: {q['question']}
Background context: {q['background']}
As of {questions_data['forecast_due_date']}, what is the probability (from 0.0 to 1.0) that this resolves as Yes?

Please provide your reasoning, and then at the very end, state your final probability clearly as a decimal (e.g., 0.75).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        res_text = response.choices[0].message.content
        prob = parse_probability(res_text)
        llm_forecasts[qid] = prob
        print(f"Q: {q['question'][:50]}... | Prob: {prob} | Actual: {resolutions[qid]}")
    except Exception as e:
        print(f"Error for {qid}: {e}")
        llm_forecasts[qid] = 0.5

# Load Human Super-forecasters
human_file = 'datasets/forecastbench/datasets/forecast_sets/2024-07-21/2024-07-21.ForecastBench.human_super_individual.json'
with open(human_file) as f:
    human_data = json.load(f)

# Aggregate human forecasts (average per question)
human_preds = {}
for fcast in human_data['forecasts']:
    qid = fcast['id']
    if isinstance(qid, list):
        qid = qid[0]
    if qid not in human_preds:
        human_preds[qid] = []
    # If the forecast is a list of proba distributions or single value
    try:
        if isinstance(fcast['forecast'], dict) and 'Yes' in fcast['forecast']:
            human_preds[qid].append(fcast['forecast']['Yes'])
        elif isinstance(fcast['forecast'], float):
            human_preds[qid].append(fcast['forecast'])
    except Exception:
        pass

# Average them
human_agg = {qid: np.mean(preds) for qid, preds in human_preds.items() if len(preds) > 0}

# Load Human Public
public_file = 'datasets/forecastbench/datasets/forecast_sets/2024-07-21/2024-07-21.ForecastBench.human_public_individual.json'
with open(public_file) as f:
    public_data = json.load(f)
public_preds = {}
for fcast in public_data['forecasts']:
    qid = fcast['id']
    if isinstance(qid, list):
        qid = qid[0]
    if qid not in public_preds:
        public_preds[qid] = []
    try:
        if isinstance(fcast['forecast'], dict) and 'Yes' in fcast['forecast']:
            public_preds[qid].append(fcast['forecast']['Yes'])
        elif isinstance(fcast['forecast'], float):
            public_preds[qid].append(fcast['forecast'])
    except Exception:
        pass
public_agg = {qid: np.mean(preds) for qid, preds in public_preds.items() if len(preds) > 0}

# Calculate Brier Scores
results = []
for qid in valid_ids:
    actual = resolutions[qid]
    llm_pred = llm_forecasts.get(qid, 0.5)
    
    # We use 0.5 if humans didn't predict
    super_pred = human_agg.get(qid, 0.5)
    public_pred = public_agg.get(qid, 0.5)
    
    brier_llm = (llm_pred - actual) ** 2
    brier_super = (super_pred - actual) ** 2
    brier_public = (public_pred - actual) ** 2
    
    results.append({
        'question_id': qid,
        'actual': actual,
        'llm_pred': llm_pred,
        'super_pred': super_pred,
        'public_pred': public_pred,
        'brier_llm': brier_llm,
        'brier_super': brier_super,
        'brier_public': brier_public
    })

df_res = pd.DataFrame(results)
print("\n--- Summary for Long-Term Low-Data (ForecastBench) ---")
print(df_res[['brier_llm', 'brier_super', 'brier_public']].mean())

os.makedirs("results", exist_ok=True)
df_res.to_csv("results/exp2_forecastbench_results.csv", index=False)
print("Saved exp2 results.")
