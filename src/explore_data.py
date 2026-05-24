import json
import glob
import os

print("ForecastBench Questions:")
qfile = 'datasets/forecastbench/datasets/question_sets/2024-07-21-llm.json'
with open(qfile) as f:
    q = json.load(f)
    print("Keys:", list(q.keys()))
    first_k = list(q.keys())[0]
    print("First item:", json.dumps(q[first_k], indent=2))

hfile = 'datasets/forecastbench/datasets/forecast_sets/2024-07-21/2024-07-21.ForecastBench.human_super_individual.json'
with open(hfile) as f:
    h = json.load(f)
    print("Human Keys:", list(h.keys()))
    print("First human forecast:", json.dumps(h[list(h.keys())[0]], indent=2))
