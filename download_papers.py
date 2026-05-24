import arxiv
import requests
import os

papers_to_find = [
    "A Novel Distributed PV Power Forecasting Approach Based on Time-LLM",
    "Evaluating LLMs on Real-World Forecasting Against Expert Forecasters",
    "Pitfalls in Evaluating Language Model Forecasters",
    "ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities",
    "MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering",
    "How AI Forecasts AI Jobs: Benchmarking LLM Predictions of Labor Market Changes",
    "Logo-LLM: Local and Global Modeling with Large Language Models for Time Series Forecasting",
    "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
    "One Fits All: Power General Time Series Analysis by Pretrained LM"
]

client = arxiv.Client()

if not os.path.exists("papers"):
    os.makedirs("papers")

for title in papers_to_find:
    print(f"Searching for: {title}")
    search = arxiv.Search(
        query=f'ti:"{title}"',
        max_results=1
    )
    
    try:
        results = list(client.results(search))
        if not results:
             print(f"Exact title match failed for {title}, trying broad search...")
             search = arxiv.Search(query=title, max_results=1)
             results = list(client.results(search))
             
        if results:
            paper = results[0]
            filename = paper.entry_id.split("/")[-1] + "_" + title.replace(" ", "_").replace(":", "").replace("?", "").replace("/", "_") + ".pdf"
            filepath = os.path.join("papers", filename)
            print(f"Downloading {paper.title} from {paper.pdf_url}")
            
            response = requests.get(paper.pdf_url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filepath}")
        else:
            print(f"Could not find paper: {title}")
    except Exception as e:
        print(f"Error downloading {title}: {e}")

