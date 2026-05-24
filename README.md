# LLMs vs Humans in Forecasting

## Overview
This repository contains the code, data, and findings from an empirical evaluation of Large Language Models (LLMs) against human forecasters. The study tests the hypothesis that LLMs outperform humans in two distinct extremes: short-term high-data scenarios (numerical time series) and long-term low-data scenarios (event-based reasoning).

## Key Findings
- **LLMs vs. Superforecasters:** On the ForecastBench long-term geopolitical and global event questions, the LLM (`gpt-4o-mini`) achieved a Brier score of 0.246, surpassing the human superforecaster baseline (0.287) and matching the public crowd (0.243).
- **High-data Capability:** On short-term high-data arrays (ETTm1 dataset), the LLM demonstrated zero-shot capabilities to process and project 96 numerical data points, a volume humans cannot manually process. It achieved an MSE of 3.988, remaining competitive with algorithmic moving averages.
- **Unified Forecasters:** The results suggest LLMs can act as unified forecasting tools capable of both high-frequency numerical prediction and low-frequency qualitative abstract reasoning.

## Full Report
Please see [REPORT.md](REPORT.md) for the complete methodology, empirical results, and discussion of limitations.

## How to Reproduce
1. Install `uv` and create the environment:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install numpy pandas matplotlib scikit-learn openai tenacity requests scipy
   ```
2. Download ETTm1 dataset:
   ```bash
   python src/download_ettm1.py
   ```
3. Run the short-term numerical experiment (ensure `OPENAI_API_KEY` is set):
   ```bash
   python src/exp1_short_term.py
   ```
4. Run the long-term event-based experiment:
   ```bash
   python src/exp2_long_term.py
   ```
5. View generated results in the `results/` folder.
