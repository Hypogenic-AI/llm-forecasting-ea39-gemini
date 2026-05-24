# Literature Review: LLMs and Forecasting

## Research Area Overview
Large Language Models (LLMs) have shown significant promise in forecasting, spanning two distinct domains: numerical time series forecasting (high-data, pattern-based) and real-world event forecasting (low-data, heuristic-based).

## Key Findings

### 1. Numerical Time Series Forecasting (Short-term High-data)
- **Time-LLM (Jin et al., 2024)** and **One-Fits-All (Zhou et al., 2023)** demonstrate that frozen LLMs can be "reprogrammed" to handle numerical data.
- LLMs excel in **zero-shot and few-shot** scenarios, often outperforming specialized models like Informer or Autoformer.
- **Logo-LLM (Ou et al., 2025)** improves this by modeling both local (short-term) and global (long-term) dependencies.
- **Superiority**: LLMs likely outperform humans here simply due to the volume of data and the need for high-frequency pattern recognition, which humans are not optimized for.

### 2. Event-based Forecasting (Long-term Low-data)
- **ForecastBench (Karger et al., 2025)** and **Janna Lu (2025)** provide the most direct comparison to human forecasters.
- **The "Crowd" vs. Experts**: Frontier models (GPT-4o, Claude 3.6, Deepseek R1) currently **surpass the human crowd** (average Metaculus users) but **underperform expert forecasters** (superforecasters).
- **Pitfalls (Paleka et al., 2025)**: Issues like temporal leakage (training data contains future info) and overconfidence are significant hurdles for LLM forecasters.
- **Scaffolding**: LLMs perform better when augmented with search (RAG) and prompted as "superforecasters".

## Methodology Comparison

| Feature | Numerical (High-data) | Event-based (Low-data) |
|---------|-----------------------|-------------------------|
| Input | Patched time series | News snippets, context |
| LLM Role | Pattern recognizer | Reasoning engine |
| Human Baseline | Statistical models | Experts/Crowds |
| LLM Edge | Data processing volume | Breadth of knowledge |

## Evaluation Metrics
- **Numerical**: MSE (Mean Squared Error), MAE (Mean Absolute Error).
- **Event-based**: Brier Score (lower is better), calibration curves.

## Gaps and Opportunities
- **Expert Gap**: Closing the performance gap between LLMs and superforecasters.
- **Hybrid Models**: Combining numerical time series with qualitative news reasoning.
- **Long-term Calibration**: Improving LLM performance on multi-year horizons where historical patterns may change.

## Recommendations for Experiment
1. **Test Short-term High-data**: Use ETTm1 dataset with Time-LLM or OFM baselines.
2. **Test Long-term Low-data**: Use ForecastBench dataset to compare LLM forecasts against human expert benchmarks.
3. **Verify the Hypothesis**: Specifically test if LLM performance relative to humans improves as the data volume increases (short-term) or as the data becomes more qualitative/scarce (long-term).
