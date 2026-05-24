## Motivation & Novelty Assessment

### Why This Research Matters
Forecasting is a critical capability for decision-making across domains, from energy management to geopolitics. Understanding whether LLMs outperform humans in specific forecasting scenarios (short-term high-data vs. long-term low-data) helps determine where to deploy AI versus human experts.

### Gap in Existing Work
Existing work typically isolates numerical time-series forecasting (high-data) from event-based reasoning forecasting (low-data). There is a lack of unified evaluation that directly contrasts LLM capabilities against human baselines across these two distinct extremes of the forecasting spectrum.

### Our Novel Contribution
We conduct a unified evaluation assessing LLM performance on both short-term high-data scenarios (numerical time series) and long-term low-data scenarios (event-based questions). By synthesizing these results, we establish the boundaries of LLM superiority relative to human forecasters.

### Experiment Justification
- **Experiment 1 (Short-term High-data):** Tests LLM zero-shot and few-shot capabilities on numerical time series (e.g., ETTm1) compared to simple baselines, demonstrating data-processing volume advantages over human cognitive limits.
- **Experiment 2 (Long-term Low-data):** Tests LLMs on ForecastBench, comparing their event-based probabilistic forecasts against human crowds and expert "superforecasters" to determine if LLMs truly exceed human performance in low-data regimes.

## Proposed Methodology

### Approach
We will split the research into two streams:
1. **Numerical Time Series (ETTm1)**: We will evaluate LLM predictions (via Time-LLM or direct zero-shot prompting) on the ETTm1 dataset to represent short-term high-data forecasting.
2. **Event-based Forecasting (ForecastBench)**: We will evaluate LLM forecasts on a subset of ForecastBench questions, comparing the Brier scores of the LLM against human superforecasters and the public crowd.

### Expected Outcomes
We expect LLMs to achieve very low error on short-term high-data (where humans cannot process the sheer volume) and to either match or exceed human superforecasters on long-term low-data.
