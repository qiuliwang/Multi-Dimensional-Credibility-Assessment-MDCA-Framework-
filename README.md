# Multi-Dimensional Credibility Assessment (MDCA) Framework

A framework for evaluating the credibility of AI-generated clinical MRI liver reports against original radiology reports.

**Author:** Q. Wang, Department of Radiology, Southwest Hospital, Army Medical University (Third Military Medical University), Chongqing, China

## Overview

The MDCA framework scores machine-generated liver MRI reports along three dimensions:

1. **Semantic similarity** — sentence-level cosine similarity between generated and original diagnoses, computed with a Chinese sentence-embedding model (`text2vec-base-chinese`). Malignant/benign label conflicts between matched diagnoses heavily penalize the score.
2. **Semantic authenticity** — keyword-driven verification that TOP-priority diagnoses (e.g., hepatocellular carcinoma) in the original report are correctly reflected in the generated report.
3. **Priority coverage** — weighted TOP1/TOP3/TOP5 keyword matching, where higher-priority diagnoses carry larger weight.

The final credibility score is a weighted combination:

```
score = 0.2 × authenticity + 0.4 × semantic_similarity + 0.4 × priority_coverage
```

## Dataset

The framework was evaluated on the `report_2nd_non-postoperative` batch of clinical MRI liver reports (**Batch2: 15,127 reports**). See `Liver/Readme.md` for details.

## Pipeline

```
原始报告 ──Check_Meta.py──▶ 预处理（检查所见/结论拆分）
         ──Get_trainData.py──▶ 训练/推理数据集
         ──{model}_P6.py──▶  LLM 生成报告（SiliconFlow API）
         ──Scoring.py──▶      评分结果 CSV（三维度 + 综合分）
```

- **Data preparation** (`Liver/Check_Meta.py`, `Liver/Get_trainData.py`, `Liver/CountLength.py`) — validate data integrity, split reports into findings/conclusions, and build the input dataset.
- **Report generation** (`Liver/` — one script per LLM) — batch inference through the SiliconFlow API with multi-threaded workers, retries, and progress tracking. Six models were evaluated:
  - DeepSeek-V3 (`V3_P6.py`)
  - DeepSeek-V3.1-Terminus (`V31_P6_7500.py`)
  - DeepSeek-R1 (`VR1_P6_7500.py`)
  - Kimi-K2 (`Kimi-K2_P6.py`)
  - Qwen3-235B-A22B (`Qwen3-235B_P6_7500.py`)
  - ByteDance Seed-OSS-36B (`ByteDance-Seed_P6_7500.py`)
  - Prompt variants P0–P11 are in `Liver/Prompt_Text_New/` (role setting, task requirements, TOP-graded diagnosis system, verification items, report structure, imaging diagnosis rules, and 0–25 few-shot examples).
- **Scoring** (`Liver/ReportScorer.py` — main version; `Liver/ReportScorerPlus.py` — optimized refactor with model caching, thread safety, GPU support and batched encoding; `Liver/Scoring.py` — batch scoring driver with multithreaded reading and multiprocess scoring).

## Requirements

- Python 3.9+
- `sentence-transformers`, `torch`, `pandas`, `tqdm`
- A SiliconFlow API key (set as the `SILICONFLOW_KEY` environment variable — API keys are not stored in this repository)

## Usage

```bash
export SILICONFLOW_KEY=<your-key>

# 1. Validate and preprocess the original reports
python Liver/Check_Meta.py
python Liver/Get_trainData.py

# 2. Generate reports with an LLM (example: DeepSeek-V3, prompt P6)
python Liver/V3_P6.py

# 3. Score the generated reports
python Liver/Scoring.py
```

Detailed file-structure documentation: `Liver/Readme.md`.
