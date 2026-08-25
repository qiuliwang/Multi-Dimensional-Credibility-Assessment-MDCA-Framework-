# Multi-Dimensional Quality Assessment (MDQA) Framework

A framework for evaluating the quality of LLM-generated diagnostic impressions from clinical MRI liver reports.

> **术语说明 / Terminology note:** 本仓库早期名称及早期代码使用 "Multi-Dimensional Credibility Assessment (MDCA)" 的表述；根据同行评审意见，仓库与论文均已更名为 **Multi-Dimensional Quality Assessment (MDQA)**。由于代码在开发过程中经过多次修改，若代码与论文表述存在任何差异，**最终以论文的描述为准**。 / This repository (formerly named "Multi-Dimensional-Credibility-Assessment-MDCA-Framework") and its early code use the former name "Multi-Dimensional Credibility Assessment (MDCA)"; following peer review, both the repository and the manuscript now use **Multi-Dimensional Quality Assessment (MDQA)**. The code has undergone multiple revisions during development — in case of any discrepancy, **the published manuscript is authoritative**.

**Authors:** Qiuli Wang¹,²†, Xinhuan Sun¹†, Yonglin Chen²† († contributed equally)

¹ Yu-Yue Pathology Research Center, Jinfeng Laboratory, Chongqing, China
² 7T Magnetic Resonance Imaging Translational Medical Center, Department of Radiology, Southwest Hospital, Army Medical University, Chongqing, China

## Overview

The MDQA framework evaluates LLM-generated diagnostic impressions along three dimensions:

1. **Semantic Coherence (SC)** — sentence-level cosine similarity between generated and reference impressions, computed with a Chinese sentence-embedding model (`text2vec-base-chinese`). Malignant/benign label conflicts between matched diagnoses heavily penalize the score.
2. **Diagnostic Correctness (DC)** — keyword-driven verification that TOP-priority diagnoses (e.g., hepatocellular carcinoma) in the reference report are correctly reflected in the generated impression.
3. **Clinical Prioritization Alignment (CPA)** — weighted TOP1/TOP3/TOP5 keyword matching, where higher-priority diagnoses carry larger weight.

The composite score is:

```
MDQA = 0.4 × SC + 0.2 × DC + 0.4 × CPA
```

Top-1 matching is additionally reported as a separate metric.

**Task scope:** The models process only the textual findings of radiology reports (text-to-text impression generation) and do not directly interpret MRI images. The reference standard is an expert-confirmed report-level textual reference; MDQA measures agreement and report-level quality rather than patient-level diagnostic truth.

## Dataset

The framework was evaluated on the `report_2nd_non-postoperative` batch of clinical MRI liver reports (**Batch2: 15,127 reports**). See `Liver/Readme.md` for details.

## Pipeline

```
原始报告 ──Check_Meta.py──▶ 预处理（检查所见/结论拆分）
         ──Get_trainData.py──▶ 训练/推理数据集
         ──{model}_P6.py──▶  LLM 生成报告（SiliconFlow API）
         ──Scoring.py──▶      评分结果 CSV（SC / DC / CPA / Top-1 / MDQA）
```

- **Data preparation** (`Liver/Check_Meta.py`, `Liver/Get_trainData.py`, `Liver/CountLength.py`) — validate data integrity, split reports into findings/conclusions, and build the input dataset.
- **Report generation** (`Liver/` — one script per LLM) — batch inference through the SiliconFlow API with multi-threaded workers, retries, and progress tracking. The models were evaluated:
  - DeepSeek-V3 (`V3_P6.py`)
  - DeepSeek-V3.1-Terminus (`V31_P6_7500.py`)
  - DeepSeek-R1 (`VR1_P6_7500.py`)
  - Kimi-K2 (`Kimi-K2_P6.py`)
  - Qwen3-235B-A22B (`Qwen3-235B_P6_7500.py`)
  - ByteDance Seed-OSS-36B (`ByteDance-Seed_P6_7500.py`)
  - Prompt variants P0–P11 are in `Liver/Prompt_Text_New/` (role setting, task requirements, TOP-graded diagnosis system, verification items, report structure, imaging diagnosis rules, and 0–25 few-shot examples).
- **Scoring** (`Liver/ReportScorer.py` — main version; `Liver/ReportScorerPlus.py` — optimized refactor with model caching, thread safety, GPU support and batched encoding; `Liver/Scoring.py` — batch scoring driver with multithreaded reading and multiprocess scoring). The score components correspond to the paper's SC / DC / CPA / Top-1 matching / MDQA (see `Liver/Readme.md` for the mapping).

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
