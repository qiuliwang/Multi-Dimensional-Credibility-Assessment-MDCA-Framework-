# 共有 1 个批次：

## 批次：report_2nd_non-postoperative（1 种类）
    - Batch2: 15127 份报告
###  批次 report_2nd_non-postoperative 合计：15127 份报告

1. 运行Check_Meta.py，检查数据完整性并统计
2. 允许Get_trainData.py，获取训练用的数据

prompt5是与prompt2比较，对比样本量对于prompt的影响

# 验证过程 2025 10 13
批次：report_2nd_non-postoperative（1 种类）
    - Batch2: 15127 份报告
  批次 report_2nd_non-postoperative 合计：15127 份报告

## 实验
P0 角色设定
P1 角色设定+基本任务要求
P2 角色设定+基本任务要求+样例*3
P3 角色设定+基本任务要求+TOP分级诊断体系+报告结构规范+样例*3
P4 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*3
P5 角色设定+基本任务要求+样例*10
P6 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*10
P7 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律
P8 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*5
P9 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*15
P10 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*20
P11 角色设定+基本任务要求+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例*25


7-4-8-6-9-10-11

---

# 文件结构说明（供审阅者参考）

## 数据准备
- `Check_Meta.py`：检查数据完整性并统计报告数量
- `Get_trainData.py`：生成训练/推理用数据
- `CountLength.py`：统计报告文本长度分布
- `drawtable.py`：生成表格（供论文使用）

## 报告生成（批处理推理）
每个模型各保留 1 个示例脚本（P6 提示词配置，即角色设定+TOP分级诊断体系+强制核查项+报告结构规范+影像诊断规律+样例×10）。运行前需设置环境变量 `SILICONFLOW_KEY`（API 密钥不入库）：

| 脚本 | 模型 |
|---|---|
| `V3_P6.py` | `Pro/deepseek-ai/DeepSeek-V3` |
| `V31_P6_7500.py` | `Pro/deepseek-ai/DeepSeek-V3.1-Terminus` |
| `VR1_P6_7500.py` | `Pro/deepseek-ai/DeepSeek-R1` |
| `Kimi-K2_P6.py` | `Pro/moonshotai/Kimi-K2-Instruct-0905` |
| `Qwen3-235B_P6_7500.py` | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| `ByteDance-Seed_P6_7500.py` | `ByteDance-Seed/Seed-OSS-36B-Instruct` |

- `Prompt_Text_New/prompt0~11.txt`：全部 12 个提示词变体（P0~P11），对应 `Readme.md` 中"实验"一节的配置

## 评分
- `ReportScorer.py`：主版本评分器（定义 `ReportScorerPlus` 类），`Scoring.py` 调用它
- `ReportScorerPlus.py`：评分器的优化重构版（模型缓存、线程安全、GPU、批量向量化），接口一致，可无缝替换
- `Scoring.py`：批量评分主流程（多线程读取 + 多进程评分），输出各模型各提示词的评分 CSV