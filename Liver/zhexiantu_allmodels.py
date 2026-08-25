# -*- coding: utf-8 -*-
"""
多图版本（带别名映射修复）：
- 每个指标单独一张图（共5张）
- 横轴：Prompt2, Prompt4, Prompt6
- 综合评分 ÷100
- 中文指标改为英文
- 保留 Δ 注释、不画“幅度箭头”
- 纵坐标下限固定为 0.5
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ==== 字体 ====
rcParams["axes.unicode_minus"] = False
for f in ["SimHei", "Microsoft YaHei", "PingFang SC", "Source Han Sans SC"]:
    if f in [x.name for x in font_manager.fontManager.ttflist]:
        rcParams["font.family"] = f
        break

# ==== 配置 ====
file_path = "模型对比P6_7327-mean_table_common.csv"

# 展示名（图例里显示的名字）
MODELS = ["ByteDance-Seed", "KIMI-K2", "Qwen3", "DeepSeek-V3"]

# 用来在列名中匹配的别名（不区分大小写，命中任一即可）
MODEL_ALIASES = {
    "ByteDance-Seed": ["bytedance"],
    "KIMI-K2": ["kimi"],
    "Qwen3": ["qwen"],
    "DeepSeek-V3": ["v3", "deepseek"],
}

PROMPTS = ["Prompt2", "Prompt4", "Prompt6"]  # 横轴显示
PROMPT_RE = re.compile(r"prompt(\d+)", re.IGNORECASE)  # 从列名里抽取 promptN

# 中文 → 英文
METRIC_MAP = {
    "语义相似度(BERT)": "Semantic Coherence",
    "真实性得分(语义)": "Diagnostic Correctness",
    "Top1匹配": "Top-1 Matching Scores",
    "优先级得分": "Clinical Prioritization Alignment",
    "综合评分": "MDCA Score",
}
ALL_METRICS = list(METRIC_MAP.keys())
METRIC_COMP = "综合评分"

# ==== 读取 ====
df = pd.read_csv(file_path)
df.columns = [str(c).strip() for c in df.columns]

# ==== 检测指标列 ====
def detect_metric_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["指标", "维度", "名称", "项目", "metric"]):
            return c
    return df.columns[0]

metric_col = detect_metric_col(df)

# ==== 构造 模型-列名映射 ====
# 允许列名形如：ByteDance评分结果_prompt4.csv / KIMI评分结果_prompt6.csv / Qwen评分结果_prompt2.csv / V3评分结果_prompt4.csv ...
MODEL_PROMPT_COLS = {m: {} for m in MODELS}

def normalize(s: str) -> str:
    return str(s).strip().lower()

def parse_prompt_label(col_name: str) -> str | None:
    m = PROMPT_RE.search(col_name)
    if not m:
        return None
    return f"Prompt{m.group(1)}"  # 统一首字母大写

for col in df.columns[1:]:
    col_norm = normalize(col)
    # 1) 提取 prompt 标签
    p = parse_prompt_label(col)
    if not p:
        continue
    # 2) 匹配模型（用别名）
    matched_model = None
    for display_name, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            if alias in col_norm:
                matched_model = display_name
                break
        if matched_model:
            break
    if matched_model and p in PROMPTS:
        MODEL_PROMPT_COLS[matched_model][p] = col

# 打印一下映射结果，便于核对（可注释掉）
for m in MODELS:
    print(f"[MAP] {m}: {MODEL_PROMPT_COLS[m]}")

# ==== Δ 标注 ====
def annotate_deltas(ax, x_labels, y_vals, fmt="{:+.3g}", y_shift_ratio=0.02):
    import numpy as np
    y = pd.Series(y_vals).astype(float).values
    if len(y) < 2:
        return
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    span = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
    for i in range(1, len(y)):
        if pd.isna(y[i]) or pd.isna(y[i - 1]):
            continue
        dy = y[i] - y[i - 1]
        ax.text(i, y[i] + y_shift_ratio * span, fmt.format(dy),
                ha="center", va="bottom", fontsize=9)

# ==== 提取每个模型在某个指标下的数列 ====
def series_for(model: str, metric_zh: str):
    # 重要：regex=False，避免括号被当作正则
    row = df[df[metric_col].astype(str).str.contains(metric_zh, na=False, regex=False)]
    if row.empty:
        return [None] * len(PROMPTS)
    row = row.iloc[0]
    ys = []
    for p in PROMPTS:
        col = MODEL_PROMPT_COLS.get(model, {}).get(p)
        if not col:
            ys.append(None)
            continue
        val = pd.to_numeric(row[col], errors="coerce")
        ys.append(float(val) if pd.notna(val) else None)
    return ys



# ==== 绘制：上三下二，子图大小一致，间距均匀 ====
# ==== 绘制：上三下二，子图大小一致，间距均匀，颜色改新配色 ====
def plot_all_metrics_grid():
    # 新配色：科研风柔色系（比默认更亮、更区分）
        # 新配色：高区分度，避免与以往重复
    model_colors = {
        "DeepSeek-R1":  "#1f9e89",  # 海松绿
        "DeepSeek-V3.1":"#ffb000",  # 金黄橙
        "DeepSeek-V3":  "#e377c2",  # 粉紫（原色）
    }


    # 创建 2 行 3 列子图（共 6 格，最后一格留空）
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)  # 调大间距，让图像呼吸

    LINE_WIDTH = 2.4
    MARKER_SIZE = 7
    axes = axes.flatten()

    for i, metric_zh in enumerate(ALL_METRICS):
        ax = axes[i]
        title_en = METRIC_MAP[metric_zh]
        drew_any = False

        for model in MODELS:
            ys = series_for(model, metric_zh)
            if metric_zh == METRIC_COMP:
                ys = [v / 100.0 if v is not None else None for v in ys]
            if all(v is None for v in ys):
                continue
            color = model_colors.get(model, None)
            ax.plot(
                PROMPTS,
                ys,
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=model,
                color=color,
                alpha=0.9,  # 稍微柔和一点
            )
            annotate_deltas(ax, PROMPTS, ys)
            drew_any = True

        ax.set_title(f"{title_en}")
        # ax.set_xlabel("Prompts (Prompt2, Prompt4, Prompt6)")
        ax.set_ylabel("Scores (0–1)" if metric_zh == METRIC_COMP else "Original Scores")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=0.65, top=0.825)

        if drew_any:
            ax.legend(fontsize=9)

    # 删除第 6 个空子图
    fig.delaxes(axes[-1])

    fig.tight_layout()
    plt.show()


# ==== 主流程 ====
if __name__ == "__main__":
    plot_all_metrics_grid()
