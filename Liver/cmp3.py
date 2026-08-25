# -*- coding: utf-8 -*-
# 仅在内存中过滤出“所有CSV共同的报告ID”，不导出任何 *_common.csv / 合并CSV
import os, glob, re
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 中文字体（自适应） ----------
from matplotlib import font_manager
candidates = ["Microsoft YaHei","SimHei","SimSun","NSimSun","DengXian",
              "Source Han Sans SC","Noto Sans CJK SC","PingFang SC","Hiragino Sans GB"]
installed = {f.name for f in font_manager.fontManager.ttflist}
chosen = next((f for f in candidates if f in installed), None)
if chosen:
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ---------- (1) role specification ----------
BASE_DIR = "模型对比V3_SUB"
# BASE_DIR = "模型对比P6"
# BASE_DIR = "模型对比KIMI_Sub"
# BASE_DIR = "模型对比DEEP"
# BASE_DIR = "模型对比样例数量KIMI"
# BASE_DIR = "模型对比样例数量DEEP"
# BASE_DIR = "模型对比DEEP_Sub"

NAME_GLOB = "*评分结果_prompt*.csv"

def iter_target_csvs(base):
    pattern = os.path.join(base, "*.csv")   # 不递归，只看当前目录
    for fp in glob.iglob(pattern, recursive=False):
        bn = os.path.basename(fp)
        if re.fullmatch(r".*评分结果_prompt.*\.csv", bn, re.I):
            yield fp

files = sorted(iter_target_csvs(BASE_DIR), key=lambda s: s.lower())
if not files:
    raise SystemExit(f"未找到匹配文件：{NAME_GLOB}（目录={BASE_DIR}）")

print("匹配到的CSV：")
for f in files:
    print(" -", f)

# ---------- (2) task definition ----------
def norm(s: str) -> str:
    return s.strip().lower().replace("（","(").replace("）",")").replace(" ", "")

SYNONYMS = {
    "报告ID": ["报告id","id","编号"],
    "Semantic Coherence": ["语义相似度(bert)","语义相似度","bert相似度","bert相似度分数"],
    "Diagnostic Correctness": ["真实性得分(语义)","真实性得分","真实性(语义)","语义真实性"],
    "Top-1 Matching Scores": ["top1匹配","top1","是否top1匹配","top1match"],
    "Clinical Prioritization Alignment": ["优先级得分","优先级","priority","priorityscore"],
    "MDCA Score": ["综合评分","总分","overall","overallscore"],
}
CANON_BY_NORM = {}
for canon, cands in SYNONYMS.items():
    CANON_BY_NORM[norm(canon)] = canon
    for c in cands:
        CANON_BY_NORM[norm(c)] = canon

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        nc = norm(c)
        if nc in CANON_BY_NORM:
            rename[c] = CANON_BY_NORM[nc]
    df = df.rename(columns=rename)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

# ---------- (3) tiered diagnostic taxonomy (TOP system) ----------
df_dict = {}
for fp in files:
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df = unify_columns(df)
    id_col = next((c for c in df.columns if norm(c) == norm("报告ID")), None)
    if not id_col:
        raise ValueError(f"{fp} 缺少‘报告ID’列")
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=True)
    sys_name = os.path.basename(fp)   # ✅ 直接用文件名作为系统名
    df_dict[sys_name] = df

print("参与对比的系统：", ", ".join(df_dict.keys()))

common_ids = set.intersection(*(set(df.index) for df in df_dict.values()))
common_ids = sorted(common_ids)
if not common_ids:
    raise SystemExit("❌ 各CSV间无公共报告ID，请检查文件。")

print(f"✅ 公共报告ID数量：{len(common_ids)}")
for name in df_dict:
    df_dict[name] = df_dict[name].loc[common_ids]

# ---------- 3.x) 按文件名结尾的数字排序（例如 xxx_12.csv -> 12） ----------
def _tail_number_or_inf(name: str) -> float:
    m = re.search(r'(\d+)\.csv$', os.path.basename(name), re.I)
    return int(m.group(1)) if m else float('inf')  # 无尾号的排在最后

sorted_names = sorted(df_dict.keys(), key=_tail_number_or_inf)
# 重新按顺序构造 df_dict（保持插入有序性）
df_dict = {name: df_dict[name] for name in sorted_names}

print("参与对比的系统（已按尾号排序）：", ", ".join(df_dict.keys()))

# ---------- (4) mandatory verification checkpoints ----------
CAND_NUMERIC = [
    "Semantic Coherence",
    "Diagnostic Correctness",
    "Top-1 Matching Scores",
    "Clinical Prioritization Alignment",
    "MDCA Score",
]

numeric_columns = [c for c in CAND_NUMERIC if all(c in df.columns for df in df_dict.values())]
if not numeric_columns:
    raise SystemExit("没有共同的数值列可对比，请检查表头。")
print("📌 参与对比的指标：", ", ".join(numeric_columns))

# ---------- (5) report formatting standards ----------
import re, textwrap

mean_table = pd.DataFrame(index=numeric_columns)
for name in sorted_names:  # 用已排序的名字保证列顺序
    mean_table[name] = df_dict[name][numeric_columns].mean(numeric_only=True)

colors = ['#cce5ff', '#d4edda', '#fff3cd', '#f8d7da', '#e2e3e5']
highlight = ['#339af0', '#51cf66', '#fcc419', '#f03e3e', '#868e96']

fig_w = 1.2 + len(df_dict) * 1.6
fig_h = 0.60 * len(numeric_columns) + 1.6
fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

rows = list(mean_table.index)
raw_cols = list(mean_table.columns)

# === 1) 列宽分配：行名列固定范围，数据列按“表头长度权重”分配 ===
def _len_weight(s: str) -> int:
    # 下划线/横线/空格视为断点但也占宽，中文和大写稍微权重大些
    # 这是经验性权重，避免极端长名被压扁
    w = 0
    for ch in str(s):
        if '\u4e00' <= ch <= '\u9fff': w += 2
        elif ch.isupper(): w += 2
        elif ch in "_-": w += 1
        else: w += 1
    return max(w, 6)  # 给个下限，避免 0

maxlen_metric = max(len(str(r)) for r in rows) if rows else 6
row_label_w = min(0.45, max(0.20, 0.012 * maxlen_metric))  # 指标列宽：20%~45%

weights = [_len_weight(c) for c in raw_cols]
sum_w = sum(weights)
# 数据区最小/最大列宽限制，避免过宽或过窄；剩余宽度 = 1 - row_label_w
min_w, max_w = 0.06, 0.30
data_area = 1.0 - row_label_w
col_ws = []
for w in weights:
    col_ws.append(w / sum_w * data_area)
# 归一化 + 限幅（简单两轮：先限幅，再按需把多余/不足摊回）
def _limit_and_renorm(widths, total, lo, hi):
    ws = widths[:]
    # 第一次限幅
    ws = [min(max(x, lo), hi) for x in ws]
    s = sum(ws)
    if s == total:
        return ws
    # 需要放缩
    scale = total / s
    ws = [x * scale for x in ws]
    # 二次微调避免浮点误差
    diff = total - sum(ws)
    if abs(diff) > 1e-6:
        ws[0] += diff
    return ws

col_ws = _limit_and_renorm(col_ws, data_area, min_w, max_w)

# === 2) 根据列宽对表头做分行 ===
def wrap_header(s: str, col_width: float) -> str:
    # 估算该列能容纳的字符数：列宽(0~1) * 常数 (经验值，不同字体略有偏差)
    # 常数 34 对 10pt 字体较稳妥，可按需微调
    max_chars = max(6, int(col_width * 34))
    # 优先在分隔符处换行
    parts = re.split(r'([_\- ]+)', str(s))
    merged = []
    line = ""
    for p in parts:
        if len(line) + len(p) <= max_chars:
            line += p
        else:
            if line:
                merged.append(line.rstrip())
            # 如果单段本身过长，再硬切
            if len(p) > max_chars:
                wrapped = textwrap.wrap(p, width=max_chars, break_long_words=True, break_on_hyphens=True)
                if wrapped:
                    merged.append(wrapped[0])
                    line = "".join(wrapped[1:])
                else:
                    line = ""
            else:
                line = p
    if line:
        merged.append(line.rstrip())
    # 再做一次保底硬切，避免极端长词不换行
    out_lines = []
    for seg in merged:
        if len(seg) > max_chars:
            out_lines.extend(textwrap.wrap(seg, width=max_chars, break_long_words=True, break_on_hyphens=True))
        else:
            out_lines.append(seg)
    return "\n".join(out_lines)

cols_wrapped = [wrap_header(c, w) for c, w in zip(raw_cols, col_ws)]

# === 3) 组装 cellText（第一列是“指标”）与 colLabels（加一个“指标”占位） ===
cellText = []
for r in rows:
    row_vals = [f"{mean_table.loc[r, c]:.3f}" for c in raw_cols]
    cellText.append([str(r)] + row_vals)

colLabels = ["指标"] + cols_wrapped
colWidths = [row_label_w] + col_ws

table = plt.table(
    cellText=cellText,
    colLabels=colLabels,
    cellLoc='center',
    colWidths=colWidths,
    bbox=[0.02, 0.02, 0.96, 0.92],
)

# === 4) 紧凑化 & 表头样式 ===
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.05)
for (ij, cell) in table.get_celld().items():
    cell.PAD = 0.02
    # 表头行：减小字体，允许多行
    if ij[0] == 0:
        cell.set_text_props(fontsize=9, weight='bold', va='center')

# 左侧“指标”列左对齐、浅灰底
for i in range(len(rows)):
    c0 = table[i + 1, 0]
    c0.set_text_props(ha='left', va='center', weight='bold')
    c0.set_facecolor('#f1f3f5')

# 数据区着色 + 唯一最大值高亮（注意列索引偏移 +1）
for i, r in enumerate(rows):
    row = mean_table.loc[r]
    vmax = row.max()
    uniq = (row == vmax).sum() == 1
    vmax_col = row.idxmax() if uniq else None
    for j, c in enumerate(raw_cols):
        cell = table[i + 1, j + 1]
        cell.set_facecolor(colors[i % len(colors)])
        if uniq and c == vmax_col:
            cell.set_facecolor(highlight[i % len(highlight)])
            cell.set_text_props(color='white', weight='bold')

fig.suptitle("📊 各系统均值对比（仅公共ID）", fontsize=13, y=0.985)
plt.savefig(BASE_DIR + "_" + str(len(common_ids)) +"-mean_table_common.png", dpi=180, bbox_inches='tight', pad_inches=0.2)
plt.show()

# ---------- (6) radiologic diagnostic principles ----------
# # ---------- 6) 分布图（箱线图，基于公共ID） ----------
# try:
#     import seaborn as sns
#     melted_list = []
#     for name, df in df_dict.items():
#         tmp = df[numeric_columns].copy()
#         tmp["System"] = name
#         tmp["报告ID"] = tmp.index
#         melted_list.append(tmp)
#     melted = pd.concat(melted_list, ignore_index=True)
#     melted = melted.melt(id_vars=["System","报告ID"],
#                          value_vars=numeric_columns,
#                          var_name="Metric", value_name="Score")
#
#     g = sns.FacetGrid(data=melted, col="Metric", col_wrap=3, sharey=False, height=3.8, aspect=0.95)
#
#     def draw(data, **kwargs):
#         ax = plt.gca()
#         sns.boxplot(data=data, x="System", y="Score",
#                     showmeans=True,
#                     meanprops={"marker":"o","markerfacecolor":"black","markeredgecolor":"black","markersize":5},
#                     ax=ax)
#         for i, sys in enumerate(sorted(data['System'].unique())):
#             mean_val = data[data['System']==sys]['Score'].mean()
#             y0, y1 = ax.get_ylim()
#             ax.text(i, mean_val + (y1 - y0) * 0.03, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=9)
#         ax.set_xlabel(""); ax.set_ylabel("")
#
#     g.map_dataframe(draw)
#     g.set_titles(col_template="{col_name}")
#     g.figure.subplots_adjust(top=0.88)
#     g.figure.suptitle("📦 分数分布（仅公共ID）", fontsize=15)
#     plt.savefig("score_boxplots_common.png", dpi=180)
#     plt.show()
# except Exception as e:
#     print("（提示）未安装 seaborn，跳过箱线图；错误：", e)
