# -*- coding: utf-8 -*-
"""
在内存中过滤出“所有 CSV 共同的报告 ID”，并计算：
- 各系统公共样本的指标 均值 和 标准差
结果合并为 “均值 (标准差)” 形式表格，保存到 ./Table/ 下。
"""

import os, glob, re
import pandas as pd

# ---------- 1) 文件读取 ----------
BASE_DIR = "模型对比样例数量DEEP"
NAME_GLOB = "*评分结果_prompt*.csv"

def iter_target_csvs(base):
    pattern = os.path.join(base, NAME_GLOB)
    for fp in glob.iglob(pattern, recursive=False):
        yield fp

files = sorted(iter_target_csvs(BASE_DIR), key=lambda s: s.lower())
if not files:
    raise SystemExit(f"未找到匹配文件：{NAME_GLOB}（目录={BASE_DIR}）")

print("匹配到的CSV：")
for f in files:
    print(" -", f)

# ---------- 2) 列名归一 ----------
def norm(s: str) -> str:
    return s.strip().lower().replace("（","(").replace("）",")").replace(" ", "")

SYNONYMS = {
    "报告ID": ["报告id","id","编号"],
    "语义相似度(BERT)": ["语义相似度(bert)","语义相似度","bert相似度","bert相似度分数"],
    "真实性得分(语义)": ["真实性得分(语义)","真实性得分","真实性(语义)","语义真实性"],
    "Top1匹配": ["top1匹配","top1","是否top1匹配","top1match"],
    "优先级得分": ["优先级得分","优先级","priority","priorityscore"],
    "综合评分": ["综合评分","总分","overall","overallscore"],
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

# ---------- 3) 读取 CSV 并预处理 ----------
CAND_NUMERIC = ["语义相似度(BERT)","真实性得分(语义)","Top1匹配","优先级得分","综合评分"]

df_dict = {}
for fp in files:
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df = unify_columns(df)
    id_col = next((c for c in df.columns if norm(c) == norm("报告ID")), None)
    if not id_col:
        raise ValueError(f"{fp} 缺少‘报告ID’列")
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=True)

    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]

    # 数值列转为数值
    yn_map = {"是":1, "否":0, "true":1, "false":0, "y":1, "n":0, "yes":1, "no":0}
    for col in CAND_NUMERIC:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.lower().map(yn_map).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sys_name = os.path.basename(fp)
    df_dict[sys_name] = df

print("参与对比的系统：", ", ".join(df_dict.keys()))

common_ids = set.intersection(*(set(df.index) for df in df_dict.values()))
common_ids = sorted(common_ids)
if not common_ids:
    raise SystemExit("❌ 各CSV间无公共报告ID，请检查文件。")

print(f"✅ 公共报告ID数量：{len(common_ids)}")
for name in df_dict:
    df_dict[name] = df_dict[name].loc[common_ids]

# ---------- 4) 自定义排序 ----------
ORDER_NUMS = [7, 4, 8, 6, 9, 10, 11]  # ← 你只需改这里
# ORDER_NUMS = [0, 1, 2, 3, 4, 5, 6]  # ← 你只需改这里

def _extract_prompt_num(name: str) -> int:
    m = re.search(r'prompt[_\-]?(\d+)', name, re.I)
    return int(m.group(1)) if m else float('inf')

def _custom_sort_key(name: str):
    num = _extract_prompt_num(name)
    if num in ORDER_NUMS:
        return ORDER_NUMS.index(num)
    else:
        return len(ORDER_NUMS) + num / 1000.0

sorted_names = sorted(df_dict.keys(), key=_custom_sort_key)
df_dict = {name: df_dict[name] for name in sorted_names}

print("✅ 按自定义顺序排序：", ORDER_NUMS)
for n in sorted_names:
    print("   ", n)

# ---------- 5) 共同数值列 ----------
numeric_columns = [c for c in CAND_NUMERIC if all(c in df.columns for df in df_dict.values())]
if not numeric_columns:
    raise SystemExit("没有共同的数值列可对比，请检查表头。")

print("📌 参与对比的指标：", ", ".join(numeric_columns))

# ---------- 6) 计算并合并为 “均值 (标准差)” ----------
result_table = pd.DataFrame(index=numeric_columns)

for name in sorted_names:
    df_num = df_dict[name][numeric_columns]
    mean_series = df_num.mean(numeric_only=True)
    std_series  = df_num.std(numeric_only=True, ddof=1)
    combined_series = mean_series.round(4).astype(str) + " (" + std_series.round(4).astype(str) + ")"
    result_table[name] = combined_series

# ---------- 7) 输出到 Table 文件夹 ----------
os.makedirs("Table", exist_ok=True)
base_name = os.path.basename(BASE_DIR)
out_path = os.path.join("Table", f"{base_name}_{len(common_ids)}-mean_std_common.csv")

result_table.to_csv(out_path, encoding="utf-8-sig")

print("\n✅ 均值(标准差) 表：")
print(result_table)
print(f"\n✅ 已保存：{out_path}")
