# -*- coding: utf-8 -*-
r"""
三层结构对齐 + 读取阶段多线程 + 评分阶段多进程
- 生成结论：E:\Data\Reports_Collected\AI_Conclusion\KIMI\{提示词}\{类别}\{批次}\*.txt
- 真实结论：E:\Data\Reports_Collected\Processed\检查结论\{类别}\{批次}\*.txt
- 真实所见：E:\Data\Reports_Collected\Processed\检查所见\{类别}\{批次}\*.txt

# 注意：此处用 r"""

import os
import re
from pathlib import Path
from typing import Dict, List
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ReportScorer import ReportScorerPlus  # 评分逻辑保持不变

# ===== 根目录配置（使用原始字符串 r"" 避免转义）=====
GEN_ROOT = Path(r"E:\Data\Reports_Collected\AI_Conclusion\KIMI")             # 提示词根
REAL_CONCLUSION_ROOT = Path(r"E:\Data\Reports_Collected\Processed\检查结论")
REAL_FINDINGS_ROOT   = Path(r"E:\Data\Reports_Collected\Processed\检查所见")

# ===== 并发参数 =====
READ_THREADS = 16      # 读取 txt 的线程数（I/O 密集）
SCORE_PROCS  = 8      # 评分的进程数（CPU 密集）

# ===== 输出到当前工作目录 =====
OUT_DIR = Path.cwd()

# ===== GPU 开关（尽量用，不改 ReportScorerPlus 内部计算）=====
USE_GPU = True        # 有 GPU 时优先用
CUDA_DEVICE = "0"     # 多卡时可改成 "1" 等

# ===== 基础工具 =====
def read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        return fp.read_text(encoding="utf-8").strip()

def clean_generated_conclusion(text: str) -> str:
    s = text
    s = re.sub(r"^\s*检查结论\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    for p in ("EOB-MRI示", "磁共振肝肿瘤特异性检测"):
        s = s.replace(p, "")
    s = s.lstrip()
    if not re.search(r"(?<!\d)\d+\.(?!\d)", s):
        s = "1. " + s
    else:
        s = re.sub(r"(?<!\d)(\d+)\.(?!\d)", r"\n\1.", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def clean_real_conclusion(text: str) -> str:
    s = text
    s = re.sub(r"^\s*检查结论\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    for p in ("EOB-MRI示", "磁共振肝肿瘤特异性检测"):
        s = s.replace(p, "")
    return s.strip()

# ===== 三层结构对齐 =====
def build_relpath_map(root: Path) -> Dict[str, Path]:
    """
    root 下 {类别}/{批次}/*.txt → 映射：
      key = "类别/批次/文件名"（小写，不区分大小写匹配）
      val = 实际 Path
    """
    mapping: Dict[str, Path] = {}
    for p in root.glob("*/*/*.txt"):
        rel = p.relative_to(root)
        key = str(rel).replace("\\", "/").lower()
        mapping[key] = p
    return mapping

def list_prompt_dirs(gen_root: Path) -> List[Path]:
    if not gen_root.exists():
        print(f"[错误] 生成结论根目录不存在：{gen_root}")
        return []
    return sorted([d for d in gen_root.iterdir() if d.is_dir()])

# ===== 读取阶段：多线程 =====
def _build_row_for_key(key: str, gen_map: Dict[str, Path], realC_map: Dict[str, Path], realF_map: Dict[str, Path]) -> dict:
    """线程执行：读取三方文本并清洗，构造一条记录"""
    gen_fp   = gen_map[key]
    real_cfp = realC_map[key]
    real_ffp = realF_map[key]

    gen_text   = clean_generated_conclusion(read_text(gen_fp))
    real_ctext = clean_real_conclusion(read_text(real_cfp))
    real_ftext = read_text(real_ffp)

    fname = gen_fp.name
    report_id = fname.split("_")[0] if "_" in fname else Path(fname).stem

    # “文件名”写相对路径键，避免跨批次/类别重名冲突
    rel_show = key  # 已是 类别/批次/文件名（小写）

    return {
        "id": report_id,
        "file": rel_show,
        "findings": real_ftext,
        "original_conclusion": real_ctext,
        "generated_conclusion": gen_text,
    }

def collect_for_prompt(prompt_dir: Path, read_threads: int = READ_THREADS) -> List[dict]:
    gen_map   = build_relpath_map(prompt_dir)
    realC_map = build_relpath_map(REAL_CONCLUSION_ROOT)
    realF_map = build_relpath_map(REAL_FINDINGS_ROOT)

    common_keys = sorted(set(gen_map.keys()) & set(realC_map.keys()) & set(realF_map.keys()))
    print(f"📂 提示词目录 [{prompt_dir.name}] 共同报告数：{len(common_keys)}")
    if not common_keys:
        return []

    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=read_threads) as executor:
        futures = {executor.submit(_build_row_for_key, k, gen_map, realC_map, realF_map): k for k in common_keys}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"读取(多线程) {prompt_dir.name}"):
            try:
                rows.append(fut.result())
            except Exception as e:
                k = futures[fut]
                print(f"[读取失败] {prompt_dir.name} :: {k} :: {e}")

    rows.sort(key=lambda x: x["file"])
    return rows

# ===== 评分（保持原逻辑/参数，多进程） =====
_global_scorer = None
def init_worker():
    """子进程内初始化评分器；尽量在 CUDA 上加载"""
    global _global_scorer
    device = "cpu"
    if USE_GPU:
        # 在父进程 main() 里也设置了 CUDA_VISIBLE_DEVICES，以确保子进程继承
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

    # 尽力把 device 传给评分器；若不支持则静默回退
    try:
        _global_scorer = ReportScorerPlus(device=device)
        return
    except TypeError:
        _global_scorer = ReportScorerPlus()
        # 常见接口适配：.to(device) / .set_device(device) / .use_gpu(bool)
        for meth in ("to", "set_device", "use_device", "use_gpu"):
            fn = getattr(_global_scorer, meth, None)
            if callable(fn):
                try:
                    if meth == "use_gpu":
                        fn(device == "cuda")
                    else:
                        fn(device)
                except Exception:
                    pass
                break

def score_row(row: dict) -> dict:
    global _global_scorer
    res = _global_scorer.evaluate(row["original_conclusion"], row["generated_conclusion"], 0.7, False)
    return {
        "报告ID": row["id"],
        "文件名": row["file"],
        "检查所见": row["findings"],
        "原始诊断": row["original_conclusion"],
        "生成诊断": row["generated_conclusion"],
        **res
    }

def run_prompt(prompt_dir: Path, processes: int = SCORE_PROCS) -> pd.DataFrame:
    rows = collect_for_prompt(prompt_dir, read_threads=READ_THREADS)
    if not rows:
        return pd.DataFrame()
    with Pool(processes=processes, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(score_row, rows), total=len(rows), desc=f"评分中 - {prompt_dir.name}"))
    df = pd.DataFrame(results)
    df["来源模型"] = prompt_dir.name
    cols = [
        "报告ID","文件名","来源模型","检查所见","原始诊断","生成诊断",
        "语义相似度(BERT)","真实性得分(语义)","Top1匹配","优先级得分","综合评分"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df

def main():
    # 若希望强制限定使用的 GPU（多卡），在父进程设置环境变量，子进程会继承
    if USE_GPU:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", CUDA_DEVICE)

    prompt_dirs = list_prompt_dirs(GEN_ROOT)
    if not prompt_dirs:
        print("[错误] 未发现任何提示词目录。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdir in prompt_dirs:
        # print(pdir.parts[-1])
        if 'prompt11' in pdir.parts[-1]:
            print(f"\n🚀 开始处理提示词目录：{pdir.name}")
            df = run_prompt(pdir, processes=SCORE_PROCS)
            if df.empty:
                print(f"⚠️ [{pdir.name}] 没有可对齐的报告，跳过导出。")
                continue

            out_xlsx = OUT_DIR / f"评分结果_{pdir.name}.xlsx"
            out_csv  = OUT_DIR / f"评分结果_{pdir.name}.csv"

            df.to_excel(out_xlsx, index=False)
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"✅ [{pdir.name}] 导出完成：\n- {out_xlsx}\n- {out_csv}")

if __name__ == "__main__":
    main()
