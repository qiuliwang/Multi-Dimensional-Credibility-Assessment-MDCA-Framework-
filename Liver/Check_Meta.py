import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sys
import traceback

# ===== 路径配置（按需修改）=====
SRC_ROOT = Path(r"E:\Data\Reports_Collected\Original")   # 源目录：批次/种类/报告.txt
DST_ROOT = Path(r"E:\Data\Reports_Collected\Processed")  # 输出根目录
SUB_SEEN = "检查所见"
SUB_IMPR = "检查结论"

MIN_THREADS = 4  # 至少 4 线程并发

# ===== 读写与解析 =====
def read_txt(p: Path) -> str:
    """
    读取文本并尽可能剥离 BOM：
    1) 优先 utf-8-sig（自动去 BOM）；
    2) 其次 utf-8 / gbk；均再手动去一次 BOM 以防万一；
    3) 兜底：忽略错误，并手动去 BOM。
    """
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            s = p.read_text(encoding=enc)
            return s.lstrip("\ufeff")
        except Exception:
            continue
    return p.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")

def write_txt(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    # 用带 BOM 的 UTF-8，便于 Windows 记事本/Excel 等显示
    p.write_text(text, encoding="utf-8-sig")

def extract_sections(text: str):
    """
    返回 (seen, impr) 两段文本，均为去首尾空白。
    基于明确标题“检查所见”“检查结论”，允许后面紧跟 : 或 ：，并跨行匹配。
    同时允许行首偶发 BOM（\ufeff）。
    """
    pattern_seen = re.compile(rf"^(?:\ufeff)?\s*{re.escape(SUB_SEEN)}\s*[:：]?\s*", re.M)
    pattern_impr = re.compile(rf"^(?:\ufeff)?\s*{re.escape(SUB_IMPR)}\s*[:：]?\s*", re.M)

    m_seen = pattern_seen.search(text)
    m_impr = pattern_impr.search(text)

    if not m_seen or not m_impr:
        raise ValueError("未同时找到两个标题（检查所见/检查结论）。")

    start_seen = m_seen.end()
    start_impr = m_impr.end()

    if m_seen.start() < m_impr.start():
        seen = text[start_seen:m_impr.start()]
        impr = text[start_impr:]
    else:
        # 少见但容错：若“结论”在前
        impr = text[start_impr:m_seen.start()]
        seen = text[start_seen:]

    # 统一换行到 \n 并清理首尾空白
    seen = seen.replace("\r\n", "\n").replace("\r", "\n").strip()
    impr = impr.replace("\r\n", "\n").replace("\r", "\n").strip()
    return seen, impr

def iter_reports(root: Path):
    """
    遍历 Root/批次/种类/*.txt
    如需递归更深层级，可把 kind_dir.iterdir() 改为 kind_dir.rglob("*.txt")
    """
    if not root.exists():
        raise FileNotFoundError(f"源目录不存在：{root}")
    for batch_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for kind_dir in sorted([d for d in batch_dir.iterdir() if d.is_dir()]):
            for f in sorted(kind_dir.iterdir()):
                if f.is_file() and f.suffix.lower() == ".txt":
                    yield batch_dir.name, kind_dir.name, f

def dst_paths(batch: str, kind: str, src_file: Path):
    """返回（所见目标路径, 结论目标路径），保持原文件名。"""
    rel_name = src_file.name
    p_seen = DST_ROOT / SUB_SEEN / batch / kind / rel_name
    p_impr = DST_ROOT / SUB_IMPR / batch / kind / rel_name
    return p_seen, p_impr

def process_one(job):
    batch, kind, fpath = job
    try:
        text = read_txt(fpath)
        seen, impr = extract_sections(text)

        out_seen, out_impr = dst_paths(batch, kind, fpath)
        write_txt(out_seen, seen)
        write_txt(out_impr, impr)

        return (str(fpath), True, "")
    except Exception as e:
        return (str(fpath), False, f"{e.__class__.__name__}: {e}")

def main():
    jobs = list(iter_reports(SRC_ROOT))
    if not jobs:
        print(f"未在 {SRC_ROOT} 下找到 .txt 报告。")
        return

    max_workers = max(MIN_THREADS, (os.cpu_count() or MIN_THREADS))
    print(f"共发现 {len(jobs)} 份报告，使用 {max_workers} 线程拆分保存至：{DST_ROOT}")

    ok = 0
    fails = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, j) for j in jobs]
        for fut in as_completed(futures):
            src, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fails.append((src, msg))

    print("—— 完成 ——")
    print(f"成功：{ok} / {len(jobs)}")
    if fails:
        print(f"失败：{len(fails)}，如下（最多显示 50 条）：")
        for src, msg in fails[:50]:
            print(f"- {src} -> {msg}")
        if len(fails) > 50:
            print("（仅显示前 50 条失败记录）")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("运行出现异常：")
        traceback.print_exc()
        sys.exit(1)
