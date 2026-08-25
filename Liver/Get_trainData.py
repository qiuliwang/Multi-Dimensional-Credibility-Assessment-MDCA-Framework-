import os
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 配置部分 =====
SRC_ROOT = Path(r"E:\Data\Reports_Collected\Processed\检查所见")   # 原始数据目录
DST_ROOT = Path(r"E:\Data\Reports_Collected\Train")      # 输出目录
MAX_THREADS = 8

# 限制规则：每批次下每种类最多多少份
# 本实验仅使用 report_2nd_non-postoperative 批次（Batch2, 15127 份），全部保留
LIMITS = {}

# 随机种子（为了可复现）
random.seed(42)

# ===== 工具函数 =====
def list_reports(batch_dir: Path):
    """返回 (batch, kind, [文件路径列表])"""
    batch_name = batch_dir.name
    kinds = [d for d in batch_dir.iterdir() if d.is_dir()]
    result = []
    for kind_dir in kinds:
        files = [f for f in kind_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
        result.append((batch_name, kind_dir.name, files))
    return result

def copy_one(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def process_kind(batch: str, kind: str, files: list[Path]):
    limit = LIMITS.get(batch, None)
    total = len(files)
    if limit and total > limit:
        sampled = random.sample(files, limit)
    else:
        sampled = files

    dst_dir = DST_ROOT / batch / kind
    for f in sampled:
        dst_path = dst_dir / f.name
        copy_one(f, dst_path)
    return batch, kind, len(sampled), total

# ===== 主流程 =====
def main():
    if not SRC_ROOT.exists():
        print(f"源目录不存在：{SRC_ROOT}")
        return
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # 收集任务
    jobs = []
    for batch_dir in SRC_ROOT.iterdir():
        if not batch_dir.is_dir():
            continue
        jobs.extend(list_reports(batch_dir))

    print(f"共发现 {len(jobs)} 个种类，开始复制至 {DST_ROOT}...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as ex:
        futures = [ex.submit(process_kind, b, k, files) for b, k, files in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())

    # 输出结果统计
    print("\n—— 拷贝完成 ——")
    total_kept = 0
    for batch, kind, kept, total in sorted(results):
        total_kept += kept
        print(f"{batch} / {kind}: {kept}/{total} 份（保留 {kept}）")

    print(f"\n总计保留 {total_kept} 份报告")
    print(f"训练集已生成于：{DST_ROOT}")

if __name__ == "__main__":
    main()
