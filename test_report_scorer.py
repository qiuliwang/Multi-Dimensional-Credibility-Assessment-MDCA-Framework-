# -*- coding: utf-8 -*-
'''
测试脚本：验证 ReportScorerPlus 的评分功能
用法：python test_report_scorer.py
'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ReportScorer import ReportScorerPlus

# 模拟一份肝脏 MRI 原报告（Ground Truth）
original = """1. 肝脏右叶可见肝细胞癌，增强后呈快进快出表现，大小约5cm
2. 肝硬化伴门脉高压，脾脏增大，食管胃底静脉曲张
3. 肝左叶多发小囊肿
4. 胰腺未见明确异常"""

# 高质量生成报告：与原文吻合
gen_good = """1. 肝细胞癌，动脉期明显强化，门脉期廓清，呈快进快出
2. 肝硬化、脾大、门脉高压，可见食管胃底静脉曲张
3. 左叶小囊肿
4. 胰腺未见异常"""

# 低质量生成报告：漏掉 TOP1 肝细胞癌，误诊为血管瘤
gen_bad = """1. 肝血管瘤可能，边缘结节样强化
2. 肝脏多发囊肿
3. 未见明显异常"""

scorer = ReportScorerPlus()

print("=" * 60)
print("测试1：高质量生成报告（预期高分）")
r1 = scorer.evaluate(original, gen_good)
for k, v in r1.items():
    print(f"  {k}: {v}")

print("=" * 60)
print("测试2：低质量生成报告（漏掉TOP1肝细胞癌，预期低分）")
r2 = scorer.evaluate(original, gen_bad)
for k, v in r2.items():
    print(f"  {k}: {v}")

print("=" * 60)
print("测试3：报告与自身比对（预期满分/接近满分）")
r3 = scorer.evaluate(original, original)
for k, v in r3.items():
    print(f"  {k}: {v}")

print("=" * 60)
print("测试4：detail=True 模式（输出匹配详情）")
try:
    r4 = scorer.evaluate(original, gen_good, detail=True)
    for k, v in r4.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"  ❌ 出错: {type(e).__name__}: {e}")

print("=" * 60)
print("全部测试完成")
