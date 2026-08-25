# -*- coding: utf-8 -*-
"""
ReportScorerPlus：面向医学报告匹配的评分器
- 单实例模型缓存（同进程同 device/model_name 仅加载一次）
- 线程安全（encode 周围加锁）
- 可用 GPU：自动检测或显式传入 device="cuda"
- 相似度计算支持批量向量化（一次编码一批句子）

与 ReportScorer.py 的关系（供审阅者参考）：
    ReportScorerPlus.py —— 本文件，ReportScorer.py 的优化重构版：
        - 单实例模型缓存（同进程同 device/model_name 仅加载一次）
        - 线程安全（encode 周围加锁）
        - 可用 GPU：自动检测或显式传入 device="cuda"
        - 相似度计算支持批量向量化
    类名与接口与 ReportScorer.py 保持一致，可无缝替换导入；
    Scoring.py 默认使用 ReportScorer.py 的版本。
"""

import os
import re
import unicodedata
from typing import List, Tuple, Dict, Any

from threading import RLock
import torch
from sentence_transformers import SentenceTransformer, util

# 如果需要 HF 国内镜像，请保留；不需要可注释
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ---------- 基础归一化 ----------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace(' ', '')
    text = text.lower()
    text = (text.replace('Ⅰ', 'I').replace('Ⅱ', 'II').replace('Ⅲ', 'III')
                 .replace('Ⅳ', 'IV').replace('Ⅴ', 'V').replace('Ⅵ', 'VI')
                 .replace('Ⅶ', 'VII').replace('Ⅷ', 'VIII').replace('Ⅸ', 'IX'))
    return text

# ---------- 全局模型缓存与锁（单进程内共享） ----------
_MODEL_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}
_MODEL_LOCK = RLock()   # 保护 encode 调用；谨慎起见


class ReportScorerPlus:
    def __init__(
        self,
        top_config: Dict[str, Dict[str, Any]] = None,
        weights: Dict[str, float] = None,
        model_name: str = 'shibing624/text2vec-base-chinese',
        device: str = None,
    ):
        self.TOP_CONFIG = top_config or {
            'TOP1': {'keywords': ['肝脏恶性肿瘤', '肝细胞癌', '胆管细胞癌', '良性肝病变', '转移癌', 'HCCs', 'HCC', 'ICC'], 'weight': 5},
            'TOP3': {'keywords': ['胰腺囊腺瘤', '胰腺IPMN', '血管瘤', '结节性肝硬化', '肝硬化', '脾大', '门脉高压', 'HGDN','高级别不典型增生结节','LGDN','低级别不典型增生结节','DN','不良增生结节','不典型增生结节','腹水', '食管胃底静脉曲张', '门静脉高压', '肝细胞腺瘤', '肝腺瘤', 'HCA', '脂肪肝', '铁过载', '脂肪瘤', '腹膜后淋巴结','胆结石', 'FNH'], 'weight': 3},
            'TOP5': {'keywords': ['肾结石', '肾积水', '输尿管积水', '肝囊肿','肝内小囊肿', '肾囊肿','肾小囊肿', '脾囊肿', '胰腺囊肿', '双肾小囊肿', '左肾囊肿', '右肾囊肿', '囊肿', '肾上腺稍增粗'], 'weight': 1}
        }

        self.MEDICAL_ONTOLOGY = {
            '肝细胞癌': 'malignant', 'HCC': 'malignant', 'ICC': 'malignant', '转移癌': 'malignant', '胆管细胞癌': 'malignant',
            '血管平滑肌脂肪瘤': 'benign', '肝囊肿': 'benign', '肾囊肿': 'benign', 'HCA': 'benign','肾小囊肿': 'benign',
            '肝硬化': 'neutral', '门脉高压': 'neutral', '淋巴结显示': 'neutral', '肝癌': 'malignant', '复发': 'malignant', '血管瘤':'malignant', '囊肿': 'benign', '结节': 'neutral', '支早显':'neutral'
        }

        self.KEYWORD_NORMALIZATION = {
            '细小囊肿': '囊肿', '小囊肿': '囊肿',
            'HCC': '肝细胞癌', 'HCCs': '肝细胞癌', '肝细胞肝癌': '肝细胞癌',
            'ICC': '胆管细胞癌', '异型增生结节': 'HGDN', '局灶性结节增生': 'FNH',
            'DN': '增生结节', 'HGDN': '高级别不典型增生', '胆囊结石': '胆结石',
            '铁沉积': '铁过载', '肝癌': '肝细胞癌', '肝内胆管癌': '胆管细胞癌',
            '胆管癌': '胆管细胞癌', '铁量过载': '铁过载', '小肝癌': '肝细胞癌',
            '肝Ca': '肝细胞癌', '良性病变': '囊肿'
        }

        self.weights = weights or {'auth': 0.2, 'sim': 0.4, 'pri': 0.4}

        # ------- 设备与模型单例 -------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cache_key = (model_name, self.device)
        if cache_key not in _MODEL_CACHE:
            # 首次加载
            try:
                model = SentenceTransformer(model_name, device=self.device)
            except TypeError:
                model = SentenceTransformer(model_name)
                try:
                    model.to(self.device)
                except Exception:
                    pass
            _MODEL_CACHE[cache_key] = model

        self.model = _MODEL_CACHE[cache_key]
        print("✅ SentenceTransformer 初始化完成，设备：", self.model.device)

    # ---------- 小工具 ----------
    def normalize_diagnosis_with_dict(self, diag: str) -> str:
        norm_diag = normalize_text(diag)
        for k, v in self.KEYWORD_NORMALIZATION.items():
            nk, nv = normalize_text(k), normalize_text(v)
            if nk in norm_diag:
                norm_diag = norm_diag.replace(nk, nv)
        return norm_diag

    def extract_diagnoses(self, text: str) -> List[str]:
        # 匹配以 1. / 1、 / 1: / 1： 开头的条目
        lines = re.findall(r'(?:(?<=^)|(?<=\s))(\d+[\.、:：]\s*.*?)(?=(?:\s+\d+[\.、:：])|$)', text.strip())
        return [l.strip() for l in lines if l.strip()]

    def get_label_from_diag(self, diag: str) -> str:
        for term, label in self.MEDICAL_ONTOLOGY.items():
            if term in diag:
                return label
        return 'unknown'

    # ---------- 相似度 ----------
    def sentence_level_similarity(self, gt_text: str, gen_text: str, return_detail: bool = False):
        # 正确：gt 从 gt_text；gen 从 gen_text
        gt_diags = self.extract_diagnoses(gt_text)
        gen_diags = self.extract_diagnoses(gen_text)
        if not gt_diags or not gen_diags:
            return (0.0, []) if return_detail else 0.0

        # 批量编码（加锁，避免线程竞态）
        with _MODEL_LOCK, torch.inference_mode():
            gt_embs = self.model.encode(gt_diags, convert_to_tensor=True, normalize_embeddings=True)
            gen_embs = self.model.encode(gen_diags, convert_to_tensor=True, normalize_embeddings=True)

        similarities: List[float] = []
        detail_matches: List[Dict[str, Any]] = []

        for idx, gt in enumerate(gt_diags):
            gt_vec = gt_embs[idx]
            gt_label = self.get_label_from_diag(gt)

            # 与你原来一致的“窗口比对”策略
            if idx == 0:
                gen_range = range(0, min(1, len(gen_diags)))
            elif idx in [1, 2]:
                gen_range = range(1, min(3, len(gen_diags)))
            elif idx in [3, 4]:
                gen_range = range(3, min(5, len(gen_diags)))
            else:
                gen_range = range(0, len(gen_diags))

            if len(gen_range) == 0:
                similarities.append(0.0)
                if return_detail:
                    detail_matches.append({
                        "gt_diag": gt, "gen_diag": None, "similarity": 0.0,
                        "gt_label": gt_label, "gen_label": "unknown", "label_conflict": False
                    })
                continue

            idxs = list(gen_range)
            cand_embs = gen_embs[idxs]
            sims = util.cos_sim(gt_vec, cand_embs).flatten()
            best_rel = int(torch.argmax(sims).item())
            best_sim = float(sims[best_rel].item())
            best_j = idxs[best_rel]

            gen_candidate = gen_diags[best_j]
            gen_label = self.get_label_from_diag(gen_candidate)
            label_conflict = (
                gt_label != 'unknown' and gen_label != 'unknown' and gt_label != gen_label
            )
            if label_conflict:
                best_sim *= 0.1

            similarities.append(best_sim)
            if return_detail:
                detail_matches.append({
                    "gt_diag": gt,
                    "gen_diag": gen_candidate,
                    "similarity": round(best_sim, 4),
                    "gt_label": gt_label,
                    "gen_label": gen_label,
                    "label_conflict": label_conflict
                })

        avg_sim = round(sum(similarities) / len(similarities), 4)
        return (avg_sim, detail_matches) if return_detail else avg_sim

    # ---------- 真实性 ----------
    def semantic_authenticity_score(self, gt_diags, gen_diags, threshold=None, debug=False):
        if not gt_diags or not gen_diags:
            return 0.0

        all_keywords = []
        for level in self.TOP_CONFIG.values():
            all_keywords.extend(level["keywords"])
        all_keywords = [normalize_text(k) for k in all_keywords]

        norm_gen = [self.normalize_diagnosis_with_dict(g) for g in gen_diags]

        gt_matched_flags = []
        top1_matched = False

        for idx, gt in enumerate(gt_diags):
            norm_gt = self.normalize_diagnosis_with_dict(gt)
            matched = False
            for kw in all_keywords:
                if kw in norm_gt:
                    for g in norm_gen:
                        if kw in g:
                            matched = True
                            break
                if matched:
                    break
            gt_matched_flags.append(matched)
            if idx == 0 and matched:
                top1_matched = True

        matched_count = sum(gt_matched_flags)
        base_score = matched_count / len(gt_diags)
        if not top1_matched:
            base_score *= 0.6
        return round(base_score, 4)

    # ---------- 优先级 ----------
    def compute_priority_score(self, gt_diags, gen_diags, debug=False):
        top1_hit = 0
        rest_total_weight = 0
        rest_hit_weight = 0

        total_keyword_items = 0
        hit_keyword_items = 0

        for idx, gt in enumerate(gt_diags):
            norm_gt = self.normalize_diagnosis_with_dict(gt)
            if idx == 0:
                compare_gen = gen_diags[0:1]
            elif idx in [1, 2]:
                compare_gen = gen_diags[1:3]
            elif idx in [3, 4]:
                compare_gen = gen_diags[3:5]
            else:
                compare_gen = gen_diags

            best_score = 0
            matched = False
            has_keyword = False

            for _, config in self.TOP_CONFIG.items():
                for kw in config["keywords"]:
                    norm_kw = normalize_text(kw)
                    if norm_kw in norm_gt:
                        has_keyword = True
                        for diag in compare_gen:
                            norm_diag = self.normalize_diagnosis_with_dict(diag)
                            if norm_kw in norm_diag:
                                matched = True
                                best_score = max(best_score, config["weight"])
                                break
                if matched:
                    break

            if idx == 0:
                if matched:
                    top1_hit = 1
            else:
                rest_total_weight += best_score
                if matched:
                    rest_hit_weight += best_score

            if idx != 0 and has_keyword:
                total_keyword_items += 1
                if matched:
                    hit_keyword_items += 1

        coverage_score = round(rest_hit_weight / rest_total_weight, 4) if rest_total_weight > 0 else 0.0
        keyword_hit_ratio = round(hit_keyword_items / total_keyword_items, 4) if total_keyword_items > 0 else 0.0
        priority_score = round((top1_hit * 0.5 + coverage_score * 0.5), 4)
        return top1_hit, keyword_hit_ratio, priority_score

    # ---------- 综合 ----------
    def compute_total_score(self, auth, sim, pri):
        w = self.weights
        return round((auth * w['auth'] + sim * w['sim'] + pri * w['pri']) * 100, 2)

    def evaluate(self, original_text, generated_text, auth_threshold=0.7, detail=False):
        gt_diags = self.extract_diagnoses(original_text)
        gen_diags = self.extract_diagnoses(generated_text)

        if detail:
            sim, match_detail = self.sentence_level_similarity(original_text, generated_text, return_detail=True)
        else:
            sim = self.sentence_level_similarity(original_text, generated_text, return_detail=False)
            match_detail = None

        auth = self.semantic_authenticity_score(gt_diags, gen_diags, threshold=auth_threshold, debug=False)
        top1, hit_ratio, pri = self.compute_priority_score(gt_diags, gen_diags, debug=False)
        total = self.compute_total_score(auth, sim, pri)

        result = {
            "语义相似度(BERT)": sim,
            "真实性得分(语义)": auth,
            "Top1匹配": top1,
            "优先级得分": pri,
            "综合评分": total
        }
        if detail and match_detail is not None:
            result["匹配详情"] = match_detail
        return result
