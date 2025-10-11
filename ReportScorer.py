'''
    Created by Q. Wang, Department of Radiology, Southwest Hospital, Army Medical University (Third Military Medical University), Chongqing, China
    2025.6.04

    A class for the score between generated and original clinical reports.
'''

import re
import os
import unicodedata
from sentence_transformers import SentenceTransformer, util

def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace(' ', '')
    text = text.lower()
    text = text.replace('Ⅰ', 'I').replace('Ⅱ', 'II').replace('Ⅲ', 'III') \
               .replace('Ⅳ', 'IV').replace('Ⅴ', 'V').replace('Ⅵ', 'VI') \
               .replace('Ⅶ', 'VII').replace('Ⅷ', 'VIII').replace('Ⅸ', 'IX')
    return text

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ReportScorerPlus:
    def __init__(self, top_config=None, weights=None, model_name='shibing624/text2vec-base-chinese'):
        self.TOP_CONFIG = top_config or {
            # 关键词和权重可根据实际需求调整
        'TOP1': {'keywords': ['肝脏恶性肿瘤', '肝细胞癌'], 'weight': 5},
        'TOP3': {'keywords': ['胰腺囊腺瘤', '胰腺IPMN'], 'weight': 3},
        'TOP5': {'keywords': ['肾结石', '肾积水' ], 'weight': 1}
        }

        self.MEDICAL_ONTOLOGY = {
            '肝细胞癌': 'malignant', 'HCC': 'malignant', 'ICC': 'malignant', '转移癌': 'malignant', '胆管细胞癌': 'malignant',
            '血管平滑肌脂肪瘤': 'benign', '肝囊肿': 'benign', '肾囊肿': 'benign', 'HCA': 'benign','肾小囊肿': 'benign',
            '肝硬化': 'neutral', '门脉高压': 'neutral', '淋巴结显示': 'neutral', '肝癌': 'malignant', '复发': 'malignant', '血管瘤':'malignant', '囊肿': 'benign', '结节': 'neutral', '支早显':'neutral'
        }

        self.weights = weights or {'auth': 0.2, 'sim': 0.4, 'pri': 0.4}
        self.model = SentenceTransformer(model_name)
        print("✅ SentenceTransformer 初始化完成，设备：", self.model.device)

        self.KEYWORD_NORMALIZATION = {
            '细小囊肿': '囊肿',
            '小囊肿': '囊肿',
            'HCC': '肝细胞癌',
            'HCCs': '肝细胞癌',
            '肝细胞肝癌': '肝细胞癌',
            'ICC': '胆管细胞癌',
            '异型增生结节': 'HGDN',
            '局灶性结节增生': 'FNH',
            'DN': '增生结节',
            'HGDN': '高级别不典型增生',
            '胆囊结石': '胆结石',
            '铁沉积': '铁过载',
            '肝癌': '肝细胞癌',
            '肝内胆管癌': '胆管细胞癌',
            '胆管癌': '胆管细胞癌',
            '铁量过载': '铁过载',
            '小肝癌': '肝细胞癌',
            '肝Ca': '肝细胞癌',
            '良性病变': '囊肿'
        }

    def normalize_diagnosis_with_dict(self, diag):
        norm_diag = normalize_text(diag)
        for k, v in self.KEYWORD_NORMALIZATION.items():
            if normalize_text(k) in norm_diag:
                norm_diag = norm_diag.replace(normalize_text(k), normalize_text(v))
        return norm_diag

    def extract_diagnoses(self, text):
        lines = re.findall(r'(?:(?<=^)|(?<=\s))(\d+[\.、:：]\s*.*?)(?=(?:\s+\d+[\.、:：])|$)', text.strip())
        return [l.strip() for l in lines if l.strip()]

    def get_label_from_diag(self, diag: str) -> str:
        for term, label in self.MEDICAL_ONTOLOGY.items():
            if term in diag:
                return label
        return 'unknown'

    def sentence_level_similarity(self, gt_text: str, gen_text: str, return_detail=False):
        '''
            这里稍作调整。
        '''
        gen_diags = self.extract_diagnoses(gt_text)
        gt_diags = self.extract_diagnoses(gen_text)        
        # gt_diags = self.extract_diagnoses(gt_text)
        # gen_diags = self.extract_diagnoses(gen_text)
        # print('gen_diags:', gen_diags)
        # print('gt_diags:', gt_diags)

        if not gt_diags or not gen_diags:
            return (0.0, []) if return_detail else 0.0

        similarities = []
        detail_matches = []

        for idx, gt in enumerate(gt_diags):
            gt_vec = self.model.encode(gt, convert_to_tensor=True)
            gt_label = self.get_label_from_diag(gt)

            if idx == 0:
                compare_gen = gen_diags[0:1]
            elif idx in [1, 2]:
                compare_gen = gen_diags[1:3]
            elif idx in [3, 4]:
                compare_gen = gen_diags[3:5]
            else:
                compare_gen = gen_diags

            best_sim = 0.0
            best_match = None

            for gen in compare_gen:
                gen_vec = self.model.encode(gen, convert_to_tensor=True)
                gen_label = self.get_label_from_diag(gen)

                sim = util.cos_sim(gt_vec, gen_vec).item()

                label_conflict = (
                    gt_label != 'unknown' and
                    gen_label != 'unknown' and
                    gt_label != gen_label
                )
                if label_conflict:
                    sim *= 0.1

                if sim > best_sim:
                    best_sim = sim
                    best_match = {
                        "gt_diag": gt,
                        "gen_diag": gen,
                        "similarity": round(sim, 4),
                        "gt_label": gt_label,
                        "gen_label": gen_label,
                        "label_conflict": label_conflict
                    }

            similarities.append(best_sim)
            if return_detail and best_match:
                detail_matches.append(best_match)

        avg_sim = round(sum(similarities) / len(similarities), 4)
        return (avg_sim, detail_matches) if return_detail else avg_sim

    def semantic_authenticity_score(self, gt_diags, gen_diags, threshold=None, debug=False):
        if not gt_diags or not gen_diags:
            return 0.0

        # 抽取关键词全集（合并 TOP1/3/5，不去重也可以）
        all_keywords = []
        for level in self.TOP_CONFIG.values():
            all_keywords.extend(level["keywords"])
        all_keywords = [normalize_text(k) for k in all_keywords]

        # 对生成诊断预处理一次（提速）
        norm_gen = [self.normalize_diagnosis_with_dict(g) for g in gen_diags]

        gt_matched_flags = []
        top1_matched = False

        if debug:
            print("\n🔍 [关键词真实性匹配详情]")
            print(f"共抽取关键词数：{len(all_keywords)}")
            print(f"生成诊断共 {len(gen_diags)} 条\n")

        for idx, gt in enumerate(gt_diags):
            norm_gt = self.normalize_diagnosis_with_dict(gt)
            matched = False
            matched_kw = None

            for kw in all_keywords:
                if kw in norm_gt:
                    # 看这个关键词是否在任意生成诊断中也出现
                    for g in norm_gen:
                        if kw in g:
                            matched = True
                            matched_kw = kw
                            break
                if matched:
                    break

            gt_matched_flags.append(matched)
            if idx == 0 and matched:
                top1_matched = True

            if debug:
                print(f"🧾 GT[{idx+1}]: {gt}")
                print(f"   ➤ 匹配状态: {'✅ 命中' if matched else '❌ 未命中'}")
                print(f"   ➤ 匹配关键词: {matched_kw if matched_kw else '无'}\n")

        matched_count = sum(gt_matched_flags)
        base_score = matched_count / len(gt_diags)

        if not top1_matched:
            base_score *= 0.6
            if debug:
                print("⚠️ Top1（GT[1]）未匹配关键词，得分打折 (×0.6)")

        final_score = round(base_score, 4)

        if debug:
            print(f"\n📊 最终真实性得分（关键词驱动）: {final_score}（命中 {matched_count}/{len(gt_diags)}）")

        return final_score



    def compute_priority_score(self, gt_diags, gen_diags, debug=False):
        top1_weight = 0
        top1_hit = 0
        rest_total_weight = 0
        rest_hit_weight = 0

        # 非加权统计
        total_keyword_items = 0
        hit_keyword_items = 0

        if debug:
            print("\n🔍 [优先级关键词匹配详情]")

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
            best_kw = None
            matched = False
            has_keyword = False  # 用于非加权判断

            for level, config in self.TOP_CONFIG.items():
                for kw in config["keywords"]:
                    norm_kw = normalize_text(kw)
                    if norm_kw in norm_gt:
                        has_keyword = True  # 有关键词
                        for diag in compare_gen:
                            norm_diag = self.normalize_diagnosis_with_dict(diag)
                            if norm_kw in norm_diag:
                                matched = True
                                if config["weight"] > best_score:
                                    best_score = config["weight"]
                                    best_kw = kw
                                break  # 一旦命中就退出关键词循环
                if matched:
                    break

            # 权重部分（用于加权覆盖率）
            if idx == 0:
                top1_weight = best_score
                if matched:
                    top1_hit = 1
            else:
                rest_total_weight += best_score
                if matched:
                    rest_hit_weight += best_score

            # 非加权统计
            if idx != 0 and has_keyword:
                total_keyword_items += 1
                if matched:
                    hit_keyword_items += 1

            # Debug 日志
            if debug:
                print(f"\n🧾 GT[{idx+1}] 原始诊断内容: {gt}")
                print(f"🔑 匹配关键词: {best_kw if best_kw else '无匹配关键词'}")
                print(f"📌 匹配窗口 GEN[{idx+1}] 对比内容: {compare_gen}")
                print(f"📈 匹配状态: {'✅命中' if matched else '❌未命中'}")
                print(f"🎯 权重加分: {best_score}")
                if idx == 0 and not matched:
                    print(f"👉 Top1 项未命中")

        # 加权覆盖率（rest 部分）
        coverage_score = round(
            rest_hit_weight / rest_total_weight, 4
        ) if rest_total_weight > 0 else 0.0

        # 非加权命中比例
        keyword_hit_ratio = round(
            hit_keyword_items / total_keyword_items, 4
        ) if total_keyword_items > 0 else 0.0

        # 最终优先级得分：Top1 × 0.5 + 加权覆盖率 × 0.5
        priority_score = round((top1_hit * 0.5 + coverage_score * 0.5), 4)

        if debug:
            print(f"\n🎯 Top1命中: {top1_hit}，后续覆盖率: {coverage_score}")
            print(f"✅ 关键词命中比例: {keyword_hit_ratio}")
            print(f"📌 综合优先级得分: {priority_score}")

        return top1_hit, keyword_hit_ratio, priority_score


    def compute_total_score(self, auth, sim, pri):
        w = self.weights
        return round((auth * w['auth'] + sim * w['sim'] + pri * w['pri']) * 100, 2)

    def evaluate(self, original_text, generated_text, auth_threshold=0.7, detail=False):
        gt_diags = self.extract_diagnoses(original_text)
        gen_diags = self.extract_diagnoses(generated_text)

        # print(gt_diags)
        # print(gen_diags)

        sim = self.sentence_level_similarity(original_text, generated_text, return_detail=False)

        auth = self.semantic_authenticity_score(gt_diags, gen_diags, threshold=auth_threshold, debug = False)

        # 解包四个优先级指标
        top1, hit_ratio, pri = self.compute_priority_score(gt_diags, gen_diags, debug=False)

        # 综合评分
        total = self.compute_total_score(auth, sim, pri)

        # 打印
        # print("\n📊 评分结果：")
        # print(f"语义相似度(BERT): {sim}")
        # print(f"真实性得分(语义): {auth}")
        # print(f"Top1匹配: {top1}")
        # print(f"关键词命中比例: {hit_ratio}")
        # print(f"优先级得分: {pri}")
        # print(f"综合评分: {total}")

        result = {
            "语义相似度(BERT)": sim,
            "真实性得分(语义)": auth,
            "Top1匹配": top1,
            # "关键词命中比例": hit_ratio,
            "优先级得分": pri,
            "综合评分": total
        }

        if detail:
            result["匹配详情"] = match_detail

        return result
