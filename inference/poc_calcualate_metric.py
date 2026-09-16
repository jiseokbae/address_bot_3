import csv
import json
import re
import sys
from collections import defaultdict, Counter
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    FIRST_POC_GT_CSV,
    FIRST_POC_OUTPUT_DIR,
    FIRST_POC_RESULT_DIR,
)

# =========================================
# config
# =========================================
GT_CSV_PATH = FIRST_POC_GT_CSV

GT_ID_KEY = "TC ID"
GT_NER_KEY = "NER_NORMALIZED_GT"

PRED_ID_KEY = "id"
PRED_ENTITIES_KEY = "entities"


# =========================================
# label definitions
# =========================================
GT_ALLOWED_TAGS = {
    "시도명",
    "시군구명",
    "읍면동명",
    "도로명",
    "건물본번",
    "건물부번",
    "지번본번",
    "지번부번",
    "건물명",
    "동정보",
    "층정보",
    "호정보",
    "리",
    "지하여부",
    "산여부",
}

PRED_TO_GT_TAG_MAP = {
    "시도명": "시도명",
    "시군구명": "시군구명",
    "법정읍면동명": "읍면동명",
    "리": "리",
    "도로명": "도로명",
    "건물본번": "건물본번",
    "건물부번": "건물부번",
    "지번본번": "지번본번",
    "지번부번": "지번부번",
    "건물명": "건물명",
    "동명칭": "동정보",
    "층명칭": "층정보",
    "호명칭": "호정보",
    "지하여부": "지하여부",
    "산여부": "산여부",
}

# =========================================
# regex
# =========================================
BRACKET_BLOCK_PATTERN = re.compile(r"\[([^\[\]]*)\]")



# =========================================
# helpers
# =========================================
def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def calc_metrics(
    correct_gt: int,
    correct_pred: int,
    missing: int,
    error: int,
    gt_total: int,
    pred_total: int,
) -> dict:
    recall = correct_gt / gt_total if gt_total else 0.0
    precision = correct_pred / pred_total if pred_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    missing_rate = missing / gt_total if gt_total else 0.0
    error_rate = error / pred_total if pred_total else 0.0

    return {
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "missing_rate": missing_rate,
        "error_rate": error_rate,
    }


# =========================================
# parsers
# =========================================
def parse_gt_ner_normalized_strict(tagged_text: str, tc_id: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)

    if tagged_text is None:
        return dict(result)

    tagged_text = str(tagged_text)
    blocks = BRACKET_BLOCK_PATTERN.findall(tagged_text)

    for raw_block in blocks:
        block = raw_block.strip()

        if ":" not in block:
            raise ValueError(
                f"[GT PARSE ERROR] TC ID={tc_id} | ':' 없는 block 발견: [{raw_block}]"
            )

        value, tag = block.rsplit(":", 1)
        value = value.strip()
        tag = tag.strip()

        if not value:
            raise ValueError(
                f"[GT PARSE ERROR] TC ID={tc_id} | value 비어있음: [{raw_block}]"
            )

        if not tag:
            raise ValueError(
                f"[GT PARSE ERROR] TC ID={tc_id} | tag 비어있음: [{raw_block}]"
            )

        if tag not in GT_ALLOWED_TAGS:
            raise ValueError(
                f"[GT PARSE ERROR] TC ID={tc_id} | 허용되지 않은 GT tag: {tag} | block=[{raw_block}]"
            )

        result[tag].append(value)

    return dict(result)


def parse_pred_entities(entities: list[dict], tc_id: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)

    if not entities:
        return dict(result)

    for ent in entities:
        pred_tag = str(ent.get("slot", "")).strip()
        if not pred_tag:
            continue

        if pred_tag not in PRED_TO_GT_TAG_MAP:
            raise ValueError(
                f"[PRED PARSE ERROR] TC ID={tc_id} | 허용되지 않은 pred tag: {pred_tag}"
            )

        gt_tag = PRED_TO_GT_TAG_MAP[pred_tag]

        value = ent.get("normalized_text")
        if value is None or value == "":
            value = ent.get("text", "")

        value = str(value).strip()
        if not value:
            continue

        result[gt_tag].append(value)

    return dict(result)


# =========================================
# readers
# =========================================
def load_gt_csv(path: str) -> dict[str, dict[str, list[str]]]:
    gt_by_id: dict[str, dict[str, list[str]]] = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if GT_ID_KEY not in reader.fieldnames:
            raise KeyError(f"GT ID 컬럼이 없음: {GT_ID_KEY}")
        if GT_NER_KEY not in reader.fieldnames:
            raise KeyError(f"GT NER 컬럼이 없음: {GT_NER_KEY}")

        print("GT fieldnames:", reader.fieldnames)

        for row_idx, row in enumerate(reader, start=1):
            if all((v is None or str(v).strip() == "") for v in row.values()):
                continue

            tc_id = row.get(GT_ID_KEY)
            ner_text = row.get(GT_NER_KEY, "")

            if not tc_id or str(tc_id).strip() == "":
                raise ValueError(f"[GT CSV ERROR] row={row_idx} | '{GT_ID_KEY}' 값이 비어있음")

            tc_id = str(tc_id).strip()
            parsed = parse_gt_ner_normalized_strict(ner_text, tc_id)
            gt_by_id[tc_id] = parsed

            if row_idx <= 2:
                print(f"[GT sample] TC ID={tc_id}")
                print(f"[GT sample] raw={ner_text}")
                print(f"[GT sample] parsed={parsed}")

    return gt_by_id


def load_pred_json(path: str) -> dict[str, dict[str, list[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred_by_id: dict[str, dict[str, list[str]]] = {}

    for idx, row in enumerate(data, start=1):
        pred_id = row.get(PRED_ID_KEY)
        if not pred_id:
            raise ValueError(f"[PRED JSON ERROR] row={idx} | '{PRED_ID_KEY}' 값이 비어있음")

        pred_id = str(pred_id).strip()
        entities = row.get(PRED_ENTITIES_KEY, [])
        parsed = parse_pred_entities(entities, pred_id)
        pred_by_id[pred_id] = parsed

        if idx <= 2:
            print(f"[PRED sample] id={pred_id}")
            print(f"[PRED sample] parsed={parsed}")

    return pred_by_id


# =========================================
# row diff
# =========================================
def compute_row_diffs(
    gt_map: dict[str, list[str]],
    pred_map: dict[str, list[str]]
):
    tags = set(gt_map.keys()) | set(pred_map.keys())

    missing_items = []
    error_items = []

    for tag in tags:
        gt_values = list(gt_map.get(tag, []))
        pred_values = list(pred_map.get(tag, []))

        used_gt = [False] * len(gt_values)
        used_pred = [False] * len(pred_values)

        # 1) 기본 exact/normalized 1:1 match
        for i, gt_v in enumerate(gt_values):
            norm_gt = normalize_for_compare(tag, gt_v)

            for j, pred_v in enumerate(pred_values):
                if used_pred[j]:
                    continue

                norm_pred = normalize_for_compare(tag, pred_v)

                if norm_gt == norm_pred:
                    used_gt[i] = True
                    used_pred[j] = True
                    break

        # 2) 시군구명 special rule
        remain_gt = [gt_values[i] for i in range(len(gt_values)) if not used_gt[i]]
        remain_pred_idx = [j for j in range(len(pred_values)) if not used_pred[j]]

        if tag == "시군구명" and len(remain_gt) >= 2 and len(remain_pred_idx) >= 1:
            merged_gt = normalize_space(" ".join(remain_gt))

            for j in remain_pred_idx:
                pred_v = pred_values[j]
                if normalize_space(pred_v) == merged_gt:
                    for i in range(len(gt_values)):
                        if not used_gt[i]:
                            used_gt[i] = True
                    used_pred[j] = True
                    break

        # 3) 남은 것만 missing / error
        for i, gt_v in enumerate(gt_values):
            if not used_gt[i]:
                missing_items.append({
                    "slot": tag,
                    "value": gt_v,
                })

        for j, pred_v in enumerate(pred_values):
            if not used_pred[j]:
                error_items.append({
                    "slot": tag,
                    "value": pred_v,
                })

    return missing_items, error_items


# =========================================
# matching
# =========================================
def normalize_for_compare(tag: str, value: str) -> str:
    value = str(value).strip()

    # 도로명만 공백 무시
    if tag == "도로명":
        return "".join(value.split())

    # 그 외는 기존처럼 공백 정리만
    return " ".join(value.split())

def compute_tag_counts(
    gt_map: dict[str, list[str]],
    pred_map: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """
    기본은 strict 1:1 matching.
    단,
    - 도로명: 공백 제거 후 비교
    - 시군구명: GT 여러 개를 pred 하나가 공백 결합한 경우 허용
    """
    tags = set(gt_map.keys()) | set(pred_map.keys())
    out: dict[str, dict[str, int]] = {}

    for tag in tags:
        gt_values = list(gt_map.get(tag, []))
        pred_values = list(pred_map.get(tag, []))

        gt_total = len(gt_values)
        pred_total = len(pred_values)

        used_gt = [False] * len(gt_values)
        used_pred = [False] * len(pred_values)

        correct_gt = 0
        correct_pred = 0

        # 1) 기본 exact/normalized 1:1 match
        for i, gt_v in enumerate(gt_values):
            norm_gt = normalize_for_compare(tag, gt_v)

            for j, pred_v in enumerate(pred_values):
                if used_pred[j]:
                    continue

                norm_pred = normalize_for_compare(tag, pred_v)

                if norm_gt == norm_pred:
                    used_gt[i] = True
                    used_pred[j] = True
                    correct_gt += 1
                    correct_pred += 1
                    break

        # 2) special rule: 시군구명만
        remain_gt = [gt_values[i] for i in range(len(gt_values)) if not used_gt[i]]
        remain_pred_idx = [j for j in range(len(pred_values)) if not used_pred[j]]

        if tag == "시군구명" and len(remain_gt) >= 2 and len(remain_pred_idx) >= 1:
            merged_gt = normalize_space(" ".join(remain_gt))

            for j in remain_pred_idx:
                pred_v = pred_values[j]
                if normalize_space(pred_v) == merged_gt:
                    # 남은 GT 전부 정답 처리
                    for i in range(len(gt_values)):
                        if not used_gt[i]:
                            used_gt[i] = True
                            correct_gt += 1

                    used_pred[j] = True
                    correct_pred += 1
                    break

        missing = gt_total - correct_gt
        error = pred_total - correct_pred

        out[tag] = {
            "gt_total": gt_total,
            "pred_total": pred_total,
            "correct_gt": correct_gt,
            "correct_pred": correct_pred,
            "missing": missing,
            "error": error,
        }

    return out

# =========================================
# evaluation
# =========================================
def evaluate(
    gt_by_id: dict[str, dict[str, list[str]]],
    pred_by_id: dict[str, dict[str, list[str]]]
):
    tag_table = defaultdict(lambda: {
        "gt_total": 0,
        "pred_total": 0,
        "correct_gt": 0,
        "correct_pred": 0,
        "missing": 0,
        "error": 0,
    })

    total = {
        "gt_total": 0,
        "pred_total": 0,
        "correct_gt": 0,
        "correct_pred": 0,
        "missing": 0,
        "error": 0,
    }

    row_errors = []

    common_ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    gt_only_ids = sorted(set(gt_by_id.keys()) - set(pred_by_id.keys()))
    pred_only_ids = sorted(set(pred_by_id.keys()) - set(gt_by_id.keys()))

    for tc_id in gt_only_ids:
        gt_map = gt_by_id[tc_id]
        pred_map = {}

        per_tag = compute_tag_counts(gt_map, pred_map)
        missing_items, error_items = compute_row_diffs(gt_map, pred_map)

        for tag, s in per_tag.items():
            for k, v in s.items():
                tag_table[tag][k] += v
                total[k] += v

        row_errors.append({
            "TC ID": tc_id,
            "missing": missing_items,
            "error": error_items,
        })

    for tc_id in common_ids:
        gt_map = gt_by_id[tc_id]
        pred_map = pred_by_id[tc_id]

        per_tag = compute_tag_counts(gt_map, pred_map)
        missing_items, error_items = compute_row_diffs(gt_map, pred_map)

        for tag, s in per_tag.items():
            for k, v in s.items():
                tag_table[tag][k] += v
                total[k] += v

        row_errors.append({
            "TC ID": tc_id,
            "missing": missing_items,
            "error": error_items,
        })

    final_tag_table = {}
    for tag, s in sorted(tag_table.items()):
        metrics = calc_metrics(
            correct_gt=s["correct_gt"],
            correct_pred=s["correct_pred"],
            missing=s["missing"],
            error=s["error"],
            gt_total=s["gt_total"],
            pred_total=s["pred_total"],
        )

        final_tag_table[tag] = {
            "gt_total": s["gt_total"],
            "pred_total": s["pred_total"],
            "correct_gt": s["correct_gt"],
            "correct_pred": s["correct_pred"],
            "missing": s["missing"],
            "error": s["error"],
            **metrics,
        }

    overall_metrics = calc_metrics(
        correct_gt=total["correct_gt"],
        correct_pred=total["correct_pred"],
        missing=total["missing"],
        error=total["error"],
        gt_total=total["gt_total"],
        pred_total=total["pred_total"],
    )

    summary = {
        "matched_ids": len(common_ids),
        "gt_only_ids": len(gt_only_ids),
        "pred_only_ids": len(pred_only_ids),
        "overall": {
            "gt_total": total["gt_total"],
            "pred_total": total["pred_total"],
            "correct_gt": total["correct_gt"],
            "correct_pred": total["correct_pred"],
            "missing": total["missing"],
            "error": total["error"],
            **overall_metrics,
        },
        "pred_only_ids_sample": pred_only_ids[:20],
        "gt_only_ids_sample": gt_only_ids[:20],
        "confusion_matrix_overall": {
            "TP_recall_side": total["correct_gt"],
            "TP_precision_side": total["correct_pred"],
            "FN_missing": total["missing"],
            "FP_error": total["error"],
        },
    }

    return final_tag_table, summary, row_errors


# =========================================
# main
# =========================================
def main(PRED_JSON_PATH, SAVE_TAG_TABLE_JSON, SAVE_SUMMARY_JSON, SAVE_ROW_ERRORS_JSON):
    gt_by_id = load_gt_csv(GT_CSV_PATH)
    pred_by_id = load_pred_json(PRED_JSON_PATH)

    print(f"loaded gt ids: {len(gt_by_id)}")
    print(f"loaded pred ids: {len(pred_by_id)}")

    tag_table, summary, row_errors = evaluate(gt_by_id, pred_by_id)


    with open(SAVE_TAG_TABLE_JSON, "w", encoding="utf-8") as f:
        json.dump(tag_table, f, ensure_ascii=False, indent=2)

    with open(SAVE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(SAVE_ROW_ERRORS_JSON, "w", encoding="utf-8") as f:
        json.dump(row_errors, f, ensure_ascii=False, indent=2)

    print(f"saved: {SAVE_TAG_TABLE_JSON}")
    print(f"saved: {SAVE_SUMMARY_JSON}")
    print(f"saved: {SAVE_ROW_ERRORS_JSON}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    FIRST_POC_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for MODEL_TYPE in ["klue"]:#, "uplus"]:
        for ONE in [1000,2000]: 
            for TRI in [3,4]:  
                    for QUAD in [3,4]: 
                        for IDX in [1,2,3]: 
                            PRED_JSON_PATH = (
                                FIRST_POC_OUTPUT_DIR
                                / f"poc_inference_results_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}_decoded.json"
                            )
                            
                            SAVE_TAG_TABLE_JSON = (
                                FIRST_POC_RESULT_DIR
                                / f"tag_table_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                            )

                            SAVE_SUMMARY_JSON = (
                                FIRST_POC_RESULT_DIR
                                / f"summary_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                            )

                            SAVE_ROW_ERRORS_JSON = (
                                FIRST_POC_RESULT_DIR
                                / f"row_errors_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                            )
    
                            try:
                    
                                main(PRED_JSON_PATH,SAVE_TAG_TABLE_JSON,SAVE_SUMMARY_JSON,SAVE_ROW_ERRORS_JSON)
                            except Exception as e:
                                print(str(e), file=sys.stderr)
                                # sys.exit(1)