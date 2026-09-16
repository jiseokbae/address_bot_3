import csv
import json
import re
import sys
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# =========================================
# config
# =========================================
GT_CSV_PATH = "/data/private/address_bot_3/inference/second_poc/(최종)2차PoC_신규TC_주소NER_정답지_260522.csv"

GT_ID_KEY = "TC_No"
GT_NER_KEY = "NER_NORMALIZED_GT"

PRED_ID_KEY_CANDIDATES = ["TC_No", "TC ID", "id"]
PRED_TURNS_KEY = "turns"

SKIP_GT_PARSE_ERROR_TC_NOS = {
}

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
    "읍면동명": "읍면동명",
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
    "동정보": "동정보",
    "층정보": "층정보",
    "호정보": "호정보",
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


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_first_existing(row: Dict[str, Any], candidates: List[str], row_desc: str = "") -> str:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()

    raise KeyError(
        f"{row_desc} 사용 가능한 ID key 없음. candidates={candidates}, row_keys={list(row.keys())}"
    )


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


def split_speaker_line(line: str, tc_id: str, row_idx: int, raw_turn_idx: int):
    """
    입력 예:
        고객: 서울시 강남구 ...
        상담사: 네 확인했습니다

    출력:
        speaker, text

    빈 줄은 None 반환.
    """
    line = str(line).strip()

    if not line:
        return None

    if ":" in line:
        speaker, text = line.split(":", 1)
    elif "：" in line:
        speaker, text = line.split("：", 1)
    else:
        raise ValueError(
            f"[TURN PARSE ERROR] TC_No={tc_id} row={row_idx} raw_turn_idx={raw_turn_idx} | ':' 없는 줄: {line}"
        )

    speaker = speaker.strip()
    text = text.strip()

    if not text:
        return None

    return speaker, text


# =========================================
# GT parser
# =========================================
def parse_gt_turn_tagged_text(
    tagged_text: str,
    tc_id: str,
    turn_idx: int,
) -> Tuple[str, Dict[str, List[str]]]:
    """
    GT의 한 턴 텍스트를 파싱한다.

    입력 예:
        이사 가 주소가 [월드컵로:도로명] [165:건물본번] ...

    출력:
        clean_text:
            이사 가 주소가 월드컵로 165 ...
        entities:
            {
              "도로명": ["월드컵로"],
              "건물본번": ["165"]
            }
    """
    result: Dict[str, List[str]] = defaultdict(list)

    if tagged_text is None:
        return "", dict(result)

    tagged_text = str(tagged_text)

    clean_parts = []
    cursor = 0

    for match in BRACKET_BLOCK_PATTERN.finditer(tagged_text):
        clean_parts.append(tagged_text[cursor:match.start()])

        raw_block = match.group(1).strip()

        if ":" not in raw_block:
            raise ValueError(
                f"[GT PARSE ERROR] TC_No={tc_id} turn_idx={turn_idx} | ':' 없는 block: [{raw_block}]"
            )

        value, tag = raw_block.rsplit(":", 1)
        value = value.strip()
        tag = tag.strip()

        if not value:
            raise ValueError(
                f"[GT PARSE ERROR] TC_No={tc_id} turn_idx={turn_idx} | value 비어있음: [{raw_block}]"
            )

        if not tag:
            raise ValueError(
                f"[GT PARSE ERROR] TC_No={tc_id} turn_idx={turn_idx} | tag 비어있음: [{raw_block}]"
            )

        if tag not in GT_ALLOWED_TAGS:
            raise ValueError(
                f"[GT PARSE ERROR] TC_No={tc_id} turn_idx={turn_idx} | 허용되지 않은 GT tag={tag} | block=[{raw_block}]"
            )

        clean_parts.append(value)
        result[tag].append(value)

        cursor = match.end()

    clean_parts.append(tagged_text[cursor:])
    clean_text = "".join(clean_parts).strip()

    return clean_text, dict(result)

def load_gt_csv(path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    반환 구조:
    {
      "TC_001": {
        0: {
          "turn_idx": 0,
          "raw_turn_idx": 0,
          "speaker": "고객",
          "text": "태그 제거된 턴 텍스트",
          "entities": {"도로명": ["월드컵로"], ...}
        },
        ...
      }
    }

    SKIP_GT_PARSE_ERROR_TC_NOS에 들어있는 TC_No는
    GT 파싱 에러가 발생해도 해당 TC 전체를 스킵한다.
    """
    gt_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
    skipped_tc_nos = []

    encodings = ["utf-8-sig", "utf-8", "cp949", "ms949", "euc-kr", "utf-16"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV header가 없습니다.")

                if GT_ID_KEY not in reader.fieldnames:
                    raise KeyError(
                        f"GT ID 컬럼이 없음: {GT_ID_KEY}, fieldnames={reader.fieldnames}"
                    )

                if GT_NER_KEY not in reader.fieldnames:
                    raise KeyError(
                        f"GT NER 컬럼이 없음: {GT_NER_KEY}, fieldnames={reader.fieldnames}"
                    )

                print(f"[GT CSV OK] encoding={enc}")
                print("GT fieldnames:", reader.fieldnames)

                for row_idx, row in enumerate(reader, start=1):
                    if all((v is None or str(v).strip() == "") for v in row.values()):
                        continue

                    tc_id = row.get(GT_ID_KEY)

                    if not tc_id or str(tc_id).strip() == "":
                        raise ValueError(
                            f"[GT CSV ERROR] row={row_idx} | '{GT_ID_KEY}' 값이 비어있음"
                        )

                    tc_id = str(tc_id).strip()
                    raw_dialogue = row.get(GT_NER_KEY, "")

                    try:
                        turns: Dict[int, Dict[str, Any]] = {}

                        for raw_turn_idx, line in enumerate(str(raw_dialogue).splitlines()):
                            parsed = split_speaker_line(
                                line=line,
                                tc_id=tc_id,
                                row_idx=row_idx,
                                raw_turn_idx=raw_turn_idx,
                            )

                            if parsed is None:
                                continue

                            speaker, tagged_turn_text = parsed
                            turn_idx = len(turns)

                            clean_text, entities = parse_gt_turn_tagged_text(
                                tagged_text=tagged_turn_text,
                                tc_id=tc_id,
                                turn_idx=turn_idx,
                            )

                            turns[turn_idx] = {
                                "turn_idx": turn_idx,
                                "raw_turn_idx": raw_turn_idx,
                                "speaker": speaker,
                                "text": clean_text,
                                "full_turn_text": f"{speaker}: {clean_text}",
                                "entities": entities,
                            }

                        gt_by_id[tc_id] = turns

                        if row_idx <= 2:
                            print(f"[GT sample] TC_No={tc_id}")
                            print(f"[GT sample] raw={raw_dialogue}")
                            print(f"[GT sample] turns={turns}")

                    except Exception as e:
                        if tc_id in SKIP_GT_PARSE_ERROR_TC_NOS:
                            skipped_tc_nos.append(tc_id)
                            print(
                                f"[GT SKIP] TC_No={tc_id} | row={row_idx} | reason={e}",
                                file=sys.stderr,
                            )
                            continue

                        raise

                if skipped_tc_nos:
                    print(
                        f"[GT SKIP SUMMARY] skipped {len(skipped_tc_nos)} TC(s): "
                        f"{sorted(set(skipped_tc_nos))}",
                        file=sys.stderr,
                    )

            return gt_by_id

        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(f"GT CSV 읽기 실패: {path}, last_error={last_error}")

# =========================================
# Pred parser
# =========================================
def parse_pred_entities(entities: List[Dict[str, Any]], tc_id: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)

    if not entities:
        return dict(result)

    for ent in entities:
        pred_tag = str(ent.get("slot", "")).strip()

        if not pred_tag:
            continue

        if pred_tag not in PRED_TO_GT_TAG_MAP:
            raise ValueError(
                f"[PRED PARSE ERROR] TC_No={tc_id} | 허용되지 않은 pred tag: {pred_tag}"
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


def load_pred_json(path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    새 prediction output 구조 기준.

    예상 구조:
    [
      {
        "TC_No": "TC_001",
        "turns": [
          {
            "turn_idx": 0,
            "speaker": "고객",
            "text": "...",
            "entities": [...]
          }
        ]
      }
    ]

    반환 구조:
    {
      "TC_001": {
        0: {
          "turn_idx": 0,
          "speaker": "고객",
          "text": "...",
          "entities": {"도로명": ["월드컵로"], ...}
        }
      }
    }
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    pred_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for row_idx, row in enumerate(data, start=1):
        pred_id = get_first_existing(
            row,
            PRED_ID_KEY_CANDIDATES,
            row_desc=f"[PRED JSON row={row_idx}]",
        )

        turns: Dict[int, Dict[str, Any]] = {}

        for turn in row.get(PRED_TURNS_KEY, []):
            if "turn_idx" not in turn:
                raise ValueError(
                    f"[PRED JSON ERROR] TC_No={pred_id} row={row_idx} | turn_idx 없음: {turn}"
                )

            turn_idx = int(turn.get("turn_idx"))
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()

            parsed_entities = parse_pred_entities(
                entities=turn.get("entities", []),
                tc_id=f"{pred_id}/turn{turn_idx}",
            )

            turns[turn_idx] = {
                "turn_idx": turn_idx,
                "speaker": speaker,
                "text": text,
                "full_turn_text": f"{speaker}: {text}" if speaker else text,
                "entities": parsed_entities,
            }

        pred_by_id[pred_id] = turns

        if row_idx <= 2:
            print(f"[PRED sample] TC_No={pred_id}")
            print(f"[PRED sample] turns={turns}")

    return pred_by_id


# =========================================
# matching
# =========================================
def normalize_for_compare(tag: str, value: str) -> str:
    value = str(value).strip()

    # 도로명만 공백 무시
    if tag == "도로명":
        return "".join(value.split())

    # 그 외는 공백 정리
    return " ".join(value.split())


def match_values_for_tag(
    tag: str,
    gt_values: List[str],
    pred_values: List[str],
):
    """
    tag 하나에 대해 1:1 matching 결과를 반환한다.

    반환:
        used_gt, used_pred, correct_gt, correct_pred
    """
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

    # 2) special rule: 시군구명
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
                        correct_gt += 1

                used_pred[j] = True
                correct_pred += 1
                break

    return used_gt, used_pred, correct_gt, correct_pred


def compute_tag_counts(
    gt_map: Dict[str, List[str]],
    pred_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, int]]:
    tags = set(gt_map.keys()) | set(pred_map.keys())
    out: Dict[str, Dict[str, int]] = {}

    for tag in tags:
        gt_values = list(gt_map.get(tag, []))
        pred_values = list(pred_map.get(tag, []))

        gt_total = len(gt_values)
        pred_total = len(pred_values)

        _, _, correct_gt, correct_pred = match_values_for_tag(
            tag=tag,
            gt_values=gt_values,
            pred_values=pred_values,
        )

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


def compute_row_diffs(
    gt_map: Dict[str, List[str]],
    pred_map: Dict[str, List[str]],
    tc_id: str,
    turn_idx: int,
    speaker: str,
    turn_text: str,
):
    tags = set(gt_map.keys()) | set(pred_map.keys())

    missing_items = []
    error_items = []

    for tag in tags:
        gt_values = list(gt_map.get(tag, []))
        pred_values = list(pred_map.get(tag, []))

        used_gt, used_pred, _, _ = match_values_for_tag(
            tag=tag,
            gt_values=gt_values,
            pred_values=pred_values,
        )

        for i, gt_v in enumerate(gt_values):
            if not used_gt[i]:
                missing_items.append({
                    "TC ID": tc_id,
                    "turn_idx": turn_idx,
                    "speaker": speaker,
                    "turn_text": turn_text,
                    "full_turn_text": f"{speaker}: {turn_text}" if speaker else turn_text,
                    "slot": tag,
                    "value": gt_v,
                })

        for j, pred_v in enumerate(pred_values):
            if not used_pred[j]:
                error_items.append({
                    "TC ID": tc_id,
                    "turn_idx": turn_idx,
                    "speaker": speaker,
                    "turn_text": turn_text,
                    "full_turn_text": f"{speaker}: {turn_text}" if speaker else turn_text,
                    "slot": tag,
                    "value": pred_v,
                })

    return missing_items, error_items


# =========================================
# evaluation
# =========================================
def add_counts_to_total(
    tag_table,
    total: Dict[str, int],
    per_tag: Dict[str, Dict[str, int]],
):
    for tag, stats in per_tag.items():
        for k, v in stats.items():
            tag_table[tag][k] += v
            total[k] += v


def evaluate_one_tc(
    tc_id: str,
    gt_turns: Dict[int, Dict[str, Any]],
    pred_turns: Dict[int, Dict[str, Any]],
):
    """
    TC 하나를 turn 단위로 평가한다.
    """
    tc_tag_counts = defaultdict(lambda: {
        "gt_total": 0,
        "pred_total": 0,
        "correct_gt": 0,
        "correct_pred": 0,
        "missing": 0,
        "error": 0,
    })

    tc_total = {
        "gt_total": 0,
        "pred_total": 0,
        "correct_gt": 0,
        "correct_pred": 0,
        "missing": 0,
        "error": 0,
    }

    tc_missing_items = []
    tc_error_items = []

    turn_indices = sorted(set(gt_turns.keys()) | set(pred_turns.keys()))

    for turn_idx in turn_indices:
        gt_turn = gt_turns.get(turn_idx, {})
        pred_turn = pred_turns.get(turn_idx, {})

        gt_map = gt_turn.get("entities", {})
        pred_map = pred_turn.get("entities", {})

        speaker = (
            gt_turn.get("speaker")
            or pred_turn.get("speaker")
            or ""
        )

        turn_text = (
            gt_turn.get("text")
            or pred_turn.get("text")
            or ""
        )

        per_tag = compute_tag_counts(gt_map, pred_map)
        missing_items, error_items = compute_row_diffs(
            gt_map=gt_map,
            pred_map=pred_map,
            tc_id=tc_id,
            turn_idx=turn_idx,
            speaker=speaker,
            turn_text=turn_text,
        )

        add_counts_to_total(tc_tag_counts, tc_total, per_tag)

        tc_missing_items.extend(missing_items)
        tc_error_items.extend(error_items)

    return tc_tag_counts, tc_total, tc_missing_items, tc_error_items


def evaluate(
    gt_by_id: Dict[str, Dict[int, Dict[str, Any]]],
    pred_by_id: Dict[str, Dict[int, Dict[str, Any]]],
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

    matched_turns = 0
    gt_only_turns = 0
    pred_only_turns = 0

    # GT에만 있는 TC: 전부 missing
    for tc_id in gt_only_ids:
        gt_turns = gt_by_id[tc_id]
        pred_turns = {}

        tc_tag_counts, tc_total, missing_items, error_items = evaluate_one_tc(
            tc_id=tc_id,
            gt_turns=gt_turns,
            pred_turns=pred_turns,
        )

        add_counts_to_total(tag_table, total, tc_tag_counts)

        gt_only_turns += len(gt_turns)

        row_errors.append({
            "TC ID": tc_id,
            "missing": missing_items,
            "error": error_items,
        })

    # GT/PRED 둘 다 있는 TC: turn_idx 기준 비교
    for tc_id in common_ids:
        gt_turns = gt_by_id[tc_id]
        pred_turns = pred_by_id[tc_id]

        tc_tag_counts, tc_total, missing_items, error_items = evaluate_one_tc(
            tc_id=tc_id,
            gt_turns=gt_turns,
            pred_turns=pred_turns,
        )

        add_counts_to_total(tag_table, total, tc_tag_counts)

        matched_turns += len(set(gt_turns.keys()) & set(pred_turns.keys()))
        gt_only_turns += len(set(gt_turns.keys()) - set(pred_turns.keys()))
        pred_only_turns += len(set(pred_turns.keys()) - set(gt_turns.keys()))

        row_errors.append({
            "TC ID": tc_id,
            "missing": missing_items,
            "error": error_items,
        })

    # PRED에만 있는 TC: 전부 error
    for tc_id in pred_only_ids:
        gt_turns = {}
        pred_turns = pred_by_id[tc_id]

        tc_tag_counts, tc_total, missing_items, error_items = evaluate_one_tc(
            tc_id=tc_id,
            gt_turns=gt_turns,
            pred_turns=pred_turns,
        )

        add_counts_to_total(tag_table, total, tc_tag_counts)

        pred_only_turns += len(pred_turns)

        row_errors.append({
            "TC ID": tc_id,
            "missing": missing_items,
            "error": error_items,
        })

    final_tag_table = {}

    for tag, stats in sorted(tag_table.items()):
        metrics = calc_metrics(
            correct_gt=stats["correct_gt"],
            correct_pred=stats["correct_pred"],
            missing=stats["missing"],
            error=stats["error"],
            gt_total=stats["gt_total"],
            pred_total=stats["pred_total"],
        )

        final_tag_table[tag] = {
            "gt_total": stats["gt_total"],
            "pred_total": stats["pred_total"],
            "correct_gt": stats["correct_gt"],
            "correct_pred": stats["correct_pred"],
            "missing": stats["missing"],
            "error": stats["error"],
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
        "matched_turns": matched_turns,
        "gt_only_turns": gt_only_turns,
        "pred_only_turns": pred_only_turns,
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
def main(
    PRED_JSON_PATH: str,
    SAVE_TAG_TABLE_JSON: str,
    SAVE_SUMMARY_JSON: str,
    SAVE_ROW_ERRORS_JSON: str,
):
    gt_by_id = load_gt_csv(GT_CSV_PATH)
    pred_by_id = load_pred_json(PRED_JSON_PATH)

    print(f"loaded gt ids: {len(gt_by_id)}")
    print(f"loaded pred ids: {len(pred_by_id)}")

    tag_table, summary, row_errors = evaluate(gt_by_id, pred_by_id)

    ensure_parent_dir(SAVE_TAG_TABLE_JSON)
    ensure_parent_dir(SAVE_SUMMARY_JSON)
    ensure_parent_dir(SAVE_ROW_ERRORS_JSON)

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

from pathlib import Path
if __name__ == "__main__":
    # DIR_IDX = "2"
    # DIR_IDX = "2-2"
    DIR_IDX = "2-3"

    out_dir = Path(f"result_{DIR_IDX}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for MODEL_TYPE in ["klue"]:
        for ONE in [1000, 2000]:
            for TRI in [3, 4]:
                for QUAD in [3, 4]:
                    for IDX in [1, 2, 3]:
                        PRED_JSON_PATH = (
                            f"/data/private/address_bot_3/inference/output_{DIR_IDX}/"
                            f"poc_inference_results_turn_decoded_"
                            f"{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                        )

                        SAVE_TAG_TABLE_JSON = (
                            f"result_{DIR_IDX}/tag_table_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                        )
                        SAVE_SUMMARY_JSON = (
                            f"result_{DIR_IDX}/summary_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                        )
                        SAVE_ROW_ERRORS_JSON = (
                            f"result_{DIR_IDX}/row_errors_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                        )

                        try:
                            main(
                                PRED_JSON_PATH,
                                SAVE_TAG_TABLE_JSON,
                                SAVE_SUMMARY_JSON,
                                SAVE_ROW_ERRORS_JSON,
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] MODEL_TYPE={MODEL_TYPE}, ONE={ONE}, TRI={TRI}, QUAD={QUAD}, IDX={IDX}",
                                file=sys.stderr,
                            )
                            print(str(e), file=sys.stderr)
                            # sys.exit(1)