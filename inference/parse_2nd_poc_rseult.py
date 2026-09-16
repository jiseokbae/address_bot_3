import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import OrderedDict


# =========================================
# config
# =========================================
DIR_IDX = "2"
RESULT_DIR = Path(f"result_{DIR_IDX}")

GT_CSV_PATH = Path(
    "/data/private/address_bot_3/inference/second_poc/(최종)2차PoC_신규TC_주소NER_정답지_260522.csv"
)

PRED_DIR = Path(f"/data/private/address_bot_3/inference/output_{DIR_IDX}")

MODEL_TYPES = ["klue"]
ONES = [1000]
TRIS = [3]
QUADS = [3]
IDXS = [2]

GT_ID_KEY = "TC_No"

GT_BASE_COLUMNS = [
    "NO",
    "TC_No",
    "TC_Type",
    "주소발화위치",
    "ITN_NORMALIZED_GT",
    "NER_NORMALIZED_GT",
]

# =========================================
# io
# =========================================
def read_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def read_gt_csv_rows(path: Path) -> List[Dict[str, Any]]:
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

                rows = []
                for row in reader:
                    if all((v is None or str(v).strip() == "") for v in row.values()):
                        continue
                    rows.append(row)

                print(f"[GT CSV OK] encoding={enc}, rows={len(rows)}")
                return rows

        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(f"GT CSV 읽기 실패: {path}, last_error={last_error}")


def make_prefix(model_type: str, one: int, tri: int, quad: int, idx: int) -> str:
    return f"{model_type}_{one}_{tri}_{quad}_{idx}"


def get_pred_tc_id(item: Dict[str, Any]) -> str:
    for key in ["TC_No", "TC ID", "id"]:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


# =========================================
# prediction tagged text
# =========================================
def trim_entity_span(target_text: str, entity: Dict[str, Any]) -> Dict[str, Any]:
    start_idx = entity.get("start_idx")
    end_idx = entity.get("end_idx")

    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        return dict(entity)

    if not (0 <= start_idx < end_idx <= len(target_text)):
        return dict(entity)

    original_text = target_text[start_idx:end_idx]

    left_trim = len(original_text) - len(original_text.lstrip())
    right_trim = len(original_text) - len(original_text.rstrip())

    new_start = start_idx + left_trim
    new_end = end_idx - right_trim

    if new_start >= new_end:
        return dict(entity)

    new_entity = dict(entity)
    new_entity["start_idx"] = new_start
    new_entity["end_idx"] = new_end
    new_entity["text"] = target_text[new_start:new_end]

    return new_entity


def build_tagged_text_from_target(
    target_text: str,
    entities: List[Dict[str, Any]],
) -> str:
    """
    turn text 기준 start_idx/end_idx로 [text:slot] 태깅.
    """
    if not isinstance(target_text, str):
        target_text = "" if target_text is None else str(target_text)

    if not isinstance(entities, list):
        entities = []

    valid_entities = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        if "slot" not in ent or "start_idx" not in ent or "end_idx" not in ent:
            continue

        trimmed = trim_entity_span(target_text, ent)

        start_idx = trimmed.get("start_idx")
        end_idx = trimmed.get("end_idx")
        slot = trimmed.get("slot")

        if not isinstance(start_idx, int) or not isinstance(end_idx, int):
            continue

        if not isinstance(slot, str) or not slot.strip():
            continue

        if not (0 <= start_idx < end_idx <= len(target_text)):
            continue

        valid_entities.append(trimmed)

    valid_entities.sort(key=lambda x: (x["start_idx"], x["end_idx"]))

    filtered_entities = []
    prev_end = -1

    for ent in valid_entities:
        if ent["start_idx"] < prev_end:
            continue
        filtered_entities.append(ent)
        prev_end = ent["end_idx"]

    pieces = []
    cursor = 0

    for ent in filtered_entities:
        s = ent["start_idx"]
        e = ent["end_idx"]
        slot = ent["slot"]
        text = target_text[s:e].strip()

        if not text:
            continue

        if cursor < s:
            pieces.append(target_text[cursor:s])

        pieces.append(f"[{text}:{slot}]")
        cursor = e

    if cursor < len(target_text):
        pieces.append(target_text[cursor:])

    return "".join(pieces)


def build_tagged_turn_text(turn: Dict[str, Any]) -> str:
    speaker = str(turn.get("speaker", "")).strip()
    text = turn.get("text", "")

    if text is None:
        text = ""
    text = str(text)

    tagged_text = build_tagged_text_from_target(
        target_text=text,
        entities=turn.get("entities", []),
    )

    if speaker:
        return f"{speaker}: {tagged_text}"

    return tagged_text


def build_prediction_text_from_item(item: Dict[str, Any]) -> str:
    """
    한 TC 안의 모든 turn prediction을 합쳐서 하나의 문자열로 만든다.
    """
    turns = item.get("turns", [])

    if not isinstance(turns, list) or not turns:
        return ""

    sorted_turns = sorted(
        turns,
        key=lambda x: int(x.get("turn_idx", 0))
        if str(x.get("turn_idx", "")).isdigit()
        else 0,
    )

    lines = []

    for turn in sorted_turns:
        line = build_tagged_turn_text(turn)
        if line.strip():
            lines.append(line)

    return "\n".join(lines)


def load_prediction_text_map(pred_json_path: Path) -> Dict[str, str]:
    data = read_json(pred_json_path)

    pred_map = {}

    for item in data:
        tc_id = get_pred_tc_id(item)

        if not tc_id:
            continue

        pred_map[tc_id] = build_prediction_text_from_item(item)

    return pred_map


# =========================================
# row_errors converter
# =========================================
def stringify_error_items(items: List[Dict[str, Any]]) -> str:
    """
    missing/error에는 엔티티만 남긴다.
    turn_idx, speaker, text 전부 제외.
    """
    if not items:
        return ""

    parts = []

    for x in items:
        slot = str(x.get("slot", "")).strip()
        value = str(x.get("value", "")).strip()

        if slot and value:
            parts.append(f"{slot}:{value}")
        elif value:
            parts.append(value)
        elif slot:
            parts.append(slot)

    return " | ".join(parts)


def load_row_error_map(row_error_json: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(row_error_json)

    out = {}

    for row in data:
        tc_id = str(row.get("TC ID", "")).strip()

        if not tc_id:
            continue

        out[tc_id] = row

    return out


def convert_row_errors_json_to_csv(
    row_error_json: Path,
    output_csv: Path,
):
    """
    GT CSV 기준 TC_No 순서를 유지해서 row_errors CSV 생성.
    missing/error는 엔티티만 한 셀에 모아서 저장.
    """
    gt_rows = read_gt_csv_rows(GT_CSV_PATH)
    error_map = load_row_error_map(row_error_json)

    rows = []

    for gt_row in gt_rows:
        tc_id = str(gt_row.get(GT_ID_KEY, "")).strip()
        error_row = error_map.get(tc_id, {})

        missing_items = error_row.get("missing", [])
        error_items = error_row.get("error", [])

        rows.append({
            "NO": gt_row.get("NO", ""),
            "TC_No": tc_id,
            "TC_Type": gt_row.get("TC_Type", ""),
            "주소발화위치": gt_row.get("주소발화위치", ""),
            "missing_count": len(missing_items),
            "error_count": len(error_items),
            "missing": stringify_error_items(missing_items),
            "error": stringify_error_items(error_items),
        })

    ensure_parent_dir(output_csv)

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "NO",
                "TC_No",
                "TC_Type",
                "주소발화위치",
                "missing_count",
                "error_count",
                "missing",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_csv}")


# =========================================
# tag_table converter
# =========================================
def format_float(value: Any, ndigits: int = 6):
    try:
        return round(float(value), ndigits)
    except Exception:
        return value


def convert_tag_table_json_to_csv(
    tag_table_json: Path,
    output_csv: Path,
):
    data = read_json(tag_table_json)

    rows = []

    for tag, stats in data.items():
        rows.append({
            "tag": tag,
            "gt_total": stats.get("gt_total", 0),
            "pred_total": stats.get("pred_total", 0),
            "correct_gt": stats.get("correct_gt", 0),
            "correct_pred": stats.get("correct_pred", 0),
            "missing": stats.get("missing", 0),
            "error": stats.get("error", 0),
            "f1": format_float(stats.get("f1", 0.0)),
            "recall": format_float(stats.get("recall", 0.0)),
            "precision": format_float(stats.get("precision", 0.0)),
            "missing_rate": format_float(stats.get("missing_rate", 0.0)),
            "error_rate": format_float(stats.get("error_rate", 0.0)),
        })

    rows.sort(key=lambda x: x["tag"])

    ensure_parent_dir(output_csv)

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "gt_total",
                "pred_total",
                "correct_gt",
                "correct_pred",
                "missing",
                "error",
                "f1",
                "recall",
                "precision",
                "missing_rate",
                "error_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {output_csv}")


# =========================================
# prediction full converter
# =========================================
def convert_prediction_json_to_csv(
    pred_json: Path,
    output_csv: Path,
    prediction_col_name: str,
):
    """
    GT CSV 기준 TC_No 순서를 유지해서 prediction 전체 결과 CSV 생성.

    - 한 TC당 한 row
    - 모든 turn을 하나의 셀에 합침
    - prediction 없는 TC는 빈칸
    """
    gt_rows = read_gt_csv_rows(GT_CSV_PATH)
    pred_map = load_prediction_text_map(pred_json)

    rows = []

    for gt_row in gt_rows:
        tc_id = str(gt_row.get(GT_ID_KEY, "")).strip()

        out_row = {}

        for col in GT_BASE_COLUMNS:
            out_row[col] = gt_row.get(col, "")

        out_row[prediction_col_name] = pred_map.get(tc_id, "")

        rows.append(out_row)

    ensure_parent_dir(output_csv)

    fieldnames = GT_BASE_COLUMNS + [prediction_col_name]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gt_tc_ids = {str(row.get(GT_ID_KEY, "")).strip() for row in gt_rows}
    pred_tc_ids = set(pred_map.keys())
    pred_only = sorted(pred_tc_ids - gt_tc_ids)

    print(f"saved: {output_csv}")
    print(f"prediction rows based on GT: {len(rows)}")
    print(f"pred_only_count: {len(pred_only)}")
    if pred_only:
        print(f"pred_only_sample: {pred_only[:20]}")


# =========================================
# one-set converter
# =========================================
def convert_one_result(
    model_type: str,
    one: int,
    tri: int,
    quad: int,
    idx: int,
):
    prefix = make_prefix(model_type, one, tri, quad, idx)

    row_error_json = RESULT_DIR / f"row_errors_{prefix}.json"
    tag_table_json = RESULT_DIR / f"tag_table_{prefix}.json"
    pred_json = PRED_DIR / f"poc_inference_results_turn_decoded_{prefix}.json"

    row_error_csv = RESULT_DIR / f"row_errors_{prefix}_for_excel.csv"
    tag_table_csv = RESULT_DIR / f"tag_table_{prefix}_for_excel.csv"
    prediction_csv = RESULT_DIR / f"prediction_{prefix}_for_excel.csv"

    prediction_col_name = f"prediction_{prefix}"

    if row_error_json.exists():
        convert_row_errors_json_to_csv(
            row_error_json=row_error_json,
            output_csv=row_error_csv,
        )
    else:
        print(f"[SKIP] not found: {row_error_json}")

    if tag_table_json.exists():
        convert_tag_table_json_to_csv(
            tag_table_json=tag_table_json,
            output_csv=tag_table_csv,
        )
    else:
        print(f"[SKIP] not found: {tag_table_json}")

    if pred_json.exists():
        convert_prediction_json_to_csv(
            pred_json=pred_json,
            output_csv=prediction_csv,
            prediction_col_name=prediction_col_name,
        )
    else:
        print(f"[SKIP] not found: {pred_json}")


# =========================================
# main
# =========================================
def main():
    for model_type in MODEL_TYPES:
        for one in ONES:
            for tri in TRIS:
                for quad in QUADS:
                    for idx in IDXS:
                        convert_one_result(
                            model_type=model_type,
                            one=one,
                            tri=tri,
                            quad=quad,
                            idx=idx,
                        )


if __name__ == "__main__":
    main()