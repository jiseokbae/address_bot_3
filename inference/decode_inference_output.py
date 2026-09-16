import json
import re
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from paths import FIRST_POC_OUTPUT_DIR
# =========================================================
# io (BOM 대응 및 JSONL 유연성 추가)
# =========================================================
def read_json(path: str):
    # utf-8-sig로 BOM 이슈 해결
    with open(path, "r", encoding="utf-8-sig") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# =========================================================
# BIO decode utils
# =========================================================
def parse_bio_label(label: str) -> Tuple[str, Optional[str]]:
    if label == "O":
        return "O", None
    if "-" not in label:
        return "O", None
    prefix, slot = label.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O", None
    return prefix, slot

def flush_entity(current_entity: Optional[Dict[str, Any]], entities: List[Dict[str, Any]]):
    if current_entity is not None:
        entities.append(current_entity)

def decode_token_predictions_to_entities(token_predictions: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    current_entity: Optional[Dict[str, Any]] = None

    for pred in token_predictions:
        label = pred["label"]
        start_idx = pred["start_idx"]
        end_idx = pred["end_idx"]
        prefix, slot = parse_bio_label(label)

        if prefix == "O":
            flush_entity(current_entity, entities)
            current_entity = None
            continue

        if prefix == "B":
            flush_entity(current_entity, entities)
            current_entity = {"slot": slot, "start_idx": start_idx, "end_idx": end_idx}
            continue

        if current_entity is not None and current_entity["slot"] == slot:
            current_entity["end_idx"] = end_idx
        else:
            flush_entity(current_entity, entities)
            current_entity = {"slot": slot, "start_idx": start_idx, "end_idx": end_idx}

    flush_entity(current_entity, entities)
    for ent in entities:
        ent["text"] = text[ent["start_idx"]:ent["end_idx"]]
    return entities

    
# =========================================================
# Hyphen & Unit Normalization (정규화 로직)
# =========================================================
HYPHEN_CONNECTOR_PATTERN = r"(?:다시|대시|데시|dash|에|의)"
CODE_TOKEN_PATTERN = r"(?:[A-Za-z]{1,3}\d{0,4}|\d{1,4}[A-Za-z]{0,3})"

# 단위를 제거하기 위한 정규식
# 숫자(\d)나 코드([A-Za-z]) 뒤에 '동/층/호'가 붙어 있고, 그 뒤에 단어가 끝날 때(\b) 제거
UNIT_REMOVAL_RE = re.compile(r"([\d[A-Za-z]+)(동|층|호)\b")

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def strip_unit_suffix(text: str) -> str:
    """'101동' -> '101', '3층' -> '3', '7호' -> '7' 처리"""
    return UNIT_REMOVAL_RE.sub(r"\1", text)

def restore_hyphen_for_dong_ho(text: str) -> str:
    text = normalize_space(text)
    # C 대시 7호 -> C-7호 처리
    p1 = re.compile(rf"({CODE_TOKEN_PATTERN})\s*{HYPHEN_CONNECTOR_PATTERN}\s*(\d{{1,4}})(동|호)\b")
    text = p1.sub(r"\1-\2\3", text)
    p2 = re.compile(rf"(\d{{1,4}})\s*{HYPHEN_CONNECTOR_PATTERN}\s*([A-Za-z]{{1,3}})(동|호)\b")
    text = p2.sub(r"\1-\2\3", text)
    return text

def restore_hyphen_for_building(text: str) -> str:
    text = normalize_space(text)
    pattern = re.compile(rf"({CODE_TOKEN_PATTERN})\s*{HYPHEN_CONNECTOR_PATTERN}\s*({CODE_TOKEN_PATTERN})")
    return pattern.sub(r"\1-\2", text)

def normalize_entity_text(slot: str, text: str) -> str:
    text = normalize_space(text)
    
    # 1. 하이픈 복원 우선 적용
    if slot in {"동명칭", "호명칭"}:
        text = restore_hyphen_for_dong_ho(text)
    elif slot == "건물명":
        text = restore_hyphen_for_building(text)

    # 2. 동/층/호 단위 제거 (대상 슬롯인 경우)
    if slot in {"동명칭", "층명칭", "호명칭"}:
        text = strip_unit_suffix(text)

    return text

# =========================================================
# Main Logic
# =========================================================
def decode_item(item: Dict[str, Any]) -> Dict[str, Any]:
    text = item["text"]
    token_predictions = item["token_predictions"]

    entities = decode_token_predictions_to_entities(token_predictions, text)

    for ent in entities:
        ent["normalized_text"] = normalize_entity_text(ent["slot"], ent["text"])

    new_item = dict(item)
    new_item.pop("token_predictions", None)
    new_item["entities"] = entities
    return new_item


def main(input_path: str):
    p = Path(input_path)
    output_path = p.parent / f"{p.stem}_decoded.json"

    print(f"Processing: {input_path}")
    data = read_json(str(input_path))
    
    decoded = [decode_item(item) for item in data]
    # ID 기준 정렬 (idx가 있으면 사용)
    decoded = sorted(decoded, key=lambda x: x.get("id", x.get("idx", 0)))
    
    save_json(decoded, str(output_path))
    print(f"Done. Saved to {output_path} (count: {len(decoded)})")

if __name__ == "__main__":
    base_dir = FIRST_POC_OUTPUT_DIR 
    for MODEL_TYPE in ["klue"]:#, "uplus"]:
        for ONE in [1000,2000]: 
            for TRI in [3,4]:  
                    for QUAD in [3,4]: 
                        for IDX in [1,2,3]: 
                            input_path = base_dir / f"poc_inference_results_{MODEL_TYPE}_{ONE}_{TRI}_{QUAD}_{IDX}.json"
                            # input_path = base_dir / f"poc_itn_inference_results_{m_type}_{idx}.json"

                            if input_path.exists():
                                main(str(input_path))


