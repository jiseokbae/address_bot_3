from typing import Any, Dict, List, Tuple
from make_address_insertion.insert_address_multi import _build_component_spans

PLACEHOLDER_SINGLE = "[ADDRESS]"
PLACEHOLDER_A = "[ADDRESS_A]"
PLACEHOLDER_B = "[ADDRESS_B]"
PLACEHOLDER_C = "[ADDRESS_C]"
PLACEHOLDER_1 = "[ADDRESS_1]"
PLACEHOLDER_2 = "[ADDRESS_2]"

PLACEHOLDER_D = "[ADDRESS_D]"
PLACEHOLDER_A1 = "[ADDRESS_A1]"
PLACEHOLDER_A2 = "[ADDRESS_A2]"
PLACEHOLDER_B1 = "[ADDRESS_B1]"
PLACEHOLDER_B2 = "[ADDRESS_B2]"
PLACEHOLDER_C1 = "[ADDRESS_C1]"
PLACEHOLDER_C2 = "[ADDRESS_C2]"

ALL_TRIPLE_PLACEHOLDERS = [
    PLACEHOLDER_A,
    PLACEHOLDER_B,
    PLACEHOLDER_C,
    PLACEHOLDER_1,
    PLACEHOLDER_2,
]


ALL_QUADRA_PLACEHOLDERS = [
    # 기존 triple 계열
    PLACEHOLDER_A,
    PLACEHOLDER_B,
    PLACEHOLDER_C,
    PLACEHOLDER_1,
    PLACEHOLDER_2,

    # quadra 추가 계열
    PLACEHOLDER_D,
    PLACEHOLDER_A1,
    PLACEHOLDER_A2,
    PLACEHOLDER_B1,
    PLACEHOLDER_B2,
    PLACEHOLDER_C1,
    PLACEHOLDER_C2,
]


def _get_address_item_from_bundle(
    address_bundle: Dict[str, Dict[str, Any]],
    group: str,
    form: str,
) -> Dict[str, Any]:
    if group not in address_bundle:
        raise KeyError(f"Missing group in address_bundle: {group}")
    if form not in address_bundle[group]:
        raise KeyError(f"Missing form in address_bundle[{group}]: {form}")
    return address_bundle[group][form]


def _extract_placeholder_occurrences(text: str) -> List[str]:
    """
    text 안의 placeholder를 등장 순서대로 추출.
    """
    placeholders = sorted(ALL_QUADRA_PLACEHOLDERS, key=len, reverse=True)

    out = []
    i = 0
    while i < len(text):
        matched = None
        for ph in placeholders:
            if text.startswith(ph, i):
                matched = ph
                break

        if matched is None:
            i += 1
            continue

        out.append(matched)
        i += len(matched)

    return out


def _build_occurrence_address_items(
    sample: Dict[str, Any],
    address_bundle: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    sample['address_plan'] 순서대로 실제 삽입할 address_item 리스트 생성.
    occurrence_idx 순서(1,2,3,...)와 일치해야 함.
    """
    if "address_plan" not in sample:
        raise KeyError("sample missing 'address_plan'")

    address_plan = sample["address_plan"]
    if not isinstance(address_plan, list):
        raise TypeError("sample['address_plan'] must be a list")

    sorted_plan = sorted(address_plan, key=lambda x: x["occurrence_idx"])

    occurrence_items = []
    for expected_idx, item in enumerate(sorted_plan, start=1):
        occ_idx = item["occurrence_idx"]
        group = item["group"]
        form = item["form"]

        if occ_idx != expected_idx:
            raise ValueError(
                f"address_plan occurrence_idx must be sequential from 1. "
                f"expected={expected_idx}, got={occ_idx}"
            )

        address_item = _get_address_item_from_bundle(address_bundle, group, form)
        occurrence_items.append(address_item)

    return occurrence_items


def _replace_text_by_occurrence_order(
    text: str,
    occurrence_items: List[Dict[str, Any]],
    start_occurrence_idx: int,
    collect_spans: bool,
) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    text를 왼쪽부터 읽으면서 placeholder를 등장 순서대로 occurrence_items에 매핑.
    start_occurrence_idx는 0-based index.
    반환:
      - new_text
      - spans
      - consumed_occurrence_count
    """
    result_parts = []
    spans = []

    i = 0
    built_len = 0
    occ_ptr = start_occurrence_idx

    placeholders = sorted(ALL_QUADRA_PLACEHOLDERS, key=len, reverse=True)

    while i < len(text):
        matched_placeholder = None
        for ph in placeholders:
            if text.startswith(ph, i):
                matched_placeholder = ph
                break

        if matched_placeholder is None:
            result_parts.append(text[i])
            built_len += 1
            i += 1
            continue

        if occ_ptr >= len(occurrence_items):
            raise ValueError(
                f"Not enough occurrence_items for placeholders in text. "
                f"occ_ptr={occ_ptr}, total={len(occurrence_items)}, "
                f"matched_placeholder={matched_placeholder}"
            )

        address_item = occurrence_items[occ_ptr]
        address_text = address_item["text"]
        comp_spans = _build_component_spans(address_item)

        insert_start = built_len
        result_parts.append(address_text)
        built_len += len(address_text)

        if collect_spans:
            for comp in comp_spans:
                spans.append({
                    "slot": comp["slot"],
                    "normalized": comp["normalized"],
                    "start_idx": insert_start + comp["start"],
                    "end_idx": insert_start + comp["end"],
                })

        i += len(matched_placeholder)
        occ_ptr += 1

    consumed = occ_ptr - start_occurrence_idx
    return "".join(result_parts), spans, consumed
    

def insert_triple_address(
    sample: Dict[str, Any],
    address_bundle: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    input:
      - sample
        - sample["text"] 안에 placeholder가 들어있음
        - sample["address_plan"]에 occurrence_idx 순서대로 삽입할 주소 정보가 있음
      - address_bundle
        - 2-depth: group -> form -> address_item

    output:
      - sample copy with updated text, spans
    """
    text = sample["text"]

    placeholders = _extract_placeholder_occurrences(text)

    if len(placeholders) != len(sample["address_plan"]):
        raise ValueError(
            f"Placeholder count mismatch: "
            f"text={len(placeholders)}, plan={len(sample['address_plan'])}"
        )

    plan = sorted(sample["address_plan"], key=lambda x: x["occurrence_idx"])

    for i, p in enumerate(plan, start=1):
        if p["occurrence_idx"] != i:
            raise ValueError(
                f"Invalid occurrence_idx sequence: expected {i}, got {p['occurrence_idx']}"
            )

    occurrence_items = _build_occurrence_address_items(sample, address_bundle)

    new_text, spans, consumed = _replace_text_by_occurrence_order(
        text=text,
        occurrence_items=occurrence_items,
        start_occurrence_idx=0,
        collect_spans=True,
    )

    if consumed != len(occurrence_items):
        raise ValueError(
            f"Not all address_plan occurrences were consumed. "
            f"consumed={consumed}, total={len(occurrence_items)}"
        )

    result = sample.copy()
    result["text"] = new_text
    result["spans"] = spans

    return result


def insert_quadra_address(
    sample: Dict[str, Any],
    address_bundle: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    input:
      - sample
        - sample["text"] 안에 placeholder가 들어있음
        - sample["address_plan"]에 occurrence_idx 순서대로 삽입할 주소 정보가 있음

      - address_bundle
        - 2-depth: group -> form -> address_item

    output:
      - sample copy with updated text, spans
    """
    text = sample["text"]

    placeholders = _extract_placeholder_occurrences(text)

    if len(placeholders) != len(sample["address_plan"]):
        raise ValueError(
            f"Placeholder count mismatch: "
            f"text={len(placeholders)}, plan={len(sample['address_plan'])}, "
            f"placeholders={placeholders}"
        )

    plan = sorted(sample["address_plan"], key=lambda x: x["occurrence_idx"])

    for i, p in enumerate(plan, start=1):
        if p["occurrence_idx"] != i:
            raise ValueError(
                f"Invalid occurrence_idx sequence: "
                f"expected={i}, got={p['occurrence_idx']}"
            )

    occurrence_items = _build_occurrence_address_items(
        sample=sample,
        address_bundle=address_bundle,
    )

    new_text, spans, consumed = _replace_text_by_occurrence_order(
        text=text,
        occurrence_items=occurrence_items,
        start_occurrence_idx=0,
        collect_spans=True,
    )

    if consumed != len(occurrence_items):
        raise ValueError(
            f"Not all address_plan occurrences were consumed. "
            f"consumed={consumed}, total={len(occurrence_items)}"
        )

    result = sample.copy()
    result["text"] = new_text
    result["spans"] = spans

    return result