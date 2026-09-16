from typing import Any, Dict, List, Tuple

PLACEHOLDER_SINGLE = "[ADDRESS]"
PLACEHOLDER_A = "[ADDRESS_A]"
PLACEHOLDER_B = "[ADDRESS_B]"
PLACEHOLDER_C = "[ADDRESS_C]"
PLACEHOLDER_1 = "[ADDRESS_1]"
PLACEHOLDER_2 = "[ADDRESS_2]"


def _build_component_spans(address_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    address_text = address_item["text"]
    components = address_item["components"]

    comp_spans = []
    cursor = 0

    for comp_idx, comp in enumerate(components):
        slot = comp["slot"]
        normalized = comp["normalized"]
        base_surface = comp["base_surface"]

        if not isinstance(normalized, str):
            raise TypeError(f"component[{comp_idx}] normalized must be str")
        if not isinstance(base_surface, str):
            raise TypeError(f"component[{comp_idx}] base_surface must be str")

        # 학습 대`상` 아닌 슬롯은 제외
        if normalized == "":
            continue

        matched = None
        start = -1

        # 1순위: normalized
        start = address_text.find(normalized, cursor)
        if start != -1:
            matched = normalized

        # 2순위: base_surface
        if start == -1 and base_surface != "":
            start = address_text.find(base_surface, cursor)
            if start != -1:
                matched = base_surface

        if start == -1:
            raise ValueError(
                f"component[{comp_idx}] span not found | "
                f"slot={slot}, normalized={normalized}, base_surface={base_surface}, "
                f"cursor={cursor}, address_text={address_text}"
            )

        end = start + len(matched)

        comp_spans.append({
            "slot": slot,
            "normalized": normalized,
            "start": start,
            "end": end,
        })

        cursor = end

    return comp_spans


def _insert_by_placeholder_map(
    sample: Dict[str, Any],
    placeholder_to_address: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    placeholder마다 대응되는 address_item이 정해져 있는 경우용.
    ex)
      {
        "[ADDRESS_A]": address_a,
        "[ADDRESS_B]": address_b,
      }
    """

    text = sample["text"]

    placeholder_to_text = {}
    placeholder_to_spans = {}

    for placeholder, address_item in placeholder_to_address.items():
        placeholder_to_text[placeholder] = address_item["text"]
        placeholder_to_spans[placeholder] = _build_component_spans(address_item)

    # placeholder 이름이 겹칠 수 있으므로 긴 것부터 검사
    # 예: [ADDRESS] vs [ADDRESS_A]
    placeholders = sorted(placeholder_to_address.keys(), key=len, reverse=True)  
    
    result_parts = []
    spans = []

    i = 0
    built_len = 0
    placeholders = sorted(placeholder_to_address.keys(), key=len, reverse=True)

    while i < len(text):
        matched_placeholder = None

        for placeholder in placeholders:
            if text.startswith(placeholder, i):
                matched_placeholder = placeholder
                break

        if matched_placeholder is None:
            result_parts.append(text[i])
            built_len += 1
            i += 1
            continue

        address_text = placeholder_to_text[matched_placeholder]
        comp_spans = placeholder_to_spans[matched_placeholder]

        insert_start = built_len
        result_parts.append(address_text)
        built_len += len(address_text)

        for comp in comp_spans:
            spans.append({
                "slot": comp["slot"],
                "normalized": comp["normalized"],
                "start_idx": insert_start + comp["start"],
                "end_idx": insert_start + comp["end"],
            })

        i += len(matched_placeholder)

    new_text = "".join(result_parts)

    result = sample.copy()
    result["text"] = new_text
    result["spans"] = spans
    return result


def insert_single_address(
    sample: Dict[str, Any],
    address_item: Dict[str, Any],
) -> Dict[str, Any]:
    return _insert_by_placeholder_map(
        sample=sample,
        placeholder_to_address={
            PLACEHOLDER_SINGLE: address_item,
        },
    )


def insert_double_address(
    sample: Dict[str, Any],
    address_a: Dict[str, Any],
    address_b: Dict[str, Any],
) -> Dict[str, Any]:
    return _insert_by_placeholder_map(
        sample=sample,
        placeholder_to_address={
            PLACEHOLDER_A: address_a,
            PLACEHOLDER_B: address_b,
        },
    )


def insert_split_address(
    sample: Dict[str, Any],
    address_1: Dict[str, Any],
    address_2: Dict[str, Any],
) -> Dict[str, Any]:
    return _insert_by_placeholder_map(
        sample=sample,
        placeholder_to_address={
            PLACEHOLDER_1: address_1,
            PLACEHOLDER_2: address_2,
        },
    )


