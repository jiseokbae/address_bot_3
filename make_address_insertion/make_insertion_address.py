import random
import re
from typing import Dict, Any, List, Tuple


# =========================================================
# basic utils
# =========================================================

def clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none", "null"}:
        return ""
    return x

def unique_keep_order(seq):
    out = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================================================
# particle pool
# =========================================================
PARTICLE_POOL = {
    "시도명": ["", "에", "에서", "으로", "쪽", "쪽에", "의", "인", "에 있는"],
    "시군구명": ["", "에", "에서", "으로", "쪽", "쪽에", "쪽에서", "의", "인", "에 있는", "근처", "근처에"],
    "법정읍면동명": ["", "에", "에서", "으로", "로", "쪽", "쪽에", "의", "인", "에 있는", "근처", "근처에", "부근"],
    "리": ["", "에", "에서", "으로", "로", "쪽", "쪽에", "의", "인", "에 있는", "근처"],

    "도로명": ["", "에", "에서", "으로", "로", "의", "인", "쪽", "쪽에", "에 있는", "방면", "방향", "쪽으로"],
    "건물본번": ["", "에", "에서", "으로", "로"],
    "건물부번": ["", "에", "에서", "으로", "로"],
    "지번본번": ["", "에", "에서", "으로", "로", "번지에", "번지로", "번지에서"],
    "지번부번": ["", "에", "에서", "으로", "로"],

    "건물명": ["", "에", "에서", "으로", "로", "의", "인", "에 있는", "쪽", "쪽에", "근처", "근처에", "부근"],
    "동명칭": ["", "에", "에서", "으로", "로"],
    "층명칭": ["", "에", "에서"],
    "호명칭": ["", "에", "에서", "으로", "로"],
}

def attach_particle(surface: str, slot_name: str, rng: random.Random, prob: float = 0.35) -> str:
    surface = clean(surface)
    if not surface:
        return ""

    pool = PARTICLE_POOL.get(slot_name, [""])
    if pool == [""]:
        return surface

    if rng.random() > prob:
        return surface

    return surface + rng.choice(pool)


# =========================================================
# dropout
# =========================================================

DEFAULT_DROP_PROB = {
    "시도명": 0.45,
    "시군구명": 0.20,
    "법정읍면동명": 0.20,
    "리": 0.65,
    "건물명": 0.55,
    "동명칭": 0.35,
    "층명칭": 0.60,
    "호명칭": 0.25,
}


def should_drop(slot_name: str, rng: random.Random, drop_prob: Dict[str, float]) -> bool:
    p = drop_prob.get(slot_name, 0.0)
    return rng.random() < p


# =========================================================
# hyphen / connector variation
# =========================================================

def sample_number_connector(rng: random.Random) -> str:
    return rng.choice([
        "-",               # 기본

        # 정상 발화
        " 다시 ",
        " 에 ",
        " 의 ",
        " 번지 ",
        " 번지에 ",
        " 번지의 ",

        # 구어체/흐림
        " 에서 ",
        " 쪽 ",
        " 쪽에 ",
        " 쪽으로 ",

        # STT 깨짐 느낌
        " 다시에 ",
        " 다시의 ",
        " 에다 ",
        " 에다가 ",
        " 의에 ",         # 일부러 이상하게
        " 에 의 ",
        " 의 다시 ",

        # 짧은 연결
        " 에 ",
        " 에 ",
        " 다시 ",

        # 붙여 읽힘 (띄어쓰기 깨짐)
        "다시",
        "에",
        "의",

        # 중간 잡음 느낌
        " 에 뭐 ",
        " 다시 뭐 ",
        " 에 그 ",
        " 다시 그 ",

        # 숫자 연결 발화 느낌
        " 번지 다시 ",
        " 번지 에 ",
        " 번지 의 ",
    ])


def sample_text_hyphen_connector(rng: random.Random) -> str:
    """
    건물명 / 동명칭 / 호명칭 등 일반 문자열 내부 하이픈용
    예:
    - B-7동 -> B 다시 7동 / B에 7동 / B의 7동 / B다시7동
    - C-19호 -> C 다시 19호 / C에 19호
    """
    return rng.choice([
        "-",                # 원형 유지

        # 기본 변형
        " 다시 ",
        " 에 ",
        " 의 ",

        # 붙여쓰기
        "다시",
        "에",
        "의",

        # STT/ITN 노이즈
        " 다시에 ",
        " 에다 ",
        " 의에 ",

        # 약한 변형
        " ",
    ])

def split_hyphen_text(text: str) -> List[str]:
    """
    하이픈 기준 분리. 연속 하이픈까지는 깊게 안 봄.
    """
    return [t.strip() for t in re.split(r"\s*-\s*", text) if clean(t)]


def vary_hyphenated_text(text: str, rng: random.Random, prob: float = 0.7) -> str:
    """
    건물명/동명칭/호명칭 등에 '-'가 있으면 일부를 한국어 연결로 변형
    예:
      상가-2동 -> 상가 다시 2동 / 상가에 2동 / 상가의 2동
      B-7동 -> B 다시 7동 / B에 7동 / B의 7동
      C-19호 -> C 다시 19호 / C에 19호
    """
    text = clean(text)
    if not text or "-" not in text:
        return text

    if rng.random() > prob:
        return text

    parts = split_hyphen_text(text)
    if len(parts) < 2:
        return text

    connector = sample_text_hyphen_connector(rng)
    return connector.join(parts)


# =========================================================
# normalize
# - 중요: 지번/건물번호는 합치지 않음
# =========================================================

def sample_floor_suffix(rng: random.Random) -> str:
    """
    층 suffix (무조건 붙음)
    """
    return rng.choice([
        "층",     # 정상
        "층",
        # STT 변형
        "증",
        "승",
        "쯩",
    ])

def make_floor_surface(raw_floor: str, rng: random.Random) -> str:
    raw_floor = clean(raw_floor)
    if not raw_floor:
        return ""
    floor_suffix = sample_floor_suffix(rng)
        
    return f"{raw_floor}{floor_suffix}"


def sample_ho_suffix(rng: random.Random) -> str:
    """
    호 suffix (선택적으로 붙음)
    """
    return rng.choice([
        "",
        "",
        "호",
        "호",
        "오",
        "소",
        "호수",
        "호로",
        "호에",
        "호이",
        "호여",
    ])

def make_dong_surface(raw_dong: str, rng: random.Random) -> str:
    """
    동명칭 surface 생성
    - 기본: 동 붙이기
    - STT 노이즈 일부 포함
    """
    raw_dong = clean(raw_dong)
    if not raw_dong:
        return ""

    return rng.choice([
        f"{raw_dong}",
        f"{raw_dong}동",
        f"{raw_dong}동",
        f"{raw_dong}동",   # 정상 케이스 확률 높이기

        # STT 깨짐
        f"{raw_dong}동이",
        f"{raw_dong}동으",
        f"{raw_dong}송",
        f"{raw_dong}똥",
        f"{raw_dong}똥에",
        f"{raw_dong}소에",
        f"{raw_dong}소",
    ])

def make_ho_surface(raw_ho: str, rng: random.Random, raw_ho_suffix: str = "") -> str:
    raw_ho = clean(raw_ho)
    raw_ho_suffix = clean(raw_ho_suffix)

    if not raw_ho:
        return ""

    # 호접미사명칭이 있으면 우선 활용
    # 단, 완전 중복 숫자형이면 버림
    if raw_ho_suffix:
        if raw_ho_suffix != raw_ho:
            # 숫자 중복 아닌 경우엔 합쳐진 값도 후보로 사용
            suffix_candidates = [
                raw_ho_suffix,
                f" {raw_ho_suffix}",
            ]
        else:
            suffix_candidates = [""]
    else:
        suffix_candidates = []

    default_suffix = sample_ho_suffix(rng)

    candidates = [
        f"{raw_ho}{default_suffix}",
    ]

    for s in suffix_candidates:
        candidates.append(f"{raw_ho}{s}")

    # 중복 제거
    candidates = unique_keep_order([c for c in candidates if clean(c) != ""])

    if not candidates:
        return raw_ho

    return rng.choice(candidates)


def normalize_used_keys(used: Dict[str, Any],) -> Dict[str, str]:
    slot = {}

    # 지역
    if clean(used.get("시도명")):
        slot["시도명"] = clean(used.get("시도명"))
    if clean(used.get("시군구명")):
        slot["시군구명"] = clean(used.get("시군구명"))
    if clean(used.get("법정읍면동명")):
        slot["법정읍면동명"] = clean(used.get("법정읍면동명"))

    ri = clean(used.get("리")) or clean(used.get("법정리명"))
    if ri:
        slot["리"] = ri

    # 건물명 통합
    building = (
        clean(used.get("상세건물명"))
        or clean(used.get("건축물대장 건물명"))
        or clean(used.get("시군구용 건물명"))
    )
    if building:
        slot["건물명"] = building

    # 상세주소
    dong = clean(used.get("동명칭"))
    if dong:
        slot["동명칭"] = dong

    # 층명칭
    floor = clean(used.get("층명칭"))
    if floor:
        slot["층명칭"] = floor

    # 호명칭 / 호접미사명칭
    ho = clean(used.get("호명칭"))
    ho_suffix = clean(used.get("호접미사명칭")) or clean(used.get("호전미사명칭"))
    if ho:
        slot["호명칭"] = ho
    if ho_suffix:
        slot["호접미사명칭"] = ho_suffix

    # 공통
    if clean(used.get("지하여부")):
        slot["지하여부"] = clean(used.get("지하여부"))

    # 지번형
    if clean(used.get("산여부")):
        slot["산여부"] = clean(used.get("산여부"))

    if clean(used.get("지번본번(번지)")):
        slot["지번본번"] = clean(used.get("지번본번(번지)"))
    if clean(used.get("지번부번(호)")):
        slot["지번부번"] = clean(used.get("지번부번(호)"))

    # 도로명형
    if clean(used.get("도로명")):
        slot["도로명"] = clean(used.get("도로명"))
    if clean(used.get("건물본번")):
        slot["건물본번"] = clean(used.get("건물본번"))
    if clean(used.get("건물부번")):
        slot["건물부번"] = clean(used.get("건물부번"))

    return slot


def detect_address_type(slot: Dict[str, str]) -> str:
    if slot.get("도로명") and slot.get("건물본번"):
        return "road"
    if slot.get("지번본번"):
        return "jibun"
    return "unknown"


# =========================================================
# 도로명 숫자 앞에 spacing 랜덤 삽입입
# =========================================================
def vary_road_name_spacing(text: str, rng: random.Random, prob: float = 0.6) -> str:
    text = clean(text)
    if not text:
        return text

    if rng.random() > prob:
        return text

    mode = rng.choice(["before_number", "before_number", "before_and_after"])

    if mode == "before_number":
        return re.sub(r'([가-힣A-Za-z])(\d)', r'\1 \2', text)

    text = re.sub(r'([가-힣A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([가-힣])', r'\1 \2', text)
    return text


# =========================================================
# component builders
# - 중요: 본번/부번을 합치지 않고 component를 분리 유지
# - 연결 표현은 별도 component로 넣음
# =========================================================
def build_component_sequence(
    slot: Dict[str, str],
    addr_type: str,
    rng: random.Random,
) -> List[Dict[str, str]]:
    seq: List[Dict[str, str]] = []

    def add(slot_name: str, raw_value: str, base_surface: str, normalized_value: str):
        if not clean(base_surface):
            return

        final_surface = attach_particle(base_surface, slot_name, rng)
        seq.append({
            "slot": slot_name,
            "raw": clean(raw_value),
            "surface": clean(final_surface),         # text에 실제 들어갈 값
            "base_surface": clean(base_surface),     # 조사 붙이기 전 값
            "normalized": clean(normalized_value),   # NER target 값
        })

    # -----------------------------------------------------
    # 지역
    # normalized = DB raw
    # -----------------------------------------------------
    if slot.get("시도명"):
        raw = slot["시도명"]
        add("시도명", raw, raw, raw)

    if slot.get("시군구명"):
        raw = slot["시군구명"]
        add("시군구명", raw, raw, raw)

    if slot.get("법정읍면동명"):
        raw = slot["법정읍면동명"]
        add("법정읍면동명", raw, raw, raw)

    if slot.get("리"):
        raw = slot["리"]
        add("리", raw, raw, raw)

    # -----------------------------------------------------
    # 도로명 / 지번 핵심
    # 본번/부번은 분리 유지
    # 연결어는 O용이라 normalized 없음
    # -----------------------------------------------------
    if addr_type == "road":
        if slot.get("지하여부"):
            raw = slot["지하여부"]
            add("지하여부", raw, raw, raw)

        if slot.get("도로명"):
            raw = slot["도로명"]

            base_surface = vary_road_name_spacing(raw, rng)

            add("도로명", raw, base_surface, raw)

        if slot.get("건물본번"):
            raw = slot["건물본번"]
            add("건물본번", raw, raw, raw)

        if slot.get("건물부번"):
            seq.append({
                "slot": "건물번호연결",
                "raw": "",
                "surface": sample_number_connector(rng),
                "base_surface": "",
                "normalized": "",
            })
            raw = slot["건물부번"]
            add("건물부번", raw, raw, raw)

    elif addr_type == "jibun":
        if slot.get("산여부"):
            raw = slot["산여부"]
            add("산여부", raw, raw, raw)

        if slot.get("지하여부"):
            raw = slot["지하여부"]
            add("지하여부", raw, raw, raw)

        if slot.get("지번본번"):
            raw = slot["지번본번"]
            add("지번본번", raw, raw, raw)

        if slot.get("지번부번"):
            seq.append({
                "slot": "지번연결",
                "raw": "",
                "surface": sample_number_connector(rng),
                "base_surface": "",
                "normalized": "",
            })
            raw = slot["지번부번"]
            add("지번부번", raw, raw, raw)

    # -----------------------------------------------------
    # 건물명
    # 하이픈이 있으면 base_surface를 normalized로 사용
    # 조사 붙은 final surface의 조사는 O
    # -----------------------------------------------------
    if slot.get("건물명"):
        raw = slot["건물명"]
        base_surface = vary_hyphenated_text(raw, rng)
        normalized = base_surface if "-" in raw else raw
        add("건물명", raw, base_surface, normalized)

    # -----------------------------------------------------
    # 동명칭
    # 하이픈이 있으면 hyphen 변형 후 동 surface 생성
    # normalized는:
    # - raw에 하이픈 있으면 base_surface
    # - 아니면 raw
    # -----------------------------------------------------
    if slot.get("동명칭"):
        raw = slot["동명칭"]
        hyphen_base = vary_hyphenated_text(raw, rng) if "-" in raw else raw
        base_surface = make_dong_surface(hyphen_base, rng)
        normalized = base_surface if "-" in raw else raw
        add("동명칭", raw, base_surface, normalized)

    # -----------------------------------------------------
    # 층명칭
    # normalized는 항상 DB raw
    # -----------------------------------------------------
    if slot.get("층명칭"):
        raw = slot["층명칭"]
        base_surface = make_floor_surface(raw, rng)
        add("층명칭", raw, base_surface, raw)

    # -----------------------------------------------------
    # 호명칭
    # raw에 하이픈 있으면 hyphen 변형 후 ho surface 생성
    # normalized는:
    # - raw에 하이픈 있으면 base_surface
    # - 아니면 raw
    # -----------------------------------------------------
    if slot.get("호명칭"):
        raw = slot["호명칭"]
        raw_ho_suffix = slot.get("호접미사명칭", "")

        hyphen_base = vary_hyphenated_text(raw, rng) if "-" in raw else raw
        base_surface = make_ho_surface(hyphen_base, rng, raw_ho_suffix=raw_ho_suffix)

        # raw 자체에 하이픈이 있으면 그 전체 surface를 하나의 span으로 봄
        normalized = base_surface if "-" in raw else raw

        add("호명칭", raw, base_surface, normalized)

    return seq

# =========================================================
# dropout with min_keep
# =========================================================
def apply_dropout_components(
    components: List[Dict[str, str]],
    rng: random.Random,
    drop_prob: Dict[str, float],
    addr_type: str,
    min_keep: int = 3,
) -> List[Dict[str, str]]:
    """
    component dict 버전 dropout
    - 핵심 slot은 유지
    - 최종 visible slot 수가 min_keep 이상 되도록 보정
    """
    kept = []
    dropped = []

    core_slots = set()
    if addr_type == "road":
        core_slots |= {"도로명", "건물본번"}
    elif addr_type == "jibun":
        core_slots |= {"지번본번"}

    core_slots |= {"산여부", "지하여부", "건물부번", "지번부번", "건물번호연결", "지번연결"}

    for comp in components:
        slot = comp["slot"]

        if slot in core_slots:
            kept.append(comp)
            continue

        if should_drop(slot, rng, drop_prob):
            dropped.append(comp)
        else:
            kept.append(comp)

    # visible component 개수 기준 최소 보정
    visible_kept = [c for c in kept if c["slot"] not in {"건물번호연결", "지번연결"}]

    if len(visible_kept) < min_keep:
        need = min_keep - len(visible_kept)
        for comp in dropped:
            if comp not in kept:
                kept.append(comp)
                if comp["slot"] not in {"건물번호연결", "지번연결"}:
                    need -= 1
                if need == 0:
                    break

    # 원래 순서 복원
    kept_set = {id(c) for c in kept}
    kept = [c for c in components if id(c) in kept_set]

    # 연결 token 및 부번 단독 정리
    cleaned = []
    for i, comp in enumerate(kept):
        slot = comp["slot"]

        if slot == "건물번호연결":
            prev_ok = i - 1 >= 0 and kept[i - 1]["slot"] == "건물본번"
            next_ok = i + 1 < len(kept) and kept[i + 1]["slot"] == "건물부번"
            if prev_ok and next_ok:
                cleaned.append(comp)
            continue

        if slot == "지번연결":
            prev_ok = i - 1 >= 0 and kept[i - 1]["slot"] == "지번본번"
            next_ok = i + 1 < len(kept) and kept[i + 1]["slot"] == "지번부번"
            if prev_ok and next_ok:
                cleaned.append(comp)
            continue

        if slot == "건물부번":
            prev_has_main = any(c["slot"] == "건물본번" for c in kept[:i])
            if prev_has_main:
                cleaned.append(comp)
            continue

        if slot == "지번부번":
            prev_has_main = any(c["slot"] == "지번본번" for c in kept[:i])
            if prev_has_main:
                cleaned.append(comp)
            continue

        cleaned.append(comp)

    return cleaned

# =========================================================
# text compose
# - 연결 토큰은 공백 없이 붙임
# =========================================================

CONNECTOR_SLOTS = {"지번연결", "건물번호연결"}

def compose_text_from_components(components: List[Dict[str, str]]) -> str:
    buf: List[str] = []

    for comp in components:
        slot = comp["slot"]
        surface = clean(comp["surface"])
        if not surface:
            continue

        if slot in CONNECTOR_SLOTS:
            if not buf:
                continue
            buf[-1] = buf[-1] + surface
        else:
            buf.append(surface)

    return " ".join(buf).strip()


# =========================================================
# normalized_slot_values
# - NER target 기준
# - 빈 normalized("")인 연결어는 제외
# =========================================================

def build_normalized_slot_values(components: List[Dict[str, str]]) -> Dict[str, str]:
    result = {}
    for comp in components:
        slot = comp["slot"]
        normalized = clean(comp["normalized"])
        if not normalized:
            continue
        result[slot] = normalized
    return result


# =========================================================
# 건물번호 및 지번이 이상하게 존재할 경우 제거
# =========================================================

def cleanup_connector_components(components: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []

    for i, comp in enumerate(components):
        slot = comp["slot"]

        if slot == "건물번호연결":
            prev_ok = i - 1 >= 0 and components[i - 1]["slot"] == "건물본번"
            next_ok = i + 1 < len(components) and components[i + 1]["slot"] == "건물부번"
            if prev_ok and next_ok:
                cleaned.append(comp)
            continue

        if slot == "지번연결":
            prev_ok = i - 1 >= 0 and components[i - 1]["slot"] == "지번본번"
            next_ok = i + 1 < len(components) and components[i + 1]["slot"] == "지번부번"
            if prev_ok and next_ok:
                cleaned.append(comp)
            continue

        if slot == "건물부번":
            prev_has_main = any(c["slot"] == "건물본번" for c in components[:i])
            if prev_has_main:
                cleaned.append(comp)
            continue

        if slot == "지번부번":
            prev_has_main = any(c["slot"] == "지번본번" for c in components[:i])
            if prev_has_main:
                cleaned.append(comp)
            continue

        cleaned.append(comp)

    return cleaned


# =========================================================
# single address
# =========================================================
def compose_address_from_db_row(
    row: Dict[str, Any],
    rng: random.Random,
    drop_prob: Dict[str, float] = None,
    min_keep: int = 3,
) -> Dict[str, Any]:
    """
    최종 address 생성
    - raw_slot_values: DB 기준 원값
    - components: text에 실제 들어갈 surface 조각
    - normalized_slot_values: NER target 값
    """
    drop_prob = drop_prob or DEFAULT_DROP_PROB

    used = row.get("used_key_values", {})
    raw_slot_values = normalize_used_keys(used)
    addr_type = detect_address_type(raw_slot_values)

    # 1) 전체 component 생성 (이미 surface 포함)
    components = build_component_sequence(raw_slot_values, addr_type, rng)

    # 2) dropout 적용
    components = apply_dropout_components(
        components=components,
        rng=rng,
        drop_prob=drop_prob,
        addr_type=addr_type,
        min_keep=min_keep,
    )


    # 3) text / normalized 구성
    text = compose_text_from_components(components)
    normalized_slot_values = build_normalized_slot_values(components)

    return {
        "text": text,
        "components": components,
        "normalized_slot_values": normalized_slot_values,
        "raw_slot_values": raw_slot_values,
        "address_type": addr_type,
    }

# =========================================================
# single address without drop
# =========================================================
def compose_full_address_from_db_row(
    row: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    누락 없이 전체 주소 생성
    - raw_slot_values: DB 기준 원값
    - components: text에 실제 들어갈 surface 조각
    - normalized_slot_values: NER target 값
    - compose_address_from_db_row와 동일한 반환 형식
    - 차이점: dropout을 적용하지 않음
    """
    used = row.get("used_key_values", {})
    raw_slot_values = normalize_used_keys(used)
    addr_type = detect_address_type(raw_slot_values)

    # 전체 component 생성 (surface 포함)
    components = build_component_sequence(raw_slot_values, addr_type, rng)

    # 연결 token / 부번 단독 정리만 수행
    components = cleanup_connector_components(components)

    text = compose_text_from_components(components)
    normalized_slot_values = build_normalized_slot_values(components)

    return {
        "text": text,
        "components": components,
        "normalized_slot_values": normalized_slot_values,
        "raw_slot_values": raw_slot_values,
        "address_type": addr_type,
    }



# =========================================================
# split pair for [ADDRESS_1], [ADDRESS_2]
# - ADDRESS_1: 큰 주소 중심
# - ADDRESS_2: 상세 주소 중심
# - 일부 겹침 허용
# =========================================================

def build_split_address_pair(
    row: Dict[str, Any],
    rng: random.Random,
    min_a1: int = 2,
    min_a2: int = 3,
) -> Dict[str, Any]:
    """
    [ADDRESS_1], [ADDRESS_2]용
    - ADDRESS_1: 큰 주소 중심
    - ADDRESS_2: 상세 주소 중심
    - 일부 겹침 허용

    반환:
    {
        "address_1": ...,
        "address_2": ...,
        "components_1": [...],
        "components_2": [...],
        "normalized_slot_values_1": {...},
        "normalized_slot_values_2": {...},
        "raw_slot_values": {...},
        "address_type": ...
    }
    """
    used = row.get("used_key_values", {})
    raw_slot_values = normalize_used_keys(used)
    addr_type = detect_address_type(raw_slot_values)

    # 전체 component 먼저 생성
    all_components = build_component_sequence(raw_slot_values, addr_type, rng)

    # 카테고리 분리
    region_slots = []
    core_slots = []
    detail_slots = []

    region_slot_names = {"시도명", "시군구명", "법정읍면동명", "리"}
    detail_slot_names = {"건물명", "동명칭", "층명칭", "호명칭"}

    if addr_type == "road":
        core_slot_names = {"지하여부", "도로명", "건물본번", "건물번호연결", "건물부번"}
    else:
        core_slot_names = {"산여부", "지하여부", "지번본번", "지번연결", "지번부번"}

    for comp in all_components:
        slot = comp["slot"]
        if slot in region_slot_names:
            region_slots.append(comp)
        elif slot in detail_slot_names:
            detail_slots.append(comp)
        elif slot in core_slot_names:
            core_slots.append(comp)

    # =====================================================
    # ADDRESS_1: 큰 주소 중심
    # =====================================================
    a1 = []

    for comp in region_slots:
        if rng.random() < 0.85:
            a1.append(comp)

    if addr_type == "road":
        road_comp = next((c for c in core_slots if c["slot"] == "도로명"), None)
        main_comp = next((c for c in core_slots if c["slot"] == "건물본번"), None)
        conn_comp = next((c for c in core_slots if c["slot"] == "건물번호연결"), None)
        sub_comp = next((c for c in core_slots if c["slot"] == "건물부번"), None)

        if road_comp and rng.random() < 0.60:
            a1.append(road_comp)
        if main_comp and rng.random() < 0.25:
            a1.append(main_comp)
        if sub_comp and main_comp in a1 and rng.random() < 0.20:
            if conn_comp:
                a1.append(conn_comp)
            a1.append(sub_comp)

    elif addr_type == "jibun":
        san_comp = next((c for c in core_slots if c["slot"] == "산여부"), None)
        main_comp = next((c for c in core_slots if c["slot"] == "지번본번"), None)
        conn_comp = next((c for c in core_slots if c["slot"] == "지번연결"), None)
        sub_comp = next((c for c in core_slots if c["slot"] == "지번부번"), None)

        if san_comp and rng.random() < 0.45:
            a1.append(san_comp)
        if main_comp and rng.random() < 0.60:
            a1.append(main_comp)
        if sub_comp and main_comp in a1 and rng.random() < 0.20:
            if conn_comp:
                a1.append(conn_comp)
            a1.append(sub_comp)

    a1_candidates = region_slots + core_slots
    visible_a1 = [c for c in a1 if c["slot"] not in {"건물번호연결", "지번연결"}]
    if len(visible_a1) < min_a1:
        for comp in a1_candidates:
            if comp not in a1:
                a1.append(comp)
                visible_a1 = [c for c in a1 if c["slot"] not in {"건물번호연결", "지번연결"}]
                if len(visible_a1) >= min_a1:
                    break

    a1_set = {id(c) for c in a1}
    a1_ordered = [c for c in (region_slots + core_slots) if id(c) in a1_set]

    # 연결 component 정리
    a1_ordered = cleanup_connector_components(a1_ordered)

    address_1 = compose_text_from_components(a1_ordered)
    normalized_slot_values_1 = build_normalized_slot_values(a1_ordered)

    # =====================================================
    # ADDRESS_2: 상세 주소 중심
    # =====================================================
    a2 = []

    for comp in region_slots:
        if rng.random() < 0.45:
            a2.append(comp)

    for comp in core_slots:
        if comp["slot"] in {"도로명", "건물본번", "지번본번"}:
            a2.append(comp)
        elif comp["slot"] in {"건물번호연결", "지번연결"}:
            a2.append(comp)
        else:
            if rng.random() < 0.80:
                a2.append(comp)

    for comp in detail_slots:
        if rng.random() < 0.75:
            a2.append(comp)

    a2_candidates = region_slots + core_slots + detail_slots
    visible_a2 = [c for c in a2 if c["slot"] not in {"건물번호연결", "지번연결"}]
    if len(visible_a2) < min_a2:
        for comp in a2_candidates:
            if comp not in a2:
                a2.append(comp)
                visible_a2 = [c for c in a2 if c["slot"] not in {"건물번호연결", "지번연결"}]
                if len(visible_a2) >= min_a2:
                    break

    a2_set = {id(c) for c in a2}
    a2_ordered = [c for c in (region_slots + core_slots + detail_slots) if id(c) in a2_set]

    # 연결 component 정리
    a2_ordered = cleanup_connector_components(a2_ordered)

    address_2 = compose_text_from_components(a2_ordered)
    normalized_slot_values_2 = build_normalized_slot_values(a2_ordered)

    address_1_item = {
        "text": address_1,
        "components": a1_ordered,
        "normalized_slot_values": normalized_slot_values_1,
        "raw_slot_values": raw_slot_values,
        "address_type": addr_type,
    }

    address_2_item = {
        "text": address_2,
        "components": a2_ordered,
        "normalized_slot_values": normalized_slot_values_2,
        "raw_slot_values": raw_slot_values,
        "address_type": addr_type,
    }

    return address_1_item, address_2_item
    # return {
    #     "address_1": address_1,
    #     "address_2": address_2,
    #     "address_type": addr_type,
    #     "components_1": a1_ordered,
    #     "components_2": a2_ordered,
    #     "normalized_slot_values_1": normalized_slot_values_1,
    #     "normalized_slot_values_2": normalized_slot_values_2,
    #     "raw_slot_values": raw_slot_values,
    # }

def build_shallow_component_candidates(
    slot: Dict[str, str],
    addr_type: str,
    rng: random.Random,
) -> List[List[Dict[str, str]]]:
    """
    짧은 주소 후보 패턴들.
    각 후보는 component dict list.
    목표:
    - 도로명/지번 수준까지만
    - 최대 3개 정도의 필드
    - 문장 안에 여러 ADDRESS가 들어갈 때 쓰기 좋게 짧게
    """

    def make_comp(slot_name: str, raw_value: str) -> Dict[str, str]:
        raw_value = clean(raw_value)

        # 연결 토큰
        if slot_name in {"지번연결", "건물번호연결"}:
            return {
                "slot": slot_name,
                "raw": "",
                "surface": raw_value,
                "base_surface": "",
                "normalized": "",
            }

        # 지역 / 일반 슬롯
        if slot_name in {"시도명", "시군구명", "법정읍면동명", "리", "도로명",
                         "건물본번", "건물부번", "지번본번", "지번부번",
                         "산여부", "지하여부"}:
            base_surface = raw_value
            normalized = raw_value

        elif slot_name == "건물명":
            base_surface = vary_hyphenated_text(raw_value, rng)
            normalized = base_surface if "-" in raw_value else raw_value

        elif slot_name == "동명칭":
            hyphen_base = vary_hyphenated_text(raw_value, rng) if "-" in raw_value else raw_value
            base_surface = make_dong_surface(hyphen_base, rng)
            normalized = base_surface if "-" in raw_value else raw_value

        elif slot_name == "층명칭":
            base_surface = make_floor_surface(raw_value, rng)
            normalized = raw_value

        elif slot_name == "호명칭":
            raw_ho_suffix = slot.get("호접미사명칭", "")
            hyphen_base = vary_hyphenated_text(raw_value, rng) if "-" in raw_value else raw_value
            base_surface = make_ho_surface(hyphen_base, rng, raw_ho_suffix=raw_ho_suffix)
            normalized = base_surface if "-" in raw_value else raw_value

        else:
            base_surface = raw_value
            normalized = raw_value

        final_surface = attach_particle(base_surface, slot_name, rng)

        return {
            "slot": slot_name,
            "raw": raw_value,
            "surface": clean(final_surface),
            "base_surface": clean(base_surface),
            "normalized": clean(normalized),
        }

    candidates: List[List[Dict[str, str]]] = []

    # -------------------------
    # 지역 단독 / 지역 짧게
    # -------------------------
    if slot.get("시도명"):
        candidates.append([make_comp("시도명", slot["시도명"])])

    if slot.get("시군구명"):
        candidates.append([make_comp("시군구명", slot["시군구명"])])

    if slot.get("법정읍면동명"):
        candidates.append([make_comp("법정읍면동명", slot["법정읍면동명"])])

    if slot.get("리"):
        candidates.append([make_comp("리", slot["리"])])

    if slot.get("시도명") and slot.get("시군구명"):
        candidates.append([
            make_comp("시도명", slot["시도명"]),
            make_comp("시군구명", slot["시군구명"]),
        ])

    if slot.get("시군구명") and slot.get("법정읍면동명"):
        candidates.append([
            make_comp("시군구명", slot["시군구명"]),
            make_comp("법정읍면동명", slot["법정읍면동명"]),
        ])

    if slot.get("법정읍면동명") and slot.get("리"):
        candidates.append([
            make_comp("법정읍면동명", slot["법정읍면동명"]),
            make_comp("리", slot["리"]),
        ])

    # -------------------------
    # 도로명형
    # -------------------------
    if addr_type == "road":
        road_core: List[Dict[str, str]] = []
        if slot.get("도로명"):
            road_core.append(make_comp("도로명", slot["도로명"]))
        if slot.get("건물본번"):
            road_core.append(make_comp("건물본번", slot["건물본번"]))
        if slot.get("건물부번"):
            road_core.append(make_comp("건물번호연결", sample_number_connector(rng)))
            road_core.append(make_comp("건물부번", slot["건물부번"]))

        if slot.get("도로명"):
            candidates.append([make_comp("도로명", slot["도로명"])])

        if road_core:
            candidates.append(road_core)

        if slot.get("법정읍면동명") and road_core:
            candidates.append([make_comp("법정읍면동명", slot["법정읍면동명"])] + road_core)

        if slot.get("시군구명") and road_core:
            candidates.append([make_comp("시군구명", slot["시군구명"])] + road_core)

        if slot.get("건물명"):
            candidates.append([make_comp("건물명", slot["건물명"])])

        if slot.get("건물명") and slot.get("도로명"):
            candidates.append([
                make_comp("건물명", slot["건물명"]),
                make_comp("도로명", slot["도로명"]),
            ])

        if slot.get("건물명") and slot.get("법정읍면동명"):
            candidates.append([
                make_comp("법정읍면동명", slot["법정읍면동명"]),
                make_comp("건물명", slot["건물명"]),
            ])

    # -------------------------
    # 지번형
    # -------------------------
    elif addr_type == "jibun":
        jibun_core: List[Dict[str, str]] = []
        if slot.get("산여부"):
            jibun_core.append(make_comp("산여부", slot["산여부"]))
        if slot.get("지번본번"):
            jibun_core.append(make_comp("지번본번", slot["지번본번"]))
        if slot.get("지번부번"):
            jibun_core.append(make_comp("지번연결", sample_number_connector(rng)))
            jibun_core.append(make_comp("지번부번", slot["지번부번"]))

        if slot.get("지번본번"):
            base = [make_comp("지번본번", slot["지번본번"])]
            if slot.get("지번부번") and rng.random() < 0.7:
                base += [
                    make_comp("지번연결", sample_number_connector(rng)),
                    make_comp("지번부번", slot["지번부번"]),
                ]
            candidates.append(base)

        if jibun_core:
            candidates.append(jibun_core)

        if slot.get("법정읍면동명") and jibun_core:
            candidates.append([make_comp("법정읍면동명", slot["법정읍면동명"])] + jibun_core)

        if slot.get("시군구명") and jibun_core:
            candidates.append([make_comp("시군구명", slot["시군구명"])] + jibun_core)

        if slot.get("건물명"):
            candidates.append([make_comp("건물명", slot["건물명"])])

        if slot.get("건물명") and slot.get("법정읍면동명"):
            candidates.append([
                make_comp("법정읍면동명", slot["법정읍면동명"]),
                make_comp("건물명", slot["건물명"]),
            ])

    # 중복 제거
    dedup = []
    seen = set()
    for cand in candidates:
        sig = tuple((c["slot"], c["surface"], c["normalized"]) for c in cand)
        if sig not in seen:
            seen.add(sig)
            dedup.append(cand)

    return dedup
def limit_visible_components(
    components: List[Dict[str, str]],
    max_visible: int = 3,
) -> List[Dict[str, str]]:
    """
    보이는 component 기준 최대 개수 제한.
    연결 토큰(지번연결/건물번호연결)은 개수에서 제외.
    """
    out = []
    visible = 0

    for comp in components:
        out.append(comp)
        if comp["slot"] not in {"지번연결", "건물번호연결"}:
            visible += 1
        if visible >= max_visible:
            break

    # 마지막이 연결 토큰이면 제거
    while out and out[-1]["slot"] in {"지번연결", "건물번호연결"}:
        out.pop()

    out = cleanup_connector_components(out)
    return out

def compose_shallow_address_from_db_row(
    row: Dict[str, Any],
    rng: random.Random,
    max_visible_components: int = 4,
) -> Dict[str, Any]:
    """
    문장 내 다중 ADDRESS 삽입용 짧은 주소 생성기.
    - 도로명/지번 수준까지만
    - 최대 4개 정도의 component
    - 너무 긴 full address 생성 안 함
    """
    used = row.get("used_key_values", {})
    raw_slot_values = normalize_used_keys(used)
    addr_type = detect_address_type(raw_slot_values)

    candidates = build_shallow_component_candidates(raw_slot_values, addr_type, rng)
    if not candidates:
        return {
            "text": "",
            "address_type": addr_type,
            "components": [],
            "normalized_slot_values": {},
            "raw_slot_values": raw_slot_values,
        }

    components = rng.choice(candidates)
    components = limit_visible_components(components, max_visible=max_visible_components)

    text = compose_text_from_components(components)
    normalized_slot_values = build_normalized_slot_values(components)

    return {
        "text": text,
        "address_type": addr_type,
        "components": components,
        "normalized_slot_values": normalized_slot_values,
        "raw_slot_values": raw_slot_values,
    }


def compose_standard_address_from_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    STT 노이즈, 조사, Dropout 등 랜덤 요소를 모두 배제하고
    DB에 있는 값을 바탕으로 가장 정석적이고 깔끔한 풀(Full) 주소를 생성합니다.
    """
    used = row.get("used_key_values", {})
    slot = normalize_used_keys(used)
    addr_type = detect_address_type(slot)

    components = []

    def add(slot_name: str, surface: str):
        if not clean(surface):
            return
        components.append({
            "slot": slot_name,
            "raw": slot.get(slot_name, ""),
            "surface": clean(surface),
            "base_surface": clean(surface),
            "normalized": clean(surface),
        })

    # 1. 행정구역 (순서 보장)
    for s in ["시도명", "시군구명", "법정읍면동명", "리"]:
        if slot.get(s):
            add(s, slot[s])

    # 2. 도로명 / 지번 (정석적인 하이픈 연결 적용)
    if addr_type == "road":
        if slot.get("지하여부") == "1":  # DB 값 체계에 맞게 수정 가능
            add("지하여부", "지하")
            
        if slot.get("도로명"):
            add("도로명", slot["도로명"])
            
        if slot.get("건물본번"):
            add("건물본번", slot["건물본번"])
            
        if slot.get("건물부번"):
            components.append({
                "slot": "건물번호연결",
                "raw": "",
                "surface": "-",
                "base_surface": "-",
                "normalized": "",
            })
            add("건물부번", slot["건물부번"])
            
    else:  # jibun or unknown
        if slot.get("산여부") == "1":
            add("산여부", "산")
            
        if slot.get("지번본번"):
            add("지번본번", slot["지번본번"])
            
        if slot.get("지번부번"):
            components.append({
                "slot": "지번연결",
                "raw": "",
                "surface": "-",
                "base_surface": "-",
                "normalized": "",
            })
            add("지번부번", slot["지번부번"])

    # 3. 상세주소 (건물명 및 동/층/호 명확화)
    if slot.get("건물명"):
        add("건물명", slot["건물명"])

    if slot.get("동명칭"):
        dong = slot["동명칭"]
        if not dong.endswith("동"):
            dong += "동"
        add("동명칭", dong)

    if slot.get("층명칭"):
        floor = slot["층명칭"]
        # 지하 층수인 경우 'B1층' 혹은 '지하1층' 처리 규칙에 따라 조정 가능
        if not floor.endswith("층") and not floor.endswith("지하"):
            floor += "층"
        add("층명칭", floor)

    if slot.get("호명칭"):
        ho = slot["호명칭"]
        suffix = slot.get("호접미사명칭", "")
        # 접미사가 따로 있으면 붙이고, 아무것도 없으면 '호'를 붙임
        if suffix and suffix not in ho:
            ho += suffix
        elif not suffix and not ho.endswith("호"):
            ho += "호"
        add("호명칭", ho)

    # 4. Text 조립 (본번-부번 하이픈 띄어쓰기 제거 처리)
    text = compose_text_from_components(components)
    
    # 하이픈(-) 앞뒤 공백 제거 (예: "12- 34" -> "12-34")
    text = re.sub(r'\s*-\s*', '-', text)

    # 산/지하 같은 접두어는 띄어쓰기 없이 바로 숫자에 붙이는 것이 일반적 (예: "산 15" -> "산15")
    text = re.sub(r'산 (\d)', r'산\1', text)
    text = re.sub(r'지하 (\d)', r'지하\1', text)

    normalized_slot_values = build_normalized_slot_values(components)

    return {
        "text": text,
        "components": components,
        "normalized_slot_values": normalized_slot_values,
        "raw_slot_values": slot,
        "address_type": addr_type,
    }
