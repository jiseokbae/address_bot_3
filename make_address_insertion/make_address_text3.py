import json
import re


sido = [
    'busan', 'chungbuk', 'chungnam', 'daegu', 'daejeon', 'gangwon',
    'gwangju', 'gyeongbuk', 'gyeongnam', 'gyunggi', 'incheon',
    'jeju', 'jeonbuk', 'jeonnam', 'sejong', 'seoul', 'ulsan'
]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def clean(v):
    if v is None:
        return ""
    return str(v).strip()


def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_sido_variants(sido: str):
    """
    예:
    서울특별시 -> 서울특별시, 서울, 서울시
    부산광역시 -> 부산광역시, 부산, 부산시
    제주특별자치도 -> 제주특별자치도, 제주, 제주도
    전북특별자치도 -> 전북특별자치도, 전북, 전북도, 전라북도
    """
    sido = clean(sido)
    if not sido:
        return []

    variants = [sido]

    if sido.endswith("특별시"):
        base = sido[:-3]
        variants.extend([base, f"{base}시"])

    elif sido.endswith("광역시"):
        base = sido[:-3]
        variants.extend([base, f"{base}시"])

    elif sido.endswith("특별자치시"):
        base = sido[:-5]
        variants.extend([base, f"{base}시"])

    elif sido.endswith("특별자치도"):
        base = sido[:-5]
        variants.extend([base, f"{base}도"])

        special_map = {
            "제주": ["제주도"],
            "강원": ["강원도"],
            "전북": ["전북도", "전라북도"],
        }
        variants.extend(special_map.get(base, []))

    elif sido.endswith("도"):
        base = sido[:-1]
        variants.append(base)

        short_map = {
            "경상북": ["경북"],
            "경상남": ["경남"],
            "충청북": ["충북"],
            "충청남": ["충남"],
            "전라북": ["전북"],
            "전라남": ["전남"],
        }
        variants.extend(short_map.get(base, []))

    return unique_keep_order(variants)


def parse_building_and_dong(val: str) -> tuple[str, str]:
    new_val = val
    extracted_dong = ""
    
    if val.endswith("동"):
        # 1. 주건축물 / 부속건축물 전체를 동으로 빼기
        # 예: "우성아파트 주건축물제1동" -> 건물명: "우성아파트", 동명칭: "주건축물제1동"
        special_match = re.search(r'((?:주건축물|부속건축물|부건축물).*동)$', val)
        if special_match:
            extracted_dong = special_match.group(1)
            new_val = val[:special_match.start()].strip()
            return new_val, extracted_dong

        # 2. 숫자/영어, 가~하, 음차 패턴 분리
        dong_pattern = re.compile(
            r'('
            r'(?:제)?[a-zA-Z0-9\-]+(?:호)?동|'
            r'[가-하]동|'
            r'(에이|비|씨|시|디|이|에프|지|에이치|아이|제이|케이|엘|엠|엔|오|피|큐|알|에스|티|유|브이|더블유|엑스|와이|제트)동'
            r')$'
        )
        match = dong_pattern.search(val)
        if match:
            extracted_dong = match.group(1)
            new_val = val[:match.start()].strip()
            return new_val, extracted_dong

        # 3. 위 패턴에 안 걸렸지만 띄어쓰기가 있는 경우 (예: 상가동, 관리동 등)
        if " " in val:
            parts = val.rsplit(" ", 1)
            new_val = parts[0].strip()
            extracted_dong = parts[1].strip()
            return new_val, extracted_dong

    return new_val, extracted_dong


def get_building_candidates(item: dict):
    """
    건물명 후보 추출 및 전처리 중복 제거 적용
    우선순위: 시군구용 건물명 > 상세건물명 > 건축물대장 건물명
    """
    out = []

    # 1. 값 추출
    v_sigungu = clean(item.get("시군구용 건물명"))
    v_detail = clean(item.get("상세건물명"))
    v_arch = clean(item.get("건축물대장 건물명"))

    # 2. Candidate 생성 전 중복 제거 (우선순위가 낮은 것은 아예 빈 문자열로 처리)
    if v_detail and v_detail == v_sigungu:
        v_detail = ""
    if v_arch and (v_arch == v_sigungu or v_arch == v_detail):
        v_arch = ""

    # 3. 우선순위대로 파싱 및 out에 추가
    if v_sigungu:
        new_val, ext_dong = parse_building_and_dong(v_sigungu)
        if new_val :
            out.append((new_val, {"시군구용 건물명": v_sigungu}, ext_dong))

    if v_detail:
        new_val, ext_dong = parse_building_and_dong(v_detail)
        if new_val :
            out.append((new_val, {"상세건물명": v_detail}, ext_dong))

    # Rule 1: 건축물대장 건물명은 '동'으로 끝나면 아예 추가하지 않음
    if v_arch:
        if not v_arch.endswith("동"):
            out.append((v_arch, {"건축물대장 건물명": v_arch}, ""))

    if not out:
        return [("", {}, "")]

    # 파싱 결과(예: 동을 떼어내고 남은 건물명)가 동일해질 수 있으므로 최종 중복 제거
    dedup = []
    seen = set()
    for text, used, ext_dong in out:
        sig = (text, ext_dong)
        if sig not in seen:
            seen.add(sig)
            dedup.append((text, used, ext_dong))

    return dedup


def clean_and_split_building_dong(building_text: str, dong_text: str) -> tuple[str, str]:
    """
    건물명(building_text) 안에 동 정보(dong_text)가 중복으로 있으면,
    건물명에서 동 정보를 제거하고 각각 분리하여 반환합니다.
    """
    if not building_text or not dong_text:
        return building_text, dong_text

    b_clean = building_text.strip()
    d_clean = dong_text.strip()

    patterns = [f"{d_clean}동", d_clean]

    new_building_text = b_clean
    for p in patterns:
        if p in new_building_text:
            new_building_text = new_building_text.replace(p, "").strip()
            break

    return new_building_text, d_clean


def get_detail_text_and_used(item: dict, building_text: str = "", extracted_dong: str = ""):
    """
    - 동명칭이 비어있으면 파싱한 extracted_dong 사용
    - 동은 building_text와 중복이면 분리(깎아냄) 처리 (clean_and_split_building_dong 활용)
    """
    used = {}
    parts = []

    original_dong = clean(item.get("동명칭"))
    target_dong = original_dong if original_dong else extracted_dong

    floor = clean(item.get("층명칭"))
    ho = clean(item.get("호명칭"))
    ho_suffix = clean(item.get("호접미사명칭")) or clean(item.get("호전미사명칭"))

    # 여기서 건물명 안의 동명칭 중복 제거가 일어납니다.
    new_building_text, final_dong = clean_and_split_building_dong(building_text, target_dong)

    if final_dong:
        parts.append(final_dong)
        used["동명칭"] = final_dong

    if floor:
        parts.append(floor)
        used["층명칭"] = floor

    if ho:
        ho_text = f"{ho}{ho_suffix}" if ho_suffix else ho
        parts.append(ho_text)
        used["호명칭"] = ho
        if ho_suffix:
            used["호접미사명칭"] = ho_suffix

    return new_building_text, " ".join(parts), used


def get_jibun_text_and_used(item: dict):
    main = clean(item.get("지번본번(번지)"))
    sub = clean(item.get("지번부번(호)"))
    san = clean(item.get("산여부"))
    underground = clean(item.get("지하여부"))

    if not main:
        return "", {}

    used = {}
    prefix_parts = []

    if san == "1":
        prefix_parts.append("산")
        used["산여부"] = "산"

    if underground == "1":
        prefix_parts.append("지하")
        used["지하여부"] = "지하"

    if sub and sub != "0":
        number_text = f"{main}-{sub}"
        used["지번본번(번지)"] = main
        used["지번부번(호)"] = sub
    else:
        number_text = main
        used["지번본번(번지)"] = main

    jibun = " ".join(prefix_parts + [number_text]).strip()
    return jibun, used


def get_road_number_text_and_used(item: dict):
    road = clean(item.get("도로명"))
    main = clean(item.get("건물본번"))
    sub = clean(item.get("건물부번"))
    underground = clean(item.get("지하여부"))

    if not road or not main:
        return "", {}

    used = {
        "도로명": road,
        "건물본번": main,
    }

    prefix_parts = []

    if underground == "1":
        prefix_parts.append("지하")
        used["지하여부"] = "지하"

    if sub and sub != "0":
        number_text = f"{main}-{sub}"
        used["건물부번"] = sub
    else:
        number_text = main

    road_number_text = " ".join([road] + prefix_parts + [number_text]).strip()
    return road_number_text, used


def generate_jibun_addresses(item: dict):
    sido_variants = get_sido_variants(item.get("시도명"))
    if not sido_variants:
        sido_variants = [""]

    sigungu = clean(item.get("시군구명"))
    emd = clean(item.get("법정읍면동명"))
    ri = clean(item.get("리")) or clean(item.get("법정리명"))

    jibun_text, jibun_used = get_jibun_text_and_used(item)
    building_candidates = get_building_candidates(item)
    
    # if not building_candidates:
    #     building_candidates = [("", {}, "")]

    results = []

    for sido_variant in sido_variants:
        for building_text, building_used, ext_dong in building_candidates:
            new_building_text, detail_text, detail_used = get_detail_text_and_used(
                item=item,
                building_text=building_text,
                extracted_dong=ext_dong
            )

            parts = [
                clean(sido_variant),
                sigungu,
                emd,
                ri,
                jibun_text,
                new_building_text,
                detail_text,
            ]
            address_text = " ".join([p for p in parts if p]).strip()

            if not address_text:
                continue

            used_key_values = {}

            if clean(sido_variant):
                used_key_values["시도명"] = sido_variant
            if sigungu:
                used_key_values["시군구명"] = sigungu
            if emd:
                used_key_values["법정읍면동명"] = emd
            if ri:
                if clean(item.get("리")):
                    used_key_values["리"] = ri
                else:
                    used_key_values["법정리명"] = ri

            used_key_values.update(jibun_used)
            if new_building_text:
                for k in building_used.keys():
                    used_key_values[k] = new_building_text
            used_key_values.update(detail_used)

            results.append({
                "address_text": address_text,
                "used_key_values": used_key_values,
            })

    final = []
    seen = set()
    for row in results:
        sig = row["address_text"]
        if sig not in seen:
            seen.add(sig)
            final.append(row)

    return final


def generate_road_addresses(item: dict):
    sido_variants = get_sido_variants(item.get("시도명"))
    if not sido_variants:
        sido_variants = [""]

    sigungu = clean(item.get("시군구명"))
    emd = clean(item.get("법정읍면동명"))
    ri = clean(item.get("리")) or clean(item.get("법정리명"))

    road_number_text, road_used = get_road_number_text_and_used(item)
    if not road_number_text:
        return []

    building_candidates = get_building_candidates(item)
    # if not building_candidates:
    #     building_candidates = [("", {}, "")]

    results = []

    for sido_variant in sido_variants:
        for building_text, building_used, ext_dong in building_candidates:
            new_building_text, detail_text, detail_used = get_detail_text_and_used(
                item=item,
                building_text=building_text,
                extracted_dong=ext_dong
            )

            parts = [
                clean(sido_variant),
                sigungu,
                emd,
                ri,
                road_number_text,
                new_building_text,
                detail_text,
            ]
            address_text = " ".join([p for p in parts if p]).strip()

            if not address_text:
                continue

            used_key_values = {}

            if clean(sido_variant):
                used_key_values["시도명"] = sido_variant
            if sigungu:
                used_key_values["시군구명"] = sigungu
            if emd:
                used_key_values["법정읍면동명"] = emd
            if ri:
                if clean(item.get("리")):
                    used_key_values["리"] = ri
                else:
                    used_key_values["법정리명"] = ri

            used_key_values.update(road_used)
            if new_building_text:
                for k in building_used.keys():
                    used_key_values[k] = new_building_text
            used_key_values.update(detail_used)

            results.append({
                "address_text": address_text,
                "used_key_values": used_key_values,
            })

    final = []
    seen = set()
    for row in results:
        sig = row["address_text"]
        if sig not in seen:
            seen.add(sig)
            final.append(row)

    return final


def make_jibun(region):
    data_path = "/data/private/address_bot/processed/jibun_" + region + ".json"
    data = read_json(data_path)

    out = []
    for item in data:
        out.extend(generate_jibun_addresses(item))
    return out


def make_road(region):
    data_path = "/data/private/address_bot_3/processed/road_" + region + ".json"
    data = read_json(data_path)

    out = []
    for item in data:
        out.extend(generate_road_addresses(item))
    return out


if __name__ == "__main__":
    region = sido[0]

    print("=== ROAD ===")
    print(make_road(region)[:5])