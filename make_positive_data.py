import copy
import json
import random
import time
import os
from typing import Any, Dict, List

from tqdm import tqdm

from paths import TEMPLATE_DIR, generated_data_dir

from make_address_insertion.make_insertion_address import (
    compose_address_from_db_row,
    compose_full_address_from_db_row,
    build_split_address_pair,
    compose_shallow_address_from_db_row,
)

from make_address_insertion.insert_address_multi import (
    insert_single_address,
    insert_double_address,
    insert_split_address,
)

from make_address_insertion.insert_address_by_plan import (
    insert_triple_address,
    insert_quadra_address,
)

from make_address_insertion.make_address_text3 import (
    sido,
    read_json,
    make_road,
    make_jibun,
)


REFERS = ["aicc", "gpt", "qwen3"]
ADD_TYPES = ["single", "double", "split"]

DATA_IDX = 2
SEED = 2 #7

ONELINE_REPEAT_PER_TEMPLATE = 1000
TRIPLE_REPEAT_PER_TEMPLATE = 3
QUADRA_REPEAT_PER_TEMPLATE = 3

TRIPLE_GROUPS = ["A", "B", "C"]
QUADRA_GROUPS = ["A", "B", "C", "D"]

POP_WEIGHTS = {
    "busan": 3.3,
    "chungbuk": 1.6,
    "chungnam": 2.1,
    "daegu": 2.4,
    "daejeon": 1.45,
    "gangwon": 1.5,
    "gwangju": 1.45,
    "gyeongbuk": 2.55,
    "gyeongnam": 3.25,
    "gyunggi": 13.7,
    "incheon": 3.0,
    "jeju": 0.7,
    "jeonbuk": 1.75,
    "jeonnam": 1.8,
    "sejong": 0.4,
    "seoul": 9.4,
    "ulsan": 1.1,
}

ROAD_RATIO = 0.95
JIBUN_RATIO = 0.05

SINGLE_GENERAL_RATIO = 0.7
SINGLE_SHALLOW_RATIO = 0.3

DOUBLE_MODE_RATIOS = {
    ("general", "general"): 0.45,
    ("general", "shallow"): 0.3,
    ("shallow", "shallow"): 0.25,
}


def round_counts(total, ratio_map):
    raw = {k: total * v for k, v in ratio_map.items()}
    base = {k: int(v) for k, v in raw.items()}
    remain = total - sum(base.values())

    frac_order = sorted(
        ratio_map.keys(),
        key=lambda k: raw[k] - base[k],
        reverse=True,
    )

    for k in frac_order[:remain]:
        base[k] += 1

    return base


def make_task_list(counts, rng):
    out = []

    for k, v in counts.items():
        out.extend([k] * v)

    rng.shuffle(out)
    return out


def weighted_region_counts(total, weights):
    regions = list(weights.keys())
    n = len(regions)

    if total <= 0:
        return {r: 0 for r in regions}

    result = {r: 0 for r in regions}

    if total >= n:
        for r in regions:
            result[r] = 1
        remain = total - n
    else:
        remain = total
        ordered = sorted(regions, key=lambda x: weights[x], reverse=True)
        for r in ordered[:total]:
            result[r] = 1
        return result

    if remain > 0:
        weight_sum = sum(weights.values())
        raw = {r: remain * weights[r] / weight_sum for r in regions}
        base = {r: int(raw[r]) for r in regions}

        for r in regions:
            result[r] += base[r]

        leftover = remain - sum(base.values())
        frac_order = sorted(regions, key=lambda r: raw[r] - base[r], reverse=True)

        for r in frac_order[:leftover]:
            result[r] += 1

    return result


def pick_region(region_remain, rng):
    candidates = [r for r, c in region_remain.items() if c > 0]

    if not candidates:
        return None

    weights = [region_remain[r] for r in candidates]
    region = rng.choices(candidates, weights=weights, k=1)[0]
    region_remain[region] -= 1

    return region


class AddressPool:
    def __init__(self, road_db, jibun_db, rng):
        self.rng = rng
        self.db = {
            "road": road_db,
            "jibun": jibun_db,
        }
        self.orders = {
            "road": {},
            "jibun": {},
        }
        self.ptrs = {
            "road": {},
            "jibun": {},
        }
        self.used = set()

        for source in ["road", "jibun"]:
            for region, rows in self.db[source].items():
                idxs = list(range(len(rows)))
                rng.shuffle(idxs)
                self.orders[source][region] = idxs
                self.ptrs[source][region] = 0

    def pop(self, source, preferred_region=None):
        regions = []

        if preferred_region is not None:
            regions.append(preferred_region)

        others = [r for r in self.db[source].keys() if r != preferred_region]
        self.rng.shuffle(others)
        regions.extend(others)

        for region in regions:
            order = self.orders[source][region]
            ptr = self.ptrs[source][region]

            while ptr < len(order):
                row_idx = order[ptr]
                ptr += 1

                key = (source, region, row_idx)

                if key in self.used:
                    continue

                self.used.add(key)
                self.ptrs[source][region] = ptr

                return self.db[source][region][row_idx]

            self.ptrs[source][region] = ptr

        raise RuntimeError(f"No available address row left in source={source}")


def load_region_dbs():
    road_db = {}
    jibun_db = {}

    for region in sido:
        road_db[region] = make_road(region)
        jibun_db[region] = make_jibun(region)

    return road_db, jibun_db


def build_oneline_list(templates, pool, rng):
    """
    1줄 single 템플릿용:
    - 각 템플릿 ONELINE_REPEAT_PER_TEMPLATE 번 반복
    - single placeholder 대상
    """
    expanded_templates = []

    for template in templates:
        for _ in range(ONELINE_REPEAT_PER_TEMPLATE):
            expanded_templates.append(template)

    n = len(expanded_templates)

    source_counts = round_counts(n, {"road": ROAD_RATIO, "jibun": JIBUN_RATIO})
    source_tasks = make_task_list(source_counts, rng)

    region_remain = {
        source: weighted_region_counts(source_counts[source], POP_WEIGHTS)
        for source in ["road", "jibun"]
    }

    outputs = []

    for template, source in tqdm(
        zip(expanded_templates, source_tasks),
        total=len(expanded_templates),
        desc="build oneline",
    ):
        region = pick_region(region_remain[source], rng)
        row = pool.pop(source=source, preferred_region=region)

        if rng.random() < 0.5:
            addr = compose_address_from_db_row(row, rng)
        else:
            addr = compose_full_address_from_db_row(row, rng)

        single_out = insert_single_address(template, addr)
        outputs.append(single_out)

    return outputs


def build_oneline_dataset(template_path, seed=7):
    """
    text_template_oneline_single.json 기반
    """
    rng = random.Random(seed)

    templates = read_json(template_path)
    road_db, jibun_db = load_region_dbs()
    pool = AddressPool(road_db, jibun_db, rng)

    final_data = build_oneline_list(templates, pool, rng)

    return {
        "gpt": {
            "oneline_single": final_data,
        }
    }


def build_oneline_split_list(templates, pool, rng):
    """
    1줄 split 템플릿용:
    - 각 템플릿 ONELINE_REPEAT_PER_TEMPLATE 번 반복
    - split placeholder 대상
    """
    expanded_templates = []

    for template in templates:
        for _ in range(ONELINE_REPEAT_PER_TEMPLATE):
            expanded_templates.append(template)

    n = len(expanded_templates)

    source_counts = round_counts(n, {"road": ROAD_RATIO, "jibun": JIBUN_RATIO})
    source_tasks = make_task_list(source_counts, rng)

    region_remain = {
        source: weighted_region_counts(source_counts[source], POP_WEIGHTS)
        for source in ["road", "jibun"]
    }

    outputs = []

    for template, source in tqdm(
        zip(expanded_templates, source_tasks),
        total=len(expanded_templates),
        desc="build oneline split",
    ):
        region = pick_region(region_remain[source], rng)
        row = pool.pop(source=source, preferred_region=region)

        out1, out2 = build_split_address_pair(row, rng)
        split_out = insert_split_address(template, out1, out2)
        outputs.append(split_out)

    return outputs


def build_oneline_split_dataset(template_path, seed=7):
    """
    text_template_oneline_split.json 기반
    """
    rng = random.Random(seed)

    templates = read_json(template_path)
    road_db, jibun_db = load_region_dbs()
    pool = AddressPool(road_db, jibun_db, rng)

    final_data = build_oneline_split_list(templates, pool, rng)

    return {
        "gpt": {
            "oneline_split": final_data,
        }
    }


def build_single_list(templates, pool, rng):
    n = len(templates)

    source_counts = round_counts(n, {"road": ROAD_RATIO, "jibun": JIBUN_RATIO})
    source_tasks = make_task_list(source_counts, rng)

    mode_counts = round_counts(
        n,
        {
            "general": SINGLE_GENERAL_RATIO,
            "shallow": SINGLE_SHALLOW_RATIO,
        },
    )
    mode_tasks = make_task_list(mode_counts, rng)

    region_remain = {
        source: weighted_region_counts(source_counts[source], POP_WEIGHTS)
        for source in ["road", "jibun"]
    }

    outputs = []

    for template, source, mode in tqdm(
        zip(templates, source_tasks, mode_tasks),
        total=len(templates),
        desc="build single",
    ):
        region = pick_region(region_remain[source], rng)
        row = pool.pop(source=source, preferred_region=region)

        if mode == "general":
            if rng.random() < 0.5:
                addr = compose_address_from_db_row(row, rng)
            else:
                addr = compose_full_address_from_db_row(row, rng)
        else:
            addr = compose_shallow_address_from_db_row(row, rng)

        single_out = insert_single_address(template, addr)
        outputs.append(single_out)

    return outputs


def build_double_list(templates, pool, rng):
    n = len(templates)

    total_slots = 2 * n

    source_counts = round_counts(
        total_slots,
        {
            "road": ROAD_RATIO,
            "jibun": JIBUN_RATIO,
        },
    )
    source_tasks = make_task_list(source_counts, rng)

    combo_counts = round_counts(n, DOUBLE_MODE_RATIOS)
    combo_tasks = make_task_list(combo_counts, rng)

    region_remain = {
        source: weighted_region_counts(source_counts[source], POP_WEIGHTS)
        for source in ["road", "jibun"]
    }

    outputs = []

    for i, template in enumerate(tqdm(templates, desc="build double")):
        source_a = source_tasks[2 * i]
        source_b = source_tasks[2 * i + 1]

        mode_a, mode_b = combo_tasks[i]

        if (mode_a, mode_b) == ("general", "shallow") and rng.random() < 0.5:
            mode_a, mode_b = mode_b, mode_a

        region_a = pick_region(region_remain[source_a], rng)
        region_b = pick_region(region_remain[source_b], rng)

        row_a = pool.pop(source=source_a, preferred_region=region_a)
        row_b = pool.pop(source=source_b, preferred_region=region_b)

        if mode_a == "general":
            addr_a = compose_address_from_db_row(row_a, rng)
        else:
            addr_a = compose_shallow_address_from_db_row(row_a, rng)

        if mode_b == "general":
            addr_b = compose_address_from_db_row(row_b, rng)
        else:
            addr_b = compose_shallow_address_from_db_row(row_b, rng)

        double_out = insert_double_address(template, addr_a, addr_b)
        outputs.append(double_out)

    return outputs


def build_split_list(templates, pool, rng):
    n = len(templates)

    source_counts = round_counts(n, {"road": ROAD_RATIO, "jibun": JIBUN_RATIO})
    source_tasks = make_task_list(source_counts, rng)

    region_remain = {
        source: weighted_region_counts(source_counts[source], POP_WEIGHTS)
        for source in ["road", "jibun"]
    }

    outputs = []

    for template, source in tqdm(
        zip(templates, source_tasks),
        total=len(templates),
        desc="build split",
    ):
        region = pick_region(region_remain[source], rng)
        row = pool.pop(source=source, preferred_region=region)

        out1, out2 = build_split_address_pair(row, rng)
        split_out = insert_split_address(template, out1, out2)
        outputs.append(split_out)

    return outputs


def build_positive_dataset(template_path, seed=7):
    rng = random.Random(seed)

    templates = read_json(template_path)
    road_db, jibun_db = load_region_dbs()
    pool = AddressPool(road_db, jibun_db, rng)

    final_data = {}

    for refer in REFERS:
        final_data[refer] = {}

        single_templates = templates[refer]["single"]["has_placeholder"]
        double_templates = templates[refer]["double"]["has_placeholder"]
        split_templates = templates[refer]["split"]["has_placeholder"]

        final_data[refer]["single"] = build_single_list(single_templates, pool, rng)
        final_data[refer]["double"] = build_double_list(double_templates, pool, rng)
        final_data[refer]["split"] = build_split_list(split_templates, pool, rng)

    return final_data


def _clone_address_item(address_item, group, form):
    item = copy.deepcopy(address_item)
    item["group"] = group
    item["form"] = form
    return item


def build_address_bundle_by_group(
    group_to_row: Dict[str, Any],
    rng,
) -> Dict[str, Dict[str, Any]]:
    """
    group별 row를 받아서 address_bundle 생성.

    기존 build_triple_address_bundle()은 A/B/C만 하드코딩되어 있었는데,
    이 함수는 A/B/C/D 등 임의 group 목록을 처리할 수 있음.

    output:
    {
        "A": {
            "single": address_item,
            "shallow": address_item,
            "split_1": address_item,
            "split_2": address_item,
        },
        ...
    }
    """
    address_bundle = {}

    for group, row in group_to_row.items():
        addr_single = compose_address_from_db_row(row, rng)
        addr_shallow = compose_shallow_address_from_db_row(row, rng)
        addr_split_1, addr_split_2 = build_split_address_pair(row, rng)

        address_bundle[group] = {
            "single": _clone_address_item(
                addr_single,
                group=group,
                form="single",
            ),
            "shallow": _clone_address_item(
                addr_shallow,
                group=group,
                form="shallow",
            ),
            "split_1": _clone_address_item(
                addr_split_1,
                group=group,
                form="split_1",
            ),
            "split_2": _clone_address_item(
                addr_split_2,
                group=group,
                form="split_2",
            ),
        }

    return address_bundle


def build_round_robin_regions(regions, total, rng):
    """
    total 개수만큼 region을 골고루 순환 배치.
    시작점만 랜덤.
    """
    if total <= 0:
        return []

    regions = list(regions)

    if not regions:
        raise ValueError("regions must not be empty when total > 0")

    start = rng.randrange(len(regions))
    ordered = regions[start:] + regions[:start]

    out = []
    idx = 0

    for _ in range(total):
        out.append(ordered[idx])
        idx += 1

        if idx == len(ordered):
            idx = 0

    return out


def _pop_region_from_round_robin(
    source: str,
    road_regions: List[str],
    jibun_regions: List[str],
    region_ptrs: Dict[str, int],
) -> str:
    """
    source별 round-robin region list에서 region 하나를 꺼냄.
    """
    if source == "road":
        ptr = region_ptrs["road"]

        if ptr >= len(road_regions):
            raise IndexError(
                f"road region pointer out of range: ptr={ptr}, len={len(road_regions)}"
            )

        region = road_regions[ptr]
        region_ptrs["road"] += 1
        return region

    if source == "jibun":
        ptr = region_ptrs["jibun"]

        if ptr >= len(jibun_regions):
            raise IndexError(
                f"jibun region pointer out of range: ptr={ptr}, len={len(jibun_regions)}"
            )

        region = jibun_regions[ptr]
        region_ptrs["jibun"] += 1
        return region

    raise ValueError(f"Unknown source: {source}")


def build_plan_list(
    templates,
    pool,
    rng,
    groups: List[str],
    repeat_per_template: int,
    insert_func,
    desc: str,
):
    """
    triple/quadra 공통 데이터 생성 함수.

    핵심 규칙:
    - group 하나당 base row 하나를 뽑음.
    - 각 group row에서 single / shallow / split_1 / split_2 주소 표현을 모두 생성.
    - template["address_plan"]은 occurrence_idx 순서대로 group/form을 참조.
    - 실제 placeholder 치환은 insert_func에서 수행.

    triple:
      groups = ["A", "B", "C"]

    quadra:
      groups = ["A", "B", "C", "D"]
    """
    if not groups:
        raise ValueError("groups must not be empty")

    if repeat_per_template <= 0:
        raise ValueError("repeat_per_template must be positive")

    expanded_templates = []

    for template in templates:
        for _ in range(repeat_per_template):
            expanded_templates.append(template)

    n = len(expanded_templates)
    group_count = len(groups)
    total_group_slots = group_count * n

    source_counts = round_counts(
        total_group_slots,
        {
            "road": ROAD_RATIO,
            "jibun": JIBUN_RATIO,
        },
    )
    source_tasks = make_task_list(source_counts, rng)

    road_regions = build_round_robin_regions(
        regions=sido,
        total=source_counts["road"],
        rng=rng,
    )
    jibun_regions = build_round_robin_regions(
        regions=sido,
        total=source_counts["jibun"],
        rng=rng,
    )

    region_ptrs = {
        "road": 0,
        "jibun": 0,
    }

    outputs = []

    for i, template in enumerate(tqdm(expanded_templates, desc=desc)):
        group_sources = {}
        group_regions = {}
        group_to_row = {}

        for j, group in enumerate(groups):
            source_idx = group_count * i + j
            source = source_tasks[source_idx]

            region = _pop_region_from_round_robin(
                source=source,
                road_regions=road_regions,
                jibun_regions=jibun_regions,
                region_ptrs=region_ptrs,
            )

            row = pool.pop(source=source, preferred_region=region)

            group_sources[group] = source
            group_regions[group] = region
            group_to_row[group] = row

        address_bundle = build_address_bundle_by_group(
            group_to_row=group_to_row,
            rng=rng,
        )

        out = insert_func(template, address_bundle)

        out["plan_meta"] = {
            "groups": list(groups),
            "group_sources": group_sources,
            "group_regions": group_regions,
        }

        outputs.append(out)

    return outputs


def build_plan_dataset(
    template_path,
    dataset_key: str,
    groups: List[str],
    repeat_per_template: int,
    insert_func,
    seed=7,
):
    """
    triple/quadra 공통 dataset 생성 함수.
    """
    rng = random.Random(seed)

    templates = read_json(template_path)
    road_db, jibun_db = load_region_dbs()
    pool = AddressPool(road_db, jibun_db, rng)

    data = build_plan_list(
        templates=templates,
        pool=pool,
        rng=rng,
        groups=groups,
        repeat_per_template=repeat_per_template,
        insert_func=insert_func,
        desc=f"build {dataset_key}",
    )

    return {
        "aicc": {
            dataset_key: data,
        }
    }


def build_triple_list(templates, pool, rng):
    """
    기존 호출부 호환용 wrapper.
    """
    return build_plan_list(
        templates=templates,
        pool=pool,
        rng=rng,
        groups=TRIPLE_GROUPS,
        repeat_per_template=TRIPLE_REPEAT_PER_TEMPLATE,
        insert_func=insert_triple_address,
        desc="build triple",
    )


def build_quadra_list(templates, pool, rng):
    """
    quadra 생성용 wrapper.
    """
    return build_plan_list(
        templates=templates,
        pool=pool,
        rng=rng,
        groups=QUADRA_GROUPS,
        repeat_per_template=QUADRA_REPEAT_PER_TEMPLATE,
        insert_func=insert_quadra_address,
        desc="build quadra",
    )


def build_triple_dataset(template_path, seed=7):
    return build_plan_dataset(
        template_path=template_path,
        dataset_key="triple",
        groups=TRIPLE_GROUPS,
        repeat_per_template=TRIPLE_REPEAT_PER_TEMPLATE,
        insert_func=insert_triple_address,
        seed=seed,
    )


def build_quadra_dataset(template_path, seed=7):
    return build_plan_dataset(
        template_path=template_path,
        dataset_key="quadra",
        groups=QUADRA_GROUPS,
        repeat_per_template=QUADRA_REPEAT_PER_TEMPLATE,
        insert_func=insert_quadra_address,
        seed=seed,
    )


if __name__ == "__main__":
    start_time = time.time()

    DATA_DIR = generated_data_dir(
        ONELINE_REPEAT_PER_TEMPLATE,
        TRIPLE_REPEAT_PER_TEMPLATE,
        QUADRA_REPEAT_PER_TEMPLATE,
        DATA_IDX,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    template_path = TEMPLATE_DIR / "text_template.json"
    save_path = (
        f"{DATA_DIR}/"
        f"final_positive_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
    )

    final_data = build_positive_dataset(template_path, seed=SEED)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"saved: {save_path}, time: {end_time - start_time}sec")

    start_time = time.time()

    triple_path = TEMPLATE_DIR / "text_template_triple.json"
    triple_save_path = (
        f"{DATA_DIR}/"
        f"final_triple_dataset_{TRIPLE_REPEAT_PER_TEMPLATE}.json"
    )

    triple_data = build_triple_dataset(triple_path, seed=SEED)

    with open(triple_save_path, "w", encoding="utf-8") as f:
        json.dump(triple_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"saved: {triple_save_path}, time: {end_time - start_time}sec")

    start_time = time.time()

    quadra_path = TEMPLATE_DIR / "text_template_quadra.json"
    quadra_save_path = (
        f"{DATA_DIR}/"
        f"final_quadra_dataset_{QUADRA_REPEAT_PER_TEMPLATE}.json"
    )

    quadra_data = build_quadra_dataset(quadra_path, seed=SEED)

    with open(quadra_save_path, "w", encoding="utf-8") as f:
        json.dump(quadra_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"saved: {quadra_save_path}, time: {end_time - start_time}sec")

    start_time = time.time()

    oneline_template_path = (
        TEMPLATE_DIR / "text_template_oneline_single.json"
    )
    oneline_save_path = (
        f"{DATA_DIR}/"
        f"final_oneline_single_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
    )

    oneline_data = build_oneline_dataset(oneline_template_path, seed=SEED)

    with open(oneline_save_path, "w", encoding="utf-8") as f:
        json.dump(oneline_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"saved: {oneline_save_path}, time: {end_time - start_time}sec")

    start_time = time.time()

    oneline_split_template_path = (
        TEMPLATE_DIR / "text_template_oneline_split.json"
    )
    oneline_split_save_path = (
        f"{DATA_DIR}/"
        f"final_oneline_split_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
    )

    oneline_split_data = build_oneline_split_dataset(oneline_split_template_path, seed=SEED)

    with open(oneline_split_save_path, "w", encoding="utf-8") as f:
        json.dump(oneline_split_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"saved: {oneline_split_save_path}, time: {end_time - start_time}sec")