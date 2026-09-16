import json
import os
import pickle
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer


# =========================================================
# config
# =========================================================

bert = "klue"
# bert = "uplus"

if bert == "klue":
    bert_path = "klue/roberta-large"
elif bert == "uplus":
    bert_path = "/data/private/address_bot/models/ixi-RoBERTa-base_250203"
else:
    raise ValueError(f"Invalid BERT name: {bert}")

DATA_IDX = 3
ONELINE_REPEAT_PER_TEMPLATE = 2000
TRIPLE_REPEAT_PER_TEMPLATE = 4
QUADRA_REPEAT_PER_TEMPLATE = 4


DATA_DIR = f"/data/private/address_bot_3/make_address_insertion/data/oneline{ONELINE_REPEAT_PER_TEMPLATE}_triple{TRIPLE_REPEAT_PER_TEMPLATE}_quadra{QUADRA_REPEAT_PER_TEMPLATE}_dataidx{DATA_IDX}"
NEGATIV_DIR = "/data/private/address_bot_3/make_address_insertion/data"


NEGATIVE_GROUPED_PATH = (
    f"{NEGATIV_DIR}/final_negative_dataset_addr0_dedup_thinned.json"
)


POSITIVE_GROUPED_PATH = (
    f"{DATA_DIR}/"
    f"final_positive_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
)

TRIPLE_GROUPED_PATH = (
    f"{DATA_DIR}/"
    f"final_triple_dataset_{TRIPLE_REPEAT_PER_TEMPLATE}.json"
)

QUADRA_GROUPED_PATH = (
    f"{DATA_DIR}/"
    f"final_quadra_dataset_{QUADRA_REPEAT_PER_TEMPLATE}.json"
)

ONELINE_SINGLE_GROUPED_PATH = (
    f"{DATA_DIR}/"
    f"final_oneline_single_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
)

ONELINE_SPLIT_GROUPED_PATH = (
    f"{DATA_DIR}/"
    f"final_oneline_split_dataset_{ONELINE_REPEAT_PER_TEMPLATE}.json"
)


SAVE_DIR = f"train_data/oneline{ONELINE_REPEAT_PER_TEMPLATE}_triple{TRIPLE_REPEAT_PER_TEMPLATE}_quadra{QUADRA_REPEAT_PER_TEMPLATE}_dataidx{DATA_IDX}"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_POSITIVE_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_positive_{ONELINE_REPEAT_PER_TEMPLATE}.pkl"
)

SAVE_NEGATIVE_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_negative_addr0.pkl"
)

SAVE_TRIPLE_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_triple_{TRIPLE_REPEAT_PER_TEMPLATE}.pkl"
)

SAVE_QUADRA_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_quadra_{QUADRA_REPEAT_PER_TEMPLATE}.pkl"
)

SAVE_ONELINE_SINGLE_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_oneline_single_{ONELINE_REPEAT_PER_TEMPLATE}.pkl"
)

SAVE_ONELINE_SPLIT_PATH = (
    f"{SAVE_DIR}/{bert}_bert_token_cls_oneline_split_{ONELINE_REPEAT_PER_TEMPLATE}.pkl"
)


MAX_LENGTH = 512

# 현재 주소 NER slot 수가 15개라고 기대하는 경우
EXPECTED_SLOT_COUNT = 15


# =========================================================
# io
# =========================================================

def read_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# data structure utils
# =========================================================

def iter_samples(grouped: Dict[str, Any]):
    """
    허용 구조:

    1) positive
    grouped[refer][add_type] = [sample, sample, ...]

    예:
    {
        "aicc": {
            "single": [...],
            "double": [...],
            "split": [...]
        },
        "gpt": {...},
        "qwen3": {...}
    }

    2) triple / quadra / negative / oneline
    grouped[source][dataset_key] = [sample, sample, ...]

    예:
    {
        "aicc": {
            "triple": [...]
        }
    }

    yield:
        refer, add_type, sample
    """
    if not isinstance(grouped, dict):
        raise TypeError(f"grouped must be dict, got {type(grouped)}")

    for refer, refer_dict in grouped.items():
        if not isinstance(refer_dict, dict):
            raise TypeError(
                f"grouped[{refer}] value must be dict, got {type(refer_dict)}"
            )

        for add_type, samples in refer_dict.items():
            if not isinstance(samples, list):
                raise TypeError(
                    f"grouped[{refer}][{add_type}] must be list, got {type(samples)}"
                )

            for sample in samples:
                yield refer, add_type, sample


def count_grouped_samples(grouped: Dict[str, Any]) -> int:
    total = 0

    for _, _, _ in iter_samples(grouped):
        total += 1

    return total


# =========================================================
# label utils
# =========================================================

def collect_all_slot_names_from_grouped(grouped: Dict[str, Any]) -> List[str]:
    slot_names = set()

    for _, _, sample in iter_samples(grouped):
        for span in sample.get("spans", []):
            slot_names.add(span["slot"])

    return sorted(slot_names)


def collect_all_slot_names_from_grouped_list(
    grouped_list: List[Dict[str, Any]],
) -> List[str]:
    slot_names = set()

    for grouped in grouped_list:
        slot_names.update(collect_all_slot_names_from_grouped(grouped))

    return sorted(slot_names)


def build_label_list(
    slot_names: List[str],
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    slot_names = sorted(set(slot_names))

    label_list = ["O"]

    for slot in slot_names:
        label_list.append(f"B-{slot}")
        label_list.append(f"I-{slot}")

    label2id = {
        label: i
        for i, label in enumerate(label_list)
    }

    id2label = {
        i: label
        for label, i in label2id.items()
    }

    return label_list, label2id, id2label


# =========================================================
# span / token align
# =========================================================

def validate_spans(spans: List[Dict[str, Any]], text: str) -> None:
    prev_end = -1

    sorted_spans = sorted(
        spans,
        key=lambda x: (x["start_idx"], x["end_idx"]),
    )

    for i, span in enumerate(sorted_spans):
        start = span["start_idx"]
        end = span["end_idx"]
        slot = span["slot"]

        if not (0 <= start < end <= len(text)):
            raise ValueError(
                f"Invalid span[{i}] | "
                f"slot={slot}, start={start}, end={end}, text_len={len(text)}"
            )

        if start < prev_end:
            raise ValueError(
                f"Overlapping spans | "
                f"slot={slot}, start={start}, end={end}, prev_end={prev_end}"
            )

        prev_end = end


def char_to_token_bio_labels(
    text: str,
    spans: List[Dict[str, Any]],
    offsets: List[Tuple[int, int]],
    sequence_ids: List[Any],
    label2id: Dict[str, int],
    target_sequence_id: int = 0,
) -> List[int]:
    """
    single text input 기준 BIO label 생성.

    tokenizer(text, ...) 형태에서는 본문 token의 sequence_id가 보통 0이고,
    special token은 None이다.

    따라서:
      - seq_id != 0 -> -100
      - seq_id == 0 -> O 또는 BIO label
    """
    validate_spans(spans, text)

    sorted_spans = sorted(
        spans,
        key=lambda x: (x["start_idx"], x["end_idx"]),
    )

    labels: List[int] = []

    for (start, end), seq_id in zip(offsets, sequence_ids):
        if seq_id != target_sequence_id:
            labels.append(-100)
        elif start == end:
            labels.append(-100)
        else:
            labels.append(label2id["O"])

    for span in sorted_spans:
        s = span["start_idx"]
        e = span["end_idx"]
        slot = span["slot"]

        token_indices = []

        for idx, ((start, end), seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != target_sequence_id:
                continue

            if start == end:
                continue

            # span과 token offset이 겹치면 해당 token에 label 부여
            if not (end <= s or start >= e):
                token_indices.append(idx)

        if not token_indices:
            raise ValueError(
                f"No token aligned to span | "
                f"slot={slot}, start={s}, end={e}, text={text}"
            )

        labels[token_indices[0]] = label2id[f"B-{slot}"]

        for idx in token_indices[1:]:
            labels[idx] = label2id[f"I-{slot}"]

    return labels


# =========================================================
# validate BIO
# =========================================================

def validate_bio_label_sequence(
    label_ids: List[int],
    id2label: Dict[int, str],
) -> None:
    prev_label = "O"

    for i, lid in enumerate(label_ids):
        if lid == -100:
            continue

        cur_label = id2label[lid]

        if cur_label == "O":
            prev_label = "O"
            continue

        prefix, slot = cur_label.split("-", 1)

        if prefix == "B":
            prev_label = cur_label
            continue

        if prefix == "I":
            if prev_label == "O":
                raise ValueError(
                    f"Invalid BIO at token {i}: {cur_label} after O"
                )

            prev_prefix, prev_slot = prev_label.split("-", 1)

            if prev_prefix not in {"B", "I"} or prev_slot != slot:
                raise ValueError(
                    f"Invalid BIO at token {i}: {cur_label} after {prev_label}"
                )

            prev_label = cur_label
            continue

        raise ValueError(f"Unknown label format at token {i}: {cur_label}")


# =========================================================
# tokenize single sample
# =========================================================

def tokenize_and_align_sample(
    sample: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    현재 데이터 구조 기준:
      sample["text"]
      sample["spans"]

    더 이상 context_text / target_text pair를 사용하지 않는다.
    """
    if "text" not in sample:
        raise KeyError(f"sample missing 'text': keys={list(sample.keys())}")

    text = sample["text"]

    if not isinstance(text, str):
        raise TypeError(f"sample['text'] must be str, got {type(text)}")

    spans = sample.get("spans", [])

    if spans is None:
        spans = []

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"]
    sequence_ids = encoding.sequence_ids()

    labels = char_to_token_bio_labels(
        text=text,
        spans=spans,
        offsets=offsets,
        sequence_ids=sequence_ids,
        label2id=label2id,
        target_sequence_id=0,
    )

    validate_bio_label_sequence(labels, id2label)

    row = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }

    if "token_type_ids" in encoding:
        row["token_type_ids"] = encoding["token_type_ids"]

    return row


# =========================================================
# grouped -> rows
# =========================================================

def convert_grouped_to_token_classification_data(
    grouped: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    max_length: int = 512,
    dataset_name: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []

    total = count_grouped_samples(grouped)

    for row_idx, (refer, add_type, sample) in enumerate(
        tqdm(
            iter_samples(grouped),
            total=total,
            desc=f"tokenize {dataset_name}".strip(),
        )
    ):
        try:
            row = tokenize_and_align_sample(
                sample=sample,
                tokenizer=tokenizer,
                label2id=label2id,
                id2label=id2label,
                max_length=max_length,
            )

            row["refer"] = refer
            row["add_type"] = add_type

            rows.append(row)

        except Exception as e:
            invalid_rows.append({
                "dataset_name": dataset_name,
                "row_idx": row_idx,
                "refer": refer,
                "add_type": add_type,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "sample": sample,
            })

    if invalid_rows:
        invalid_path = f"{SAVE_DIR}/{bert}_bert_token_cls_invalid_{dataset_name}.json"

        os.makedirs(os.path.dirname(invalid_path), exist_ok=True)

        with open(invalid_path, "w", encoding="utf-8") as f:
            json.dump(invalid_rows, f, ensure_ascii=False, indent=2)

        print(
            f"[WARN] {dataset_name}: invalid rows = {len(invalid_rows)}, "
            f"saved to {invalid_path}"
        )

    return rows


# =========================================================
# stats
# =========================================================

def print_grouped_stats(name: str, grouped: Dict[str, Any]) -> None:
    print(f"\n===== {name} grouped stats =====")

    total = 0

    for refer, refer_dict in grouped.items():
        refer_total = 0

        print(f"[{refer}]")

        for add_type, samples in refer_dict.items():
            n = len(samples)
            refer_total += n
            total += n
            print(f"  {add_type:16s} | {n}")

        print(f"  {'subtotal':16s} | {refer_total}")

    print(f"TOTAL: {total}")


def print_row_stats(name: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n===== {name} tokenized rows =====")
    print(f"num_rows: {len(rows)}")


# =========================================================
# main
# =========================================================

def main():
    # 1) grouped 데이터 로드
    positive_grouped = read_json(POSITIVE_GROUPED_PATH)
    negative_grouped = read_json(NEGATIVE_GROUPED_PATH)
    triple_grouped = read_json(TRIPLE_GROUPED_PATH)
    quadra_grouped = read_json(QUADRA_GROUPED_PATH)
    oneline_single_grouped = read_json(ONELINE_SINGLE_GROUPED_PATH)
    oneline_split_grouped = read_json(ONELINE_SPLIT_GROUPED_PATH)

    print_grouped_stats("positive", positive_grouped)
    print_grouped_stats("negative", negative_grouped)
    print_grouped_stats("triple", triple_grouped)
    print_grouped_stats("quadra", quadra_grouped)
    print_grouped_stats("oneline single", oneline_single_grouped)
    print_grouped_stats("oneline split", oneline_split_grouped)

    # 2) 전체 라벨셋 생성
    all_slot_names = collect_all_slot_names_from_grouped_list(
        [
            positive_grouped,
            negative_grouped,
            triple_grouped,
            quadra_grouped,
            oneline_single_grouped,
            oneline_split_grouped,
        ]
    )

    print("\nall_slot_names:")
    print(all_slot_names)
    print("slot count:", len(all_slot_names))

    if len(all_slot_names) != EXPECTED_SLOT_COUNT:
        raise ValueError(
            f"slot 개수 에러: expected={EXPECTED_SLOT_COUNT}, "
            f"actual={len(all_slot_names)}, slots={all_slot_names}"
        )

    label_list, label2id, id2label = build_label_list(all_slot_names)

    print("\nlabel_list:")
    print(label_list)
    print("label count:", len(label_list))

    # 3) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        bert_path,
        use_fast=True,
    )

    # 4) 각각 token classification data 생성
    positive_rows = convert_grouped_to_token_classification_data(
        grouped=positive_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="positive",
    )

    negative_rows = convert_grouped_to_token_classification_data(
        grouped=negative_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="negative",
    )

    triple_rows = convert_grouped_to_token_classification_data(
        grouped=triple_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="triple",
    )

    quadra_rows = convert_grouped_to_token_classification_data(
        grouped=quadra_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="quadra",
    )

    oneline_single_rows = convert_grouped_to_token_classification_data(
        grouped=oneline_single_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="oneline_single",
    )

    oneline_split_rows = convert_grouped_to_token_classification_data(
        grouped=oneline_split_grouped,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        max_length=MAX_LENGTH,
        dataset_name="oneline_split",
    )

    print_row_stats("positive", positive_rows)
    print_row_stats("negative", negative_rows)
    print_row_stats("triple", triple_rows)
    print_row_stats("quadra", quadra_rows)
    print_row_stats("oneline single", oneline_single_rows)
    print_row_stats("oneline split", oneline_split_rows)

    # 5) 저장
    common_meta = {
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
        "bert_path": bert_path,
        "max_length": MAX_LENGTH,
    }

    save_pkl(
        {
            "data": positive_rows,
            **common_meta,
        },
        SAVE_POSITIVE_PATH,
    )

    save_pkl(
        {
            "data": negative_rows,
            **common_meta,
        },
        SAVE_NEGATIVE_PATH,
    )

    save_pkl(
        {
            "data": triple_rows,
            **common_meta,
        },
        SAVE_TRIPLE_PATH,
    )

    save_pkl(
        {
            "data": quadra_rows,
            **common_meta,
        },
        SAVE_QUADRA_PATH,
    )

    save_pkl(
        {
            "data": oneline_single_rows,
            **common_meta,
        },
        SAVE_ONELINE_SINGLE_PATH,
    )

    save_pkl(
        {
            "data": oneline_split_rows,
            **common_meta,
        },
        SAVE_ONELINE_SPLIT_PATH,
    )

    print("\nSaved:")
    print(SAVE_POSITIVE_PATH)
    print(SAVE_NEGATIVE_PATH)
    print(SAVE_TRIPLE_PATH)
    print(SAVE_QUADRA_PATH)
    print(SAVE_ONELINE_SINGLE_PATH)
    print(SAVE_ONELINE_SPLIT_PATH)


if __name__ == "__main__":
    main()