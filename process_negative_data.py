import copy
import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


# =========================================================
# Path config
# =========================================================

INPUT_PATH = (
    ADDRESS_DATA_DIR
    / "final_negative_dataset_addr0.json"
)

OUTPUT_PATH = (
    ADDRESS_DATA_DIR
    / "final_negative_dataset_addr0_dedup_thinned.json"
)

REPORT_PATH = (
    ADDRESS_DATA_DIR
    / "final_negative_dataset_addr0_dedup_thinned_report.txt"
)

SOURCE_KEY = "aicc"
DATASET_KEY = "negative"


# =========================================================
# Dedupe config
# =========================================================

# speaker까지 비교할지 여부.
# True면 (speaker, text) 기준이라 더 보수적.
# False면 text만 기준이라 더 공격적으로 제거.
USE_SPEAKER_IN_KEY = True

# overlap thinning 기준.
# 예:
#   last kept: [1,2,3,4,5,6]
#   candidate: [4,5,6,7,8,9]
#   overlap = [4,5,6] 길이 3
#
# MIN_OVERLAP_TURNS=3이면 candidate 제거.
MIN_OVERLAP_TURNS = 2


# =========================================================
# IO
# =========================================================

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_samples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data[SOURCE_KEY][DATASET_KEY]


def rebuild_data(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    return {
        SOURCE_KEY: {
            DATASET_KEY: samples
        }
    }


# =========================================================
# Sequence key
# =========================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def turn_to_key(turn: Dict[str, Any]) -> str:
    text = normalize_text(turn.get("text", ""))

    if USE_SPEAKER_IN_KEY:
        speaker = normalize_text(turn.get("speaker", ""))
        return f"{speaker}|||{text}"

    return text


def sample_to_seq(sample: Dict[str, Any]) -> Tuple[str, ...]:
    dialogue = sample.get("dialogue", [])

    if not isinstance(dialogue, list):
        return tuple()

    seq = []

    for turn in dialogue:
        if not isinstance(turn, dict):
            continue

        key = turn_to_key(turn)
        if not key.strip("|"):
            continue

        seq.append(key)

    return tuple(seq)


# =========================================================
# 1. Exact duplicate 제거
# =========================================================

def remove_exact_duplicates(samples: List[Dict[str, Any]]):
    """
    완전히 같은 dialogue sequence는 첫 번째 것만 유지.
    """
    seen = {}
    kept = []
    removed = []

    for idx, sample in enumerate(samples):
        seq = sample_to_seq(sample)

        if not seq:
            removed.append({
                "reason": "empty_sequence",
                "idx": idx,
                "dialogue_len": 0,
                "text": sample.get("text", ""),
            })
            continue

        if seq in seen:
            removed.append({
                "reason": "exact_duplicate",
                "idx": idx,
                "kept_idx": seen[seq],
                "dialogue_len": len(seq),
                "text": sample.get("text", ""),
            })
            continue

        seen[seq] = idx

        new_sample = copy.deepcopy(sample)
        new_sample["_original_order"] = idx
        kept.append(new_sample)

    return kept, removed


# =========================================================
# 2. 연속 부분수열 포함 제거
# =========================================================

def contiguous_subsequences(seq: Tuple[str, ...]):
    """
    seq의 모든 연속 부분수열 생성.
    자기 자신은 제외.
    """
    n = len(seq)

    for length in range(1, n):
        for start in range(0, n - length + 1):
            yield seq[start:start + length]


def remove_contained_sequences(samples: List[Dict[str, Any]]):
    """
    짧은 dialogue sequence가 더 긴 sequence 안에 연속으로 포함되면 제거.

    예:
      [0,1]       제거
      [0,1,2]     제거
      [0,1,2,3]   유지
    """
    records = []

    for idx, sample in enumerate(samples):
        seq = sample_to_seq(sample)
        records.append({
            "idx": idx,
            "seq": seq,
            "seq_len": len(seq),
            "sample": sample,
            "original_order": sample.get("_original_order", idx),
        })

    # 긴 sequence를 먼저 보면서, 그 안의 모든 subsequence를 index에 등록한다.
    records.sort(key=lambda r: (-r["seq_len"], r["original_order"]))

    subseq_to_parent = {}
    kept_records = []
    removed = []

    for rec in records:
        seq = rec["seq"]

        if seq in subseq_to_parent:
            parent = subseq_to_parent[seq]

            removed.append({
                "reason": "contained_in_longer_sequence",
                "idx_after_exact": rec["idx"],
                "original_order": rec["original_order"],
                "dialogue_len": rec["seq_len"],
                "parent_original_order": parent["original_order"],
                "parent_dialogue_len": parent["seq_len"],
                "text": rec["sample"].get("text", ""),
            })
            continue

        kept_records.append(rec)

        # 현재 kept sequence의 모든 연속 부분수열을 등록.
        # 이후 짧은 sample이 나오면 여기 걸려서 제거된다.
        for sub in contiguous_subsequences(seq):
            if sub not in subseq_to_parent:
                subseq_to_parent[sub] = rec

    kept_samples = [copy.deepcopy(r["sample"]) for r in kept_records]

    # 원래 등장 순서로 복원
    kept_samples.sort(key=lambda s: s.get("_original_order", 10**18))

    return kept_samples, removed


# =========================================================
# 3. Overlap window thinning
# =========================================================

def suffix_prefix_overlap_len(
    left: Tuple[str, ...],
    right: Tuple[str, ...],
) -> int:
    """
    left의 suffix와 right의 prefix가 같은 최대 길이 반환.

    예:
      left  = [1,2,3,4,5,6]
      right = [4,5,6,7,8,9]
      overlap = [4,5,6]
      return 3
    """
    max_len = min(len(left), len(right))

    for length in range(max_len, 0, -1):
        if left[-length:] == right[:length]:
            return length

    return 0


def thin_overlap_windows(samples: List[Dict[str, Any]], min_overlap_turns: int = 3):
    """
    원래 순서대로 보면서, 마지막으로 유지한 sample과 overlap이 큰 candidate를 제거한다.

    예:
      keep:      [1,2,3,4,5,6]
      candidate: [4,5,6,7,8,9]
      overlap 3 이상이면 candidate 제거

      다음 candidate:
      [7,8,9,10,11,12]
      마지막 유지 sample은 여전히 [1,2,3,4,5,6]
      overlap 없음
      => 유지
    """
    kept = []
    removed = []

    for idx, sample in enumerate(samples):
        seq = sample_to_seq(sample)

        if not seq:
            removed.append({
                "reason": "empty_sequence_after_contained",
                "idx_after_contained": idx,
                "original_order": sample.get("_original_order"),
                "text": sample.get("text", ""),
            })
            continue

        if not kept:
            kept.append(copy.deepcopy(sample))
            continue

        last_kept = kept[-1]
        last_seq = sample_to_seq(last_kept)

        overlap_len = suffix_prefix_overlap_len(last_seq, seq)

        if overlap_len >= min_overlap_turns:
            removed.append({
                "reason": "overlap_window_thinning",
                "idx_after_contained": idx,
                "original_order": sample.get("_original_order"),
                "last_kept_original_order": last_kept.get("_original_order"),
                "dialogue_len": len(seq),
                "last_kept_dialogue_len": len(last_seq),
                "overlap_len": overlap_len,
                "text": sample.get("text", ""),
            })
            continue

        kept.append(copy.deepcopy(sample))

    return kept, removed


# =========================================================
# Cleanup / report
# =========================================================

def cleanup_internal_keys(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []

    for sample in samples:
        item = copy.deepcopy(sample)
        item.pop("_original_order", None)
        cleaned.append(item)

    return cleaned


def build_report(
    before_count: int,
    after_exact_count: int,
    after_contained_count: int,
    final_count: int,
    exact_removed: List[Dict[str, Any]],
    contained_removed: List[Dict[str, Any]],
    overlap_removed: List[Dict[str, Any]],
) -> str:
    lines = []

    all_removed = exact_removed + contained_removed + overlap_removed

    lines.append("=" * 100)
    lines.append("NEGATIVE FINAL DEDUPE / THINNING REPORT")
    lines.append("=" * 100)
    lines.append(f"input_path: {INPUT_PATH}")
    lines.append(f"output_path: {OUTPUT_PATH}")
    lines.append("")
    lines.append(f"before_count: {before_count}")
    lines.append(f"after_exact_duplicate_count: {after_exact_count}")
    lines.append(f"after_contained_sequence_count: {after_contained_count}")
    lines.append(f"final_count: {final_count}")
    lines.append("")
    lines.append(f"exact_duplicate_removed: {len(exact_removed)}")
    lines.append(f"contained_sequence_removed: {len(contained_removed)}")
    lines.append(f"overlap_window_removed: {len(overlap_removed)}")
    lines.append(f"total_removed: {before_count - final_count}")
    lines.append("")
    lines.append(f"USE_SPEAKER_IN_KEY: {USE_SPEAKER_IN_KEY}")
    lines.append(f"MIN_OVERLAP_TURNS: {MIN_OVERLAP_TURNS}")
    lines.append("")

    reason_counter = Counter(item["reason"] for item in all_removed)
    lines.append("[removed reason counts]")
    for reason, count in reason_counter.items():
        lines.append(f"- {reason}: {count}")

    lines.append("")
    lines.append("[contained removed examples]")
    for item in contained_removed[:20]:
        lines.append("-" * 100)
        lines.append(
            f"original_order={item.get('original_order')}, "
            f"dialogue_len={item.get('dialogue_len')}, "
            f"parent_original_order={item.get('parent_original_order')}, "
            f"parent_dialogue_len={item.get('parent_dialogue_len')}"
        )
        lines.append(str(item.get("text", ""))[:1000])

    lines.append("")
    lines.append("[overlap removed examples]")
    for item in overlap_removed[:20]:
        lines.append("-" * 100)
        lines.append(
            f"original_order={item.get('original_order')}, "
            f"last_kept_original_order={item.get('last_kept_original_order')}, "
            f"dialogue_len={item.get('dialogue_len')}, "
            f"last_kept_dialogue_len={item.get('last_kept_dialogue_len')}, "
            f"overlap_len={item.get('overlap_len')}"
        )
        lines.append(str(item.get("text", ""))[:1000])

    return "\n".join(lines)


def main():
    data = read_json(INPUT_PATH)
    samples = get_samples(data)

    before_count = len(samples)

    exact_kept, exact_removed = remove_exact_duplicates(samples)
    after_exact_count = len(exact_kept)

    contained_kept, contained_removed = remove_contained_sequences(exact_kept)
    after_contained_count = len(contained_kept)

    overlap_kept, overlap_removed = thin_overlap_windows(
        contained_kept,
        min_overlap_turns=MIN_OVERLAP_TURNS,
    )

    final_samples = cleanup_internal_keys(overlap_kept)
    final_data = rebuild_data(final_samples)

    write_json(final_data, OUTPUT_PATH)

    report = build_report(
        before_count=before_count,
        after_exact_count=after_exact_count,
        after_contained_count=after_contained_count,
        final_count=len(final_samples),
        exact_removed=exact_removed,
        contained_removed=contained_removed,
        overlap_removed=overlap_removed,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"saved: {OUTPUT_PATH}")
    print(f"saved report: {REPORT_PATH}")
    print(f"before_count: {before_count}")
    print(f"after_exact_duplicate_count: {after_exact_count}")
    print(f"after_contained_sequence_count: {after_contained_count}")
    print(f"final_count: {len(final_samples)}")
    print(f"removed_count: {before_count - len(final_samples)}")


if __name__ == "__main__":
    main()