import copy
import json
import time
from typing import Any, Dict, List

from tqdm import tqdm

from paths import (
    ADDRESS_DATA_DIR,
    NEGATIVE_ADDR0_PATH,
    NEGATIVE_ADDR5PLUS0_PATH,
)



BASE_WINDOW_SIZE = 5
STRIDE = 6
INCLUDE_LAST_WINDOW = True


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_angle_tags(text: str) -> str:
    """
    <surface:LABEL> -> surface

    예:
      <이혜정:NAME> -> 이혜정
      <서울특별시:시도명> -> 서울특별시
      <[ADDRESS]:ADDRESS> -> [ADDRESS]
      <[ADDRESS_A1]:ADDRESS> -> [ADDRESS_A1]
    """
    if not isinstance(text, str):
        return ""

    def replace_tag(match):
        inner = match.group(1)

        if ":" in inner:
            return inner.split(":", 1)[0]

        return inner

    return ANGLE_TAG_PATTERN.sub(replace_tag, text)

def pick_text_from_turn(turn: Dict[str, Any]) -> str:
    """
    addr0 turn에서 최종 text로 쓸 필드 선택.

    addr0에 original_text가 있으면 original_text 우선,
    없으면 text 사용.
    """
    if isinstance(turn.get("original_text"), str):
        return turn["original_text"]

    if isinstance(turn.get("text"), str):
        return turn["text"]

    if isinstance(turn.get("content"), str):
        return turn["content"]

    if isinstance(turn.get("tagged_text"), str):
        return clean_angle_tags(turn.get("tagged_text"))

    return ""


def clean_dialogue(
    dialogue: List[Dict[str, Any]],
    orig_idx: int,
) -> List[Dict[str, Any]]:
    """
    최종 sample 안 dialogue 양식을 맞춘다.

    output turn:
    {
        "turn_idx": ...,
        "speaker": ...,
        "text": ...
    }
    """
    if not isinstance(dialogue, list):
        raise TypeError(
            f"orig_idx={orig_idx}: dialogue must be list, got {type(dialogue)}"
        )

    cleaned = []

    for turn_pos, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            raise TypeError(
                f"orig_idx={orig_idx}, turn_pos={turn_pos}: "
                f"turn must be dict, got {type(turn)}"
            )

        text = pick_text_from_turn(turn)
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        if not text:
            continue

        speaker = turn.get("speaker", "")
        if speaker is None:
            speaker = ""

        if not isinstance(speaker, str):
            speaker = str(speaker)

        speaker = speaker.strip()

        cleaned.append({
            "turn_idx": turn.get("turn_idx", turn_pos),
            "speaker": speaker,
            "text": text,
        })

    return cleaned


def build_text_from_dialogue(dialogue: List[Dict[str, Any]]) -> str:
    """
    기존 positive template 쪽과 동일하게 speaker prefix 없이 text만 줄바꿈으로 연결.
    """
    return "\n".join(
        turn["text"].strip()
        for turn in dialogue
        if isinstance(turn.get("text"), str) and turn["text"].strip()
    )


def make_prefix5_then_sliding6_samples(
    dialogue: List[Dict[str, Any]],
    orig_idx: int,
    base_window_size: int = 5,
    stride: int = 2,
    include_last_window: bool = True,
) -> List[Dict[str, Any]]:
    """
    기존 windowed_dialogues 생성 규칙과 맞춤.

    - len(dialogue) < 6이면 short sample 1개
    - len(dialogue) >= 6이면 6턴 sliding window 생성
    - prefix_5는 기존 코드에서 주석 처리되어 있었으므로 여기서도 만들지 않음
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if base_window_size <= 0:
        raise ValueError(f"base_window_size must be positive, got {base_window_size}")

    n = len(dialogue)
    samples = []

    if n == 0:
        return samples

    full_window_size = base_window_size + 1

    if n < full_window_size:
        samples.append({
            "orig_idx": orig_idx,
            "window_idx": 0,
            "turn_start_idx": 0,
            "turn_end_idx": n - 1,
            "window_type": "short",
            "dialogue": copy.deepcopy(dialogue),
        })
        return samples

    last_start_idx = n - full_window_size
    start_indices = list(range(0, last_start_idx + 1, stride))

    if include_last_window and start_indices[-1] != last_start_idx:
        start_indices.append(last_start_idx)

    window_idx = 1

    for start_idx in start_indices:
        end_idx = start_idx + full_window_size - 1

        samples.append({
            "orig_idx": orig_idx,
            "window_idx": window_idx,
            "turn_start_idx": start_idx,
            "turn_end_idx": end_idx,
            "window_type": "sliding_6",
            "dialogue": copy.deepcopy(dialogue[start_idx:end_idx + 1]),
        })

        window_idx += 1

    return samples


def build_negative_samples_from_addr0(
    addr0_data: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    addr0.json -> negative sample list.

    addr0은 placeholder가 없으므로:
    - 주소 삽입 없음
    - address_plan 없음
    - span 없음
    """
    outputs = []
    invalid = []

    for orig_idx, item in enumerate(tqdm(addr0_data, desc="build addr0 negative")):
        try:
            if not isinstance(item, dict):
                raise TypeError(
                    f"orig_idx={orig_idx}: item must be dict, got {type(item)}"
                )

            dialogue = item.get("dialogue", [])
            cleaned_dialogue = clean_dialogue(
                dialogue=dialogue,
                orig_idx=orig_idx,
            )

            if not cleaned_dialogue:
                continue

            window_samples = make_prefix5_then_sliding6_samples(
                dialogue=cleaned_dialogue,
                orig_idx=orig_idx,
                base_window_size=BASE_WINDOW_SIZE,
                stride=STRIDE,
                include_last_window=INCLUDE_LAST_WINDOW,
            )

            for sample in window_samples:
                text = build_text_from_dialogue(sample["dialogue"])

                if not text.strip():
                    continue

                out = copy.deepcopy(sample)
                out["text"] = text
                out["spans"] = []
                out["negative_source"] = "addr0"

                outputs.append(out)

        except Exception as e:
            invalid.append({
                "orig_idx": orig_idx,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "item": item,
            })

    return outputs, invalid


def build_negative_dataset(addr0_data):
    """
    방금 준 파일의 triple/quadra/oneline 계열 output과 같은 구조.

    output:
    {
        "aicc": {
            "negative": [...]
        }
    }
    """

    negative_samples, invalid = build_negative_samples_from_addr0(addr0_data)

    final_data = {
        "aicc": {
            "negative": negative_samples,
        }
    }

    return final_data, invalid, len(addr0_data)


def main():


    SAVE_PATH = (
        ADDRESS_DATA_DIR / "final_negative_dataset_addr0.json"
    )

    INVALID_SAVE_PATH = (
        ADDRESS_DATA_DIR / "final_negative_dataset_addr0_invalid.json"
    )

    start_time = time.time()

    addr0_data = read_json(NEGATIVE_ADDR0_PATH)
    addr0_data += read_json(NEGATIVE_ADDR5PLUS0_PATH)

    final_data, invalid, raw_count = build_negative_dataset(addr0_data)

    write_json(final_data, SAVE_PATH)
    write_json(invalid, INVALID_SAVE_PATH)

    end_time = time.time()

    print(f"saved: {SAVE_PATH}")
    print(f"saved invalid: {INVALID_SAVE_PATH}")
    print(f"raw addr0 dialogues: {raw_count}")
    print(f"aicc negative samples: {len(final_data['aicc']['negative'])}")
    print(f"invalid samples: {len(invalid)}")
    print(f"time: {end_time - start_time}sec")


if __name__ == "__main__":
    main()