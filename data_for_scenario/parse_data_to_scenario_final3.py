import json
import copy
from typing import Dict, List, Any, Tuple, Optional

KEYS = ["single", "double", "split"]

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_prefix5_then_sliding6_samples(
    dialogue: List[Dict[str, Any]],
    orig_idx: int,
    base_window_size: int = 5,
    stride: int = 1,
    include_last_window: bool = True,
) -> List[Dict[str, Any]]:
    """
    규칙:
    - len(dialogue) < 6이면 있는 만큼만 한 개 생성
    - len(dialogue) >= 6이면
      1) 먼저 5턴짜리 [0~4] 생성
      2) 이후 6턴짜리 sliding window 생성
         [0~5], [stride~stride+5], ...
      3) include_last_window=True이면 마지막 turn이 빠지지 않도록 마지막 window 추가
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

    # 6턴이 안 되면 있는 만큼만 한 개 생성
    if n < full_window_size:
        samples.append(
            {
                "orig_idx": orig_idx,
                "window_idx": 0,
                "turn_start_idx": 0,
                "turn_end_idx": n - 1,
                "window_type": "short",
                "dialogue": copy.deepcopy(dialogue),
            }
        )
        return samples

    # 먼저 5턴짜리 prefix sample 생성: [0~4]
    # samples.append(
    #     {
    #         "orig_idx": orig_idx,
    #         "window_idx": 0,
    #         "turn_start_idx": 0,
    #         "turn_end_idx": base_window_size - 1,
    #         "window_type": "prefix_5",
    #         "dialogue": copy.deepcopy(dialogue[0:base_window_size]),
    #     }
    # )

    # 이후 6턴짜리 sliding window 생성
    last_start_idx = n - full_window_size
    start_indices = list(range(0, last_start_idx + 1, stride))

    # stride 때문에 마지막 turn이 빠지면 마지막 window 강제 추가
    if include_last_window and start_indices[-1] != last_start_idx:
        start_indices.append(last_start_idx)

    window_idx = 1

    for start_idx in start_indices:
        end_idx = start_idx + full_window_size - 1

        samples.append(
            {
                "orig_idx": orig_idx,
                "window_idx": window_idx,
                "turn_start_idx": start_idx,
                "turn_end_idx": end_idx,
                "window_type": "sliding_6",
                "dialogue": copy.deepcopy(dialogue[start_idx:end_idx + 1]),
            }
        )

        window_idx += 1

    return samples


def build_new_data(
    qwen_data,
    other_data,
    base_window_size: int = 5,
    stride: int = 2,
    include_last_window: bool = True,
):
    new_data = {
        "aicc": {key: [] for key in KEYS},
        "gpt": {key: [] for key in KEYS},
        "qwen3": {key: [] for key in KEYS},
    }

    # 1) qwen3
    # qwen_data[key][idx]["dialogue"]
    for key in KEYS:
        if key not in qwen_data:
            continue

        for orig_idx, item in enumerate(qwen_data[key]):
            if not isinstance(item, dict):
                continue

            dialogue = item.get("dialogue", [])
            if not isinstance(dialogue, list) or len(dialogue) == 0:
                continue

            samples = make_prefix5_then_sliding6_samples(
                dialogue=dialogue,
                orig_idx=orig_idx,
                base_window_size=base_window_size,
                stride=stride,
                include_last_window=include_last_window,
            )

            new_data["qwen3"][key].extend(samples)

    # 2) gpt
    # other_data["gpt"][key][idx] == dialogue
    if "gpt" in other_data:
        for key in KEYS:
            if key not in other_data["gpt"]:
                continue

            for orig_idx, dialogue in enumerate(other_data["gpt"][key]):
                if not isinstance(dialogue, list) or len(dialogue) == 0:
                    continue

                samples = make_prefix5_then_sliding6_samples(
                    dialogue=dialogue,
                    orig_idx=orig_idx,
                    base_window_size=base_window_size,
                    stride=stride,
                    include_last_window=include_last_window,
                )

                new_data["gpt"][key].extend(samples)

    # 3) aicc
    # other_data["aicc"][key][idx] == dialogue
    if "aicc" in other_data:
        for key in KEYS:
            if key not in other_data["aicc"]:
                continue

            for orig_idx, dialogue in enumerate(other_data["aicc"][key]):
                if not isinstance(dialogue, list) or len(dialogue) == 0:
                    continue

                samples = make_prefix5_then_sliding6_samples(
                    dialogue=dialogue,
                    orig_idx=orig_idx,
                    base_window_size=base_window_size,
                    stride=stride,
                    include_last_window=include_last_window,
                )

                new_data["aicc"][key].extend(samples)

    return new_data

if __name__ == "__main__":
    qwen_path = "/data/private/address_bot/data_for_scenario/augment/augmented_by_qwen3/qwen_augmented_num50000_S42_FS6.json"
    other_path = "/data/private/address_bot_3/data_for_scenario/dialogues_by_source_gptaicc.json"
    save_path = "/data/private/address_bot_3/data_for_scenario/windowed_dialogues.json"

    qwen_data = read_json(qwen_path)
    other_data = read_json(other_path)

    new_data = build_new_data(qwen_data, other_data)
    write_json(new_data, save_path)

    print("saved to:", save_path)
    for source in ["qwen3", "gpt", "aicc"]:
        print(f"\n[{source}]")
        for key in KEYS:
            print(f"  {key}: {len(new_data[source][key])}")