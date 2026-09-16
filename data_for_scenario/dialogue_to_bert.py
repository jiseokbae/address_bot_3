from typing import Dict, List, Any, Tuple, Optional
from copy import deepcopy

PLACEHOLDERS = {
    "[ADDRESS]",
    "[ADDRESS_A]",
    "[ADDRESS_B]",
    "[ADDRESS_1]",
    "[ADDRESS_2]",
    "[ADDRESS_C]",
    "[ADDRESS_D]",
    "[ADDRESS_A1]",
    "[ADDRESS_A2]",
    "[ADDRESS_B1]",
    "[ADDRESS_B2]",
    "[ADDRESS_C1]",
    "[ADDRESS_C2]",
}


def has_placeholder(text: str) -> bool:
    return any(ph in text for ph in PLACEHOLDERS)


def validate_turn(turn: Dict[str, Any], sample_idx: int, turn_pos: int) -> None:
    if not isinstance(turn, dict):
        raise TypeError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: turn is not dict, got {type(turn)}"
        )

    if "speaker" not in turn:
        raise KeyError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: missing key 'speaker' in turn={turn}"
        )

    if "text" not in turn:
        raise KeyError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: missing key 'text' in turn={turn}"
        )

    if turn["speaker"] is None:
        raise ValueError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: speaker is None"
        )

    if turn["text"] is None:
        raise ValueError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: text is None"
        )

    if not isinstance(turn["speaker"], str):
        raise TypeError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: 'speaker' must be str, "
            f"got {type(turn['speaker'])}, value={turn['speaker']}"
        )

    if not isinstance(turn["text"], str):
        raise TypeError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: 'text' must be str, "
            f"got {type(turn['text'])}, value={turn['text']}"
        )

    if not turn["text"].strip():
        raise ValueError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: text is empty after strip"
        )

def render_turn(turn: Dict[str, Any], speaker_prefix: bool = True) -> str:
    speaker = turn["speaker"].strip()
    text = turn["text"].strip()
    return f"{text}" if speaker_prefix else text

def build_text_from_dialogue(
    dialogue: List[Dict[str, Any]],
    sample_idx: int,
    speaker_prefix: bool = False,
) -> Tuple[str, str]:
    if not isinstance(dialogue, list):
        raise TypeError(
            f"sample_idx={sample_idx}: dialogue must be list, got {type(dialogue)}"
        )
    if len(dialogue) == 0:
        raise ValueError(f"sample_idx={sample_idx}: empty dialogue")

    for turn_pos, turn in enumerate(dialogue):
        validate_turn(turn, sample_idx=sample_idx, turn_pos=turn_pos)

    rendered_turns = [
        render_turn(turn, speaker_prefix=speaker_prefix)
        for turn in dialogue
    ]
    text = "\n".join(rendered_turns)
    return text



def split_9_to_18_groups_with_final_speaker_fix(
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    speaker_prefix: bool = False,
):
    grouped = {}
    invalid_samples = []
    speaker_fix_logs = []

    for source, scenario_dict in data.items():
        grouped[source] = {}

        for scenario_key, samples in scenario_dict.items():
            grouped[source][scenario_key] = {
                "has_placeholder": [],
                "no_placeholder": [],
            }

            for sample_idx, sample in enumerate(samples):
                try:
                    if not isinstance(sample, dict):
                        raise TypeError(
                            f"sample_idx={sample_idx}: sample is not dict, got {type(sample)}"
                        )
                    if "dialogue" not in sample:
                        raise KeyError(
                            f"sample_idx={sample_idx}: missing key 'dialogue'"
                        )

                    dialogue=sample["dialogue"]
                    text = build_text_from_dialogue(
                        dialogue=dialogue,
                        sample_idx=sample_idx,
                        speaker_prefix=speaker_prefix,
                    )

                    new_sample = dict(sample)
                    new_sample["dialogue"] = dialogue
                    new_sample["text"] = text
                    new_sample["has_placeholder"] = has_placeholder(text)

                    bucket = "has_placeholder" if new_sample["has_placeholder"] else "no_placeholder"
                    grouped[source][scenario_key][bucket].append(new_sample)

                except Exception as e:
                    invalid_samples.append({
                        "source": source,
                        "scenario_key": scenario_key,
                        "sample_idx": sample_idx,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "sample": sample,
                    })

    return grouped, invalid_samples



def build_text_from_tagged_dialogue(
    dialogue: List[Dict[str, Any]],
    sample_idx: int,
    speaker_prefix: bool = False,
) -> Dict[str, str]:
    if not isinstance(dialogue, list):
        raise TypeError(
            f"sample_idx={sample_idx}: dialogue must be list, got {type(dialogue)}"
        )

    if len(dialogue) == 0:
        raise ValueError(f"sample_idx={sample_idx}: empty dialogue")

    rendered_turns = []

    for turn_pos, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: turn must be dict, got {type(turn)}"
            )

        if "speaker" not in turn:
            raise KeyError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: missing key 'speaker'"
            )

        if "tagged_text" not in turn:
            raise KeyError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: missing key 'tagged_text'"
            )

        speaker = turn["speaker"]
        tagged_text = turn["tagged_text"]

        if speaker is None:
            raise ValueError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: speaker is None"
            )

        if not isinstance(speaker, str):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: speaker must be str, got {type(speaker)}"
            )

        if not isinstance(speaker, str):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: speaker must be str, got {type(speaker)}"
            )

        if tagged_text is None:
            raise ValueError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: tagged_text is None"
            )

        if not isinstance(tagged_text, str):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: tagged_text must be str, got {type(tagged_text)}"
            )

        speaker = speaker.strip()
        tagged_text = tagged_text.strip()

        if speaker_prefix:
            rendered = f"{tagged_text}"
        else:
            rendered = tagged_text

        rendered_turns.append(rendered)
        text = "\n".join(rendered_turns)

    return text

def parse_aicc_triquad_to_text(
    raw_samples: List[Dict[str, Any]],
    speaker_prefix: bool = False,
) -> List[Dict[str, Any]]:
    """
    aicc triple 원본 데이터를 샘플 리스트로 변환한다.

    tagged_text를 사용한다.
    grouped['aicc']['triple'][...] 에 넣기 전 단계의 리스트를 반환한다.
    """
    if not isinstance(raw_samples, list):
        raise TypeError(f"raw_samples must be list, got {type(raw_samples)}")

    parsed_samples: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            raise TypeError(
                f"sample_idx={sample_idx}: sample must be dict, got {type(sample)}"
            )

        if "dialogue" not in sample:
            raise KeyError(f"sample_idx={sample_idx}: missing key 'dialogue'")

        dialogue = sample["dialogue"]

        text = build_text_from_tagged_dialogue(
            dialogue=dialogue,
            sample_idx=sample_idx,
            speaker_prefix=speaker_prefix,
        )

        new_sample = dict(sample)
        new_sample["text"] = text

        parsed_samples.append(new_sample)

    return parsed_samples

import json
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    data_path = "/data/private/address_bot_3/data_for_scenario/windowed_dialogues.json"

    data = read_json(data_path)

    KEYS = ["single", "double", "split"]
    for source in ["qwen3", "gpt", "aicc"]:
        print(f"\n[{source}]")
        for key in KEYS:
            print(f"  {key}: {len(data[source][key])}")

    total_has = 0
    total_no = 0
    # print(data['aicc']['single'][1])
    grouped, invalid_samples = split_9_to_18_groups_with_final_speaker_fix(data)
    for source in ["aicc", "gpt", "qwen3"]:
        for scenario in ["single", "double", "split"]:
            n_has = len(grouped[source][scenario]["has_placeholder"])
            n_no = len(grouped[source][scenario]["no_placeholder"])
            print(source, scenario, "has:", n_has, "no:", n_no, "total:", n_has + n_no)
            total_has += n_has
            total_no += n_no
    print(grouped['gpt']['single']['no_placeholder'])
    print("total has:", total_has, "total no:", total_no, "total:", total_has + total_no)
    print(f"invalid_samples에 쌓인 에러 개수: {len(invalid_samples)}")

    save_json(grouped, "/data/private/address_bot_3/make_address_insertion/data/template/text_template.json")

    tri_data = read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr3_relabeled.json")
    tri_data += read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_3_relabeled.json")
    triple_samples = parse_aicc_triquad_to_text(tri_data)
    save_json(triple_samples, "/data/private/address_bot_3/make_address_insertion/data/template/text_template_triple.json")

    quad_data = read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr4_relabeled.json")
    quad_data += read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_4_relabeled.json")
    quad_samples = parse_aicc_triquad_to_text(quad_data)
    save_json(quad_samples, "/data/private/address_bot_3/make_address_insertion/data/template/text_template_quadra.json")