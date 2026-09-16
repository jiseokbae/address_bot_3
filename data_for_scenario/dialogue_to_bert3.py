import json
import re
from typing import Any, Dict, List


# [ADDRESS], [ADDRESS_A], [ADDRESS_A1], [ADDRESS_D] 등 전부 탐지
# 예전 오타 가능성 때문에 [ADDRSS_B1]도 허용
PLACEHOLDER_PATTERN = re.compile(r"\[((?:ADDRESS|ADDRSS)(?:_[A-Z0-9]+)?)\]")

# <이혜정:NAME>, <서울특별시:시도명>, <[ADDRESS]:ADDRESS> 처리
ANGLE_TAG_PATTERN = re.compile(r"<([^<>]+)>")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_placeholder(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return PLACEHOLDER_PATTERN.search(text) is not None


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


def select_aicc_turn_text(turn: Dict[str, Any]) -> str:
    """
    AICC raw turn에서 최종 text로 쓸 문자열 선택.

    규칙:
    1. tagged_text에 ADDRESS placeholder가 있으면 tagged_text 사용
       - 단, <surface:LABEL>은 surface만 남김.
    2. tagged_text에 ADDRESS placeholder가 없으면 original_text 사용
    3. original_text가 없으면 fallback으로 tagged_text를 정리해서 사용
    """

    tagged_text = turn.get("tagged_text")
    original_text = turn.get("original_text")

    if isinstance(tagged_text, str) and has_placeholder(tagged_text):
        return clean_angle_tags(tagged_text)

    if isinstance(original_text, str):
        return original_text

    if isinstance(tagged_text, str):
        return clean_angle_tags(tagged_text)

    return ""


def validate_turn(turn: Dict[str, Any], sample_idx: int, turn_pos: int) -> None:
    if not isinstance(turn, dict):
        raise TypeError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"turn is not dict, got {type(turn)}"
        )

    if "speaker" not in turn:
        raise KeyError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"missing key 'speaker' in turn={turn}"
        )

    if "text" not in turn:
        raise KeyError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"missing key 'text' in turn={turn}"
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
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"'speaker' must be str, got {type(turn['speaker'])}, "
            f"value={turn['speaker']}"
        )

    if not isinstance(turn["text"], str):
        raise TypeError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"'text' must be str, got {type(turn['text'])}, "
            f"value={turn['text']}"
        )

    if not turn["text"].strip():
        raise ValueError(
            f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
            f"text is empty after strip"
        )


def render_turn(turn: Dict[str, Any], speaker_prefix: bool = False) -> str:
    speaker = turn["speaker"].strip()
    text = turn["text"].strip()

    if speaker_prefix:
        return f"{speaker}: {text}"

    return text


def build_text_from_dialogue(
    dialogue: List[Dict[str, Any]],
    sample_idx: int,
    speaker_prefix: bool = False,
) -> str:
    """
    windowed_dialogues.json 안의 dialogue는 이미 turn["text"]로 정리되어 있음.
    따라서 여기서는 tagged_text/original_text를 다시 보지 않는다.
    """
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

    return "\n".join(rendered_turns)


def split_9_to_18_groups_with_final_speaker_fix(
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    speaker_prefix: bool = False,
):
    grouped = {}
    invalid_samples = []

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
                            f"sample_idx={sample_idx}: "
                            f"sample is not dict, got {type(sample)}"
                        )

                    if "dialogue" not in sample:
                        raise KeyError(
                            f"sample_idx={sample_idx}: missing key 'dialogue'"
                        )

                    dialogue = sample["dialogue"]

                    text = build_text_from_dialogue(
                        dialogue=dialogue,
                        sample_idx=sample_idx,
                        speaker_prefix=speaker_prefix,
                    )

                    new_sample = dict(sample)
                    new_sample["dialogue"] = dialogue
                    new_sample["text"] = text
                    new_sample["has_placeholder"] = has_placeholder(text)

                    bucket = (
                        "has_placeholder"
                        if new_sample["has_placeholder"]
                        else "no_placeholder"
                    )

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


def build_text_from_aicc_raw_dialogue(
    dialogue: List[Dict[str, Any]],
    sample_idx: int,
    speaker_prefix: bool = False,
) -> str:
    """
    addr3 / addr4 / addr5plus_3 / addr5plus_4 같은 raw AICC dialogue를
    text 하나로 변환한다.

    여기서는 turn이 text 필드를 가진 windowed dialogue가 아니라,
    tagged_text / original_text를 가진 raw dialogue라고 본다.

    규칙:
      - tagged_text에 ADDRESS placeholder가 있으면 tagged_text 사용
      - 없으면 original_text 사용
      - tagged_text를 사용할 때는 <surface:LABEL>에서 surface만 남김
    """
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
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
                f"turn must be dict, got {type(turn)}"
            )

        if "speaker" not in turn:
            raise KeyError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
                f"missing key 'speaker'"
            )

        speaker = turn["speaker"]

        if speaker is None:
            raise ValueError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: speaker is None"
            )

        if not isinstance(speaker, str):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
                f"speaker must be str, got {type(speaker)}"
            )

        selected_text = select_aicc_turn_text(turn)

        if not isinstance(selected_text, str):
            raise TypeError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
                f"selected_text must be str, got {type(selected_text)}"
            )

        selected_text = selected_text.strip()

        if not selected_text:
            raise ValueError(
                f"sample_idx={sample_idx}, turn_pos={turn_pos}: "
                f"selected_text is empty"
            )

        speaker = speaker.strip()

        if speaker_prefix:
            rendered = f"{speaker}: {selected_text}"
        else:
            rendered = selected_text

        rendered_turns.append(rendered)

    return "\n".join(rendered_turns)


def parse_aicc_triquad_to_text(
    raw_samples: List[Dict[str, Any]],
    speaker_prefix: bool = False,
) -> List[Dict[str, Any]]:
    """
    aicc triple / quadra 원본 데이터를 샘플 리스트로 변환한다.

    output sample에는 text 필드가 추가된다.
    address_plan 등 기존 sample 필드는 그대로 유지한다.
    """
    if not isinstance(raw_samples, list):
        raise TypeError(f"raw_samples must be list, got {type(raw_samples)}")

    parsed_samples: List[Dict[str, Any]] = []
    invalid_samples: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(raw_samples):
        try:
            if not isinstance(sample, dict):
                raise TypeError(
                    f"sample_idx={sample_idx}: sample must be dict, got {type(sample)}"
                )

            if "dialogue" not in sample:
                raise KeyError(f"sample_idx={sample_idx}: missing key 'dialogue'")

            dialogue = sample["dialogue"]

            text = build_text_from_aicc_raw_dialogue(
                dialogue=dialogue,
                sample_idx=sample_idx,
                speaker_prefix=speaker_prefix,
            )

            new_sample = dict(sample)
            new_sample["text"] = text

            parsed_samples.append(new_sample)

        except Exception as e:
            invalid_samples.append({
                "sample_idx": sample_idx,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "sample": sample,
            })

    return parsed_samples, invalid_samples


def print_grouped_summary(grouped):
    total_has = 0
    total_no = 0

    for source in ["aicc", "gpt", "qwen3"]:
        print(f"\n[{source}]")

        for scenario in ["single", "double", "split"]:
            n_has = len(grouped[source][scenario]["has_placeholder"])
            n_no = len(grouped[source][scenario]["no_placeholder"])

            print(
                source,
                scenario,
                "has:",
                n_has,
                "no:",
                n_no,
                "total:",
                n_has + n_no,
            )

            total_has += n_has
            total_no += n_no

    print("total has:", total_has, "total no:", total_no, "total:", total_has + total_no)


if __name__ == "__main__":
    # =========================================================
    # Path
    # =========================================================

    data_path = "/data/private/address_bot_3/data_for_scenario/windowed_dialogues.json"

    text_template_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template.json"
    )

    text_template_invalid_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template_invalid_samples.json"
    )

    triple_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template_triple.json"
    )

    triple_invalid_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template_triple_invalid_samples.json"
    )

    quadra_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template_quadra.json"
    )

    quadra_invalid_save_path = (
        "/data/private/address_bot_3/make_address_insertion/data/template/"
        "text_template_quadra_invalid_samples.json"
    )

    # =========================================================
    # 1. single / double / split template 생성
    # =========================================================

    data = read_json(data_path)

    keys = ["single", "double", "split"]

    for source in ["qwen3", "gpt", "aicc"]:
        print(f"\n[{source}]")
        for key in keys:
            print(f"  {key}: {len(data[source][key])}")

    grouped, invalid_samples = split_9_to_18_groups_with_final_speaker_fix(data)

    print_grouped_summary(grouped)
    print(f"invalid_samples에 쌓인 에러 개수: {len(invalid_samples)}")

    save_json(grouped, text_template_save_path)
    save_json(invalid_samples, text_template_invalid_save_path)

    print("\n===== SAVED TEXT TEMPLATE =====")
    print("template:", text_template_save_path)
    print("invalid :", text_template_invalid_save_path)

    # =========================================================
    # 2. triple template 생성
    # =========================================================

    tri_data = read_json(
        "/data/private/address_bot_3/data_for_scenario/aicc_data/addr3_relabeled.json"
    )
    tri_data += read_json(
        "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_3_relabeled.json"
    )

    triple_samples, triple_invalid_samples = parse_aicc_triquad_to_text(tri_data)

    save_json(triple_samples, triple_save_path)
    save_json(triple_invalid_samples, triple_invalid_save_path)

    print("\n===== SAVED TRIPLE TEMPLATE =====")
    print("template:", triple_save_path)
    print("invalid :", triple_invalid_save_path)
    print("triple samples:", len(triple_samples))
    print("triple invalid:", len(triple_invalid_samples))

    # =========================================================
    # 3. quadra template 생성
    # =========================================================

    quad_data = read_json(
        "/data/private/address_bot_3/data_for_scenario/aicc_data/addr4_relabeled.json"
    )
    quad_data += read_json(
        "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_4_relabeled.json"
    )

    quadra_samples, quadra_invalid_samples = parse_aicc_triquad_to_text(quad_data)

    save_json(quadra_samples, quadra_save_path)
    save_json(quadra_invalid_samples, quadra_invalid_save_path)

    print("\n===== SAVED QUADRA TEMPLATE =====")
    print("template:", quadra_save_path)
    print("invalid :", quadra_invalid_save_path)
    print("quadra samples:", len(quadra_samples))
    print("quadra invalid:", len(quadra_invalid_samples))