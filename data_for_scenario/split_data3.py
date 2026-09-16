import json
import re


# [ADDRESS], [ADDRESS_A], [ADDRESS_A1], [ADDRESS_D] 등 매칭
# 예전에 [ADDRSS_B1] 오타 placeholder도 가능성이 있어서 같이 허용
PLACEHOLDER_PATTERN = re.compile(r"\[((?:ADDRESS|ADDRSS)(?:_[A-Z0-9]+)?)\]")

# <이혜정:NAME>, <서울시:시도명>, <[ADDRESS]:ADDRESS> 같은 형태 처리
ANGLE_TAG_PATTERN = re.compile(r"<([^<>]+)>")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_address_placeholder(text):
    if not isinstance(text, str):
        return False
    return PLACEHOLDER_PATTERN.search(text) is not None


def clean_angle_tags(text):
    """
    tagged_text 안의 <surface:LABEL> 형태를 surface만 남긴다.

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
            surface = inner.split(":", 1)[0]
            return surface

        return inner

    return ANGLE_TAG_PATTERN.sub(replace_tag, text)


def get_first_text_by_keys(turn, keys):
    """
    turn에서 keys 순서대로 문자열 필드를 찾는다.
    없으면 빈 문자열 반환.
    """
    for key in keys:
        value = turn.get(key)
        if isinstance(value, str):
            return value
    return ""


def select_turn_text(
    turn,
    tagged_text_key=None,
    raw_text_keys=("text",),
):
    """
    최종 output turn["text"]에 넣을 문자열 선택.

    규칙:
    1. tagged_text_key가 있고 tagged_text 안에 ADDRESS placeholder가 있으면 tagged_text 사용
       - 단, <surface:LABEL> 형태는 surface만 남긴다.
    2. tagged_text 안에 ADDRESS placeholder가 없으면 raw_text_keys 순서대로 원문 필드 사용
       - AICC는 raw_text_keys=("original_text",)
       - GPT는 raw_text_keys=("text",)
    """

    tagged_text = turn.get(tagged_text_key, "") if tagged_text_key is not None else ""

    if isinstance(tagged_text, str) and has_address_placeholder(tagged_text):
        return clean_angle_tags(tagged_text)

    return get_first_text_by_keys(turn, raw_text_keys)


def extract_placeholders_from_dialogue(dialogue):
    """
    이미 정리된 dialogue의 turn["text"] 기준으로 placeholder set 추출.
    """
    found = set()

    for turn in dialogue:
        text = turn.get("text", "")
        matches = PLACEHOLDER_PATTERN.findall(text)

        for m in matches:
            found.add(f"[{m}]")

    return found


def keep_dialogue_only(
    dialogue,
    tagged_text_key=None,
    raw_text_keys=("text",),
):
    """
    최종 dialogue 양식으로 정리.

    output:
    [
      {
        "turn_idx": ...,
        "speaker": ...,
        "text": ...
      },
      ...
    ]
    """
    new_dialogue = []

    for turn in dialogue:
        selected_text = select_turn_text(
            turn=turn,
            tagged_text_key=tagged_text_key,
            raw_text_keys=raw_text_keys,
        )

        new_dialogue.append({
            "turn_idx": turn.get("turn_idx"),
            "speaker": turn.get("speaker"),
            "text": selected_text,
        })

    return new_dialogue


def split_dialogues(
    data,
    tagged_text_key=None,
    raw_text_keys=("text",),
):
    """
    single / double / split 분류.

    GPT:
      tagged_text_key=None
      raw_text_keys=("text",)

    AICC:
      tagged_text_key="tagged_text"
      raw_text_keys=("original_text",)
    """
    result = {
        "single": [],
        "double": [],
        "split": [],
    }

    dropped = []

    for item_idx, item in enumerate(data):
        dialogue = item.get("dialogue", [])
        if not dialogue:
            continue

        cleaned_dialogue = keep_dialogue_only(
            dialogue=dialogue,
            tagged_text_key=tagged_text_key,
            raw_text_keys=raw_text_keys,
        )

        placeholders = extract_placeholders_from_dialogue(cleaned_dialogue)

        if placeholders == {"[ADDRESS]"}:
            result["single"].append(cleaned_dialogue)

        elif placeholders == {"[ADDRESS_A]", "[ADDRESS_B]"}:
            result["double"].append(cleaned_dialogue)

        elif placeholders == {"[ADDRESS_1]", "[ADDRESS_2]"}:
            result["split"].append(cleaned_dialogue)

        else:
            dropped.append({
                "item_idx": item_idx,
                "placeholders": sorted(placeholders),
                "dialogue": cleaned_dialogue,
            })

    return result, dropped


def summarize_result(title, result, dropped=None):
    print(f"\n===== {title} =====")
    print("single:", len(result["single"]))
    print("double:", len(result["double"]))
    print("split :", len(result["split"]))

    if dropped is not None:
        print("dropped:", len(dropped))


if __name__ == "__main__":
    # =========================================================
    # 0. Path
    # =========================================================

    gpt_path = "/data/private/address_bot_3/data_for_scenario/gpt_data/gpt_dialogues.json"

    aicc_addr2_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr2_relabeled.json"
    aicc_addr5plus_2_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_2_relabeled.json"

    aicc_addr1_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr1.json"
    aicc_addr5plus_1_path = "/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_1.json"

    save_path = "/data/private/address_bot_3/data_for_scenario/dialogues_by_source_gptaicc.json"
    dropped_save_path = "/data/private/address_bot_3/data_for_scenario/dialogues_by_source_gptaicc_dropped.json"

    # =========================================================
    # 1. Load
    # =========================================================

    gpt_data = read_json(gpt_path)

    aicc_addr2_data = read_json(aicc_addr2_path)
    aicc_addr2_data += read_json(aicc_addr5plus_2_path)

    aicc_addr1_data = read_json(aicc_addr1_path)
    aicc_addr1_data += read_json(aicc_addr5plus_1_path)

    # =========================================================
    # 2. GPT 데이터 처리
    #    GPT는 input key가 text
    # =========================================================

    gpt_result, gpt_dropped = split_dialogues(
        gpt_data,
        tagged_text_key=None,
        raw_text_keys=("text",),
    )

    # =========================================================
    # 3. AICC addr2 데이터 처리
    #
    #    AICC turn 구조:
    #      - tagged_text
    #      - original_text
    #
    #    규칙:
    #      - tagged_text에 ADDRESS placeholder가 있으면 tagged_text 사용
    #      - 없으면 original_text 사용
    # =========================================================

    aicc_addr2_result, aicc_addr2_dropped = split_dialogues(
        aicc_addr2_data,
        tagged_text_key="tagged_text",
        raw_text_keys=("original_text",),
    )

    # =========================================================
    # 4. AICC addr1 데이터 처리
    #
    #    기존 코드에서는 addr1은 전부 single로 넣었음.
    #    여기서도 동일하게 single에 넣는다.
    #
    #    단, text 선택 규칙은 동일:
    #      - tagged_text에 ADDRESS placeholder 있으면 tagged_text 사용
    #      - 없으면 original_text 사용
    # =========================================================

    aicc_addr1_single = []
    aicc_addr1_dropped = []

    for item_idx, item in enumerate(aicc_addr1_data):
        dialogue = item.get("dialogue", [])
        if not dialogue:
            continue

        cleaned_dialogue = keep_dialogue_only(
            dialogue=dialogue,
            tagged_text_key="tagged_text",
            raw_text_keys=("original_text",),
        )

        placeholders = extract_placeholders_from_dialogue(cleaned_dialogue)

        # 기존 코드와 동일하게 addr1은 전부 single로 넣는다.
        aicc_addr1_single.append(cleaned_dialogue)

        # 다만 확인용으로 [ADDRESS]가 없는 케이스는 dropped 로그에 기록만 한다.
        # final_result에서는 제외하지 않음.
        if placeholders != {"[ADDRESS]"}:
            aicc_addr1_dropped.append({
                "item_idx": item_idx,
                "placeholders": sorted(placeholders),
                "dialogue": cleaned_dialogue,
            })

    # =========================================================
    # 5. AICC 최종 merge
    # =========================================================

    aicc_result = {
        "single": aicc_addr2_result["single"] + aicc_addr1_single,
        "double": aicc_addr2_result["double"],
        "split": aicc_addr2_result["split"],
    }

    # =========================================================
    # 6. Summary
    # =========================================================

    summarize_result(
        title="GPT DATA",
        result=gpt_result,
        dropped=gpt_dropped,
    )

    summarize_result(
        title="AICC ADDR2 DATA",
        result=aicc_addr2_result,
        dropped=aicc_addr2_dropped,
    )

    print("\n===== AICC ADDR1 DATA =====")
    print("single:", len(aicc_addr1_single))
    print("dropped_log_only:", len(aicc_addr1_dropped))

    print("\n===== AICC TOTAL =====")
    print("single:", len(aicc_result["single"]))
    print("double:", len(aicc_result["double"]))
    print("split :", len(aicc_result["split"]))

    # =========================================================
    # 7. Save final result
    # =========================================================

    final_result = {
        "gpt": gpt_result,
        "aicc": aicc_result,
    }

    save_json(save_path, final_result)

    # =========================================================
    # 8. Save dropped logs
    # =========================================================

    dropped_result = {
        "gpt": gpt_dropped,
        "aicc_addr2": aicc_addr2_dropped,
        "aicc_addr1_log_only": aicc_addr1_dropped,
    }

    save_json(dropped_save_path, dropped_result)

    print("\n===== SAVED =====")
    print("final:", save_path)
    print("dropped:", dropped_save_path)