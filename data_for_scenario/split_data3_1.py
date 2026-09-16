
import json
import re


PLACEHOLDER_PATTERN = re.compile(r"\[(ADDRESS(?:_[A-Z0-9]+)?)\]")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_placeholders_from_dialogue(dialogue, text_key):
    found = set()

    for turn in dialogue:
        text = turn.get(text_key, "")
        matches = PLACEHOLDER_PATTERN.findall(text)
        for m in matches:
            found.add(f"[{m}]")

    return found


def keep_dialogue_only(dialogue, text_key):
    new_dialogue = []

    for turn in dialogue:
        new_dialogue.append({
            "turn_idx": turn.get("turn_idx"),
            "speaker": turn.get("speaker"),
            "text": turn.get(text_key, "")
        })

    return new_dialogue

def split_dialogues(data, text_key):
    result = {
        "single": [],
        "double": [],
        "split": []
    }

    for item in data:
        dialogue = item.get("dialogue", [])
        if not dialogue:
            continue

        placeholders = extract_placeholders_from_dialogue(dialogue, text_key)
        cleaned_dialogue = keep_dialogue_only(dialogue, text_key)

        if placeholders == {"[ADDRESS]"}:
            result["single"].append(cleaned_dialogue)

        elif placeholders == {"[ADDRESS_A]", "[ADDRESS_B]"}:
            result["double"].append(cleaned_dialogue)

        elif placeholders == {"[ADDRESS_1]", "[ADDRESS_2]"}:
            result["split"].append(cleaned_dialogue)

    return result

if __name__=="__main__":
    gpt_data = read_json("/data/private/address_bot_3/data_for_scenario/gpt_data/gpt_dialogues.json")

    aicc_addr2_data = read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr2_relabeled.json")
    aicc_addr2_data += read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_2_relabeled.json")

    aicc_addr1_data = read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr1.json")
    aicc_addr1_data += read_json("/data/private/address_bot_3/data_for_scenario/aicc_data/addr5/addr5plus_1.json")
    
    # 1) GPT 데이터
    gpt_result = split_dialogues(gpt_data, text_key="text")

    # 2) AICC addr2 데이터
    aicc_addr2_result = split_dialogues(aicc_addr2_data, text_key="tagged_text")

    # 3) AICC addr1 데이터: 전부 single
        # 3) AICC addr1 데이터: 전부 single
    aicc_addr1_single = []
    for item in aicc_addr1_data:
        dialogue = item.get("dialogue", [])
        if not dialogue:
            continue
        cleaned_dialogue = keep_dialogue_only(dialogue, text_key="tagged_text")
        aicc_addr1_single.append(cleaned_dialogue)

    aicc_result = {
        "single": aicc_addr2_result["single"] + aicc_addr1_single,
        "double": aicc_addr2_result["double"],
        "split": aicc_addr2_result["split"],
    }

    
    print("===== GPT DATA =====")
    print("single:", len(gpt_result["single"]))
    print("double:", len(gpt_result["double"]))
    print("split :", len(gpt_result["split"]))

    print("\n===== AICC ADDR2 DATA =====")
    print("single:", len(aicc_addr2_result["single"]))
    print("double:", len(aicc_addr2_result["double"]))
    print("split :", len(aicc_addr2_result["split"]))

    print("\n===== AICC ADDR1 DATA =====")
    print("single:", len(aicc_addr1_single))

    print("\n===== AICC TOTAL =====")
    print("single:", len(aicc_result["single"]))
    print("double:", len(aicc_result["double"]))
    print("split :", len(aicc_result["split"]))


    # 최종 merge (gpt + aicc)
    final_result = {
        "gpt": gpt_result,
        "aicc": aicc_result
    }

    save_json(
        "/data/private/address_bot_3/data_for_scenario/dialogues_by_source_gptaicc.json",
        # "/data/private/address_bot/data_for_scenario/dialogues_by_source.json",
        final_result
    )

    