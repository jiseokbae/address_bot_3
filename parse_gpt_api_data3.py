import json

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def dialogue_to_string(data):
    return "\nb".join(turn["original_text"] for turn in data["dialogue"])

def extract_dialogues(data):
    return [item["dialogue"] for item in data]

import re
import copy

PATTERN = re.compile(r"\(\s*(\[(?:ADDRESS(?:_(?:1|2|OLD|NEW))?)\])\s*\)")

def clean_dialogues(data):
    new_data = copy.deepcopy(data)
    changed_idxs = []

    for idx, item in enumerate(new_data):
        changed = False

        for turn in item:
            text = turn["text"]
            new_text = PATTERN.sub(r"\1", text)

            if new_text != text:
                turn["text"] = new_text
                changed = True

        if changed:
            changed_idxs.append(idx)

    return new_data, changed_idxs

def merge_fixed_cases(original_data, fixed_cases):
    merged = copy.deepcopy(original_data)

    for item in fixed_cases:
        idx = item["idx"]
        merged[idx]["dialogue"] = item["dialogue"]

    return merged

def normalize_address_tokens(data):
    for item in data:
        for turn in item["dialogue"]:
            text = turn["text"]
            text = text.replace("[ADDRESS_OLD]", "[ADDRESS_A]")
            text = text.replace("[ADDRESS_NEW]", "[ADDRESS_B]")
            turn["text"] = text
    return data

from tqdm import tqdm
if __name__ == "__main__":
    path_1 = "/data/private/address_bot/data_for_scenario/gpt_api_data/final_dialogues_100_100.json"
    path_2 = "/data/private/address_bot/data_for_scenario/gpt_api_data/final_dialogues_900_900.json"
    path_3 = "/data/private/address_bot/data_for_scenario/gpt_api_data/final_dialogues_4000_4000.json"

    data_1 = read_json(path_1)
    data_2 = read_json(path_2)
    data_3 = read_json(path_3)

    data = data_1 + data_2 + data_3

    print(len(data_1))
    print(len(data_2))
    print(len(data_3))

    
    fixed_cases = read_json(
        "/data/private/address_bot/data_for_scenario/gpt_api_data/problem_cases_filtered.json"
    )

    merged_data = merge_fixed_cases(data, fixed_cases)
    print(len(merged_data))
    merged_data = normalize_address_tokens(merged_data)

    save_json(
        "/data/private/address_bot_3/data_for_scenario/gpt_api_data/gpt_dialogues.json",
        merged_data
    )