import json
from pprint import pprint

NEGATIVE_PATH = "/data/private/address_bot_3/make_address_insertion/data/final_negative_dataset_addr0.json"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = read_json(NEGATIVE_PATH)

    samples = data["aicc"]["negative"]

    print(f"total negative samples: {len(samples)}")

    for idx in [0, 1]:
        print("=" * 120)
        print(f"[sample {idx}]")

        if idx >= len(samples):
            print(f"sample {idx} does not exist")
            continue

        pprint(samples[idx], width=160, sort_dicts=False)