import json
from collections import Counter

quadra_path = "/data/private/address_bot_3/make_address_insertion/data/template/text_template_quadra.json"

PLACEHOLDERS = [
    "[ADDRESS_A1]",
    "[ADDRESS_A2]",
    "[ADDRESS_B1]",
    "[ADDRESS_B1]",     # 혹시 오타 확인용
    "[ADDRESS_B2]",
    "[ADDRESS_C1]",
    "[ADDRESS_C2]",
    "[ADDRESS_A]",
    "[ADDRESS_B]",
    "[ADDRESS_C]",
    "[ADDRESS_D]",
    "[ADDRESS_1]",
    "[ADDRESS_2]",
    "[ADDRESS]",
]
PLACEHOLDERS = sorted(PLACEHOLDERS, key=len, reverse=True)


def extract_placeholder_occurrences(text):
    out = []
    i = 0

    while i < len(text):
        matched = None

        for ph in PLACEHOLDERS:
            if text.startswith(ph, i):
                matched = ph
                break

        if matched is None:
            i += 1
            continue

        out.append(matched)
        i += len(matched)

    return out


with open(quadra_path, "r", encoding="utf-8") as f:
    templates = json.load(f)

bad = []

for idx, sample in enumerate(templates):
    text = sample.get("text", "")
    plan = sample.get("address_plan", [])

    placeholders = extract_placeholder_occurrences(text)

    if len(placeholders) != len(plan):
        bad.append({
            "idx": idx,
            "placeholder_count": len(placeholders),
            "plan_count": len(plan),
            "placeholders": placeholders,
            "text": text,
            "address_plan": plan,
        })

print(f"total templates: {len(templates)}")
print(f"bad templates: {len(bad)}")

for item in bad[:20]:
    print("=" * 120)
    print(f"idx: {item['idx']}")
    print(f"placeholder_count: {item['placeholder_count']}")
    print(f"plan_count: {item['plan_count']}")
    print(f"placeholders: {item['placeholders']}")
    print(f"text: {item['text']}")
    print("address_plan:")
    for p in item["address_plan"]:
        print(f"  {p}")